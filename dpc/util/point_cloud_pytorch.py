import numpy as np
import torch
import torch.nn.functional as F

import util.drc_pytorch
from util.quaternion_torch import quaternion_rotate
from util.camera_torch import intrinsic_matrix
# from util.point_cloud_distance import *



def multi_expand(inp, axis, num):
    inp_big = inp
    for i in range(num):
        inp_big = torch.unsqueeze(inp_big, axis)
    return inp_big


def pointcloud2voxels(cfg, input_pc, sigma):  # [B,N,3]
    # TODO replace with split or tf.unstack
    x = input_pc[:, :, 0]
    y = input_pc[:, :, 1]
    z = input_pc[:, :, 2]

    vox_size = cfg.vox_size

    rng = torch.linspace(-1.0, 1.0, vox_size)
    xg, yg, zg = torch.meshgrid([rng, rng, rng])  # [G,G,G]

    x_big = multi_expand(x, -1, 3)  # [B,N,1,1,1]
    y_big = multi_expand(y, -1, 3)  # [B,N,1,1,1]
    z_big = multi_expand(z, -1, 3)  # [B,N,1,1,1]

    xg = multi_expand(xg, 0, 2)  # [1,1,G,G,G]
    yg = multi_expand(yg, 0, 2)  # [1,1,G,G,G]
    zg = multi_expand(zg, 0, 2)  # [1,1,G,G,G]

    # squared distance
    sq_distance = torch.square(x_big - xg) + torch.square(y_big - yg) + torch.square(z_big - zg)

    # compute gaussian
    func = torch.exp(-sq_distance / (2.0 * sigma * sigma))  # [B,N,G,G,G]

    # normalise gaussian
    if cfg.pc_normalise_gauss:
        normaliser = torch.sum(func, [2, 3, 4], keepdim=True)
        func /= normaliser
    elif cfg.pc_normalise_gauss_analytical:
        # should work with any grid sizes
        magic_factor = 1.78984352254  # see estimate_gauss_normaliser
        sigma_normalised = sigma * vox_size
        normaliser = 1.0 / (magic_factor * torch.pow(sigma_normalised, 3))
        func *= normaliser

    summed = torch.sum(func, dim=1)  # [B,G,G G]
    voxels = torch.clamp(summed, 0.0, 1.0)
    voxels = torch.unsqueeze(voxels, dim=-1)  # [B,G,G,G,1]

    return voxels


def pointcloud2voxels3d_fast(cfg, pc, rgb):  # [B,N,3]
    vox_size = cfg.vox_size
    if cfg.vox_size_z != -1:
        vox_size_z = cfg.vox_size_z
    else:
        vox_size_z = vox_size

    batch_size = pc.size(0)
    num_points = pc.size(1)

    has_rgb = rgb is not None

    grid_size = 1.0
    half_size = grid_size / 2

    filter_outliers = True
    valid = pc >= -half_size and pc <= half_size
    # valid = tf.reduce_all(valid, axis=-1)

    vox_size_tf = torch.tensor([[[vox_size_z, vox_size, vox_size]]], dtype=torch.float32)
    pc_grid = (pc + half_size) * (vox_size_tf - 1)
    indices_floor = torch.floor(pc_grid)
    indices_int = indices_floor.type_as(torch.int32)
    batch_indices = torch.range(0, batch_size, 1)
    batch_indices = torch.unsqueeze(batch_indices, -1)
    batch_indices = torch.repeat_interleave(batch_indices, torch.tensor([1, num_points]))
    batch_indices = torch.unsqueeze(batch_indices, -1)

    indices = torch.cat([batch_indices, indices_int], axis=2)
    indices = torch.reshape(indices, [-1, 4])

    r = pc_grid - indices_floor  # fractional part
    rr = [1.0 - r, r]

    if filter_outliers:
        valid = torch.reshape(valid, [-1])
        # indices = tf.boolean_mask(indices, valid)
        indices = indices[valid.nonzero()]

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][:, :, 0] * rr[pos[1]][:, :, 1] * rr[pos[2]][:, :, 2]
        updates = torch.reshape(updates_raw, [-1])
        if filter_outliers:
            # updates = tf.boolean_mask(updates, valid)
            # updates = updates[valid.nonzero(), :]
            updates = updates[valid.nonzero()]

        indices_loc = indices
        indices_shift = torch.tensor([[0] + pos])
        num_updates = indices_loc.size(0)
        indices_shift = torch.repeat_interleave(indices_shift, torch.tensor([num_updates, 1]))
        indices_loc = indices_loc + indices_shift

        voxels = torch.zeros([batch_size, vox_size_z, vox_size, vox_size], dtype=torch.float32, requires_grad=True)
        voxels.scatter_(0, indices_loc, updates)
        # voxels = tf.scatter_nd(indices_loc, updates, [batch_size, vox_size_z, vox_size, vox_size])
        # if has_rgb:
        #     if cfg.pc_rgb_stop_points_gradient:
        #         updates_raw = tf.stop_gradient(updates_raw)
        #     updates_rgb = tf.expand_dims(updates_raw, axis=-1) * rgb
        #     updates_rgb = tf.reshape(updates_rgb, [-1, 3])
        #     if filter_outliers:
        #         updates_rgb = tf.boolean_mask(updates_rgb, valid)
        #     voxels_rgb = tf.scatter_nd(indices_loc, updates_rgb, [batch_size, vox_size_z, vox_size, vox_size, 3])
        # else:
        voxels_rgb = None

        return voxels, voxels_rgb

    voxels = []
    voxels_rgb = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                vx, vx_rgb = interpolate_scatter3d([k, j, i])
                voxels.append(vx)
                voxels_rgb.append(vx_rgb)

    voxels = sum(voxels)
    voxels_rgb = sum(voxels_rgb) if has_rgb else None

    return voxels, voxels_rgb



def smoothen_voxels3d(cfg, voxels, kernel):
    """
    assume the input voxels shape is [batch, d, h, w, channel],\n
    first convert it to [batch, channel. d, h, w], then re-convert it before return
    """
    # removed this step if the input voxels is already in [batch, channel, d, h, w]
    voxels = voxels.permute((0,4,1,2,3))
    
    padding_size = int((cfg.pc_gauss_kernel_size-1)/2)
    # convolute throught different dims
    voxels = torch.nn.functional.conv3d(voxels, kernel[0], stride=(1,1,1), padding=(10,0,0))
    voxels = torch.nn.functional.conv3d(voxels, kernel[1], stride=(1,1,1), padding=(0,10,0))
    voxels = torch.nn.functional.conv3d(voxels, kernel[2], stride=(1,1,1), padding=(0,0,10))

    # removed this step if the expected output is [batch, channel, d, h, w]
    voxels = voxels.permute((0,2,3,4,1))

    return voxels


def convolve_rgb(cfg, voxels_rgb, kernel):
    pass
#     channels = [voxels_rgb[:, :, :, :, k:k+1] for k in range(3)]
#     for krnl in kernel:
#         for i in range(3):
#             channels[i] = tf.nn.conv3d(channels[i], krnl, [1, 1, 1, 1, 1], padding="SAME")
#     out = tf.concat(channels, axis=4)
#     return out


def pc_perspective_transform(cfg, point_cloud,
                             transform, predicted_translation=None,
                             focal_length=None):
    """
    :param cfg:
    :param point_cloud: [B, N, 3]
    :param transform: [B, 4] if quaternion or [B, 4, 4] if camera matrix
    :param predicted_translation: [B, 3] translation vector
    :return:
    """
    camera_distance = cfg.camera_distance

    if focal_length is None:
        focal_length = cfg.focal_length
    else:
        focal_length = torch.unsqueeze(focal_length, dim=-1)

    if cfg.pose_quaternion:
        pc2 = quaternion_rotate(point_cloud, transform)

        if predicted_translation is not None:
            predicted_translation = torch.unsqueeze(predicted_translation, dim=1)
            pc2 += predicted_translation

        # xs = tf.slice(pc2, [0, 0, 2], [-1, -1, 1])
        # ys = tf.slice(pc2, [0, 0, 1], [-1, -1, 1])
        # zs = tf.slice(pc2, [0, 0, 0], [-1, -1, 1])
        xs = pc2[:, :, 2:3]
        ys = pc2[:, :, 1:2]
        zs = pc2[:, :, 0:1]

        # translation part of extrinsic camera
        zs += camera_distance
        # intrinsic transform
        xs *= focal_length
        ys *= focal_length
    else:
        # xyz1 = tf.pad(point_cloud, tf.constant([[0, 0], [0, 0], [0, 1]]), "CONSTANT", constant_values=1.0)
        xyz1 = F.pad(point_cloud, (0, 0, 0, 0, 0, 1), "CONSTANT", value=1.0)

        extrinsic = transform
        intr = intrinsic_matrix(cfg, dims=4)
        intrinsic = torch.tensor(intr)
        intrinsic = torch.unsqueeze(intrinsic, dim=0)
        intrinsic = torch.repeat_interleave(intrinsic, [extrinsic.size(0), 1, 1])
        full_cam_matrix = torch.matmul(intrinsic, extrinsic)

        pc2 = torch.matmul(xyz1, torch.transpose(full_cam_matrix, [0, 2, 1]))

        # TODO unstack instead of split
        # xs = tf.slice(pc2, [0, 0, 2], [-1, -1, 1])
        # ys = tf.slice(pc2, [0, 0, 1], [-1, -1, 1])
        # zs = tf.slice(pc2, [0, 0, 0], [-1, -1, 1])
        xs = pc2[:, :, 2:3]
        ys = pc2[:, :, 1:2]
        zs = pc2[:, :, 0:1]

    xs /= zs
    ys /= zs

    zs -= camera_distance
    if predicted_translation is not None:
        # zt = tf.slice(predicted_translation, [0, 0, 0], [-1, -1, 1])
        zt = predicted_translation[:, :, 0:1]
        zs -= zt

    xyz2 = torch.cat([zs, ys, xs], dim=2)
    return xyz2


def pointcloud_project(cfg, point_cloud, transform, sigma):
    tr_pc = pc_perspective_transform(cfg, point_cloud, transform)
    voxels = pointcloud2voxels(cfg, tr_pc, sigma)
    voxels = torch.transpose(voxels, [0, 2, 1, 3, 4])

    proj, probs = util.drc_pytorch.drc_projection(voxels, cfg)
    proj = torch.flip(proj, [1])
    return proj, voxels


def pointcloud_project_fast(cfg, point_cloud, transform, predicted_translation,
                            all_rgb, kernel=None, scaling_factor=None, focal_length=None):
    has_rgb = all_rgb is not None

    tr_pc = pc_perspective_transform(cfg, point_cloud,
                                     transform, predicted_translation,
                                     focal_length)
    voxels, voxels_rgb = pointcloud2voxels3d_fast(cfg, tr_pc, all_rgb)
    voxels = torch.unsqueeze(voxels, dim=-1)
    voxels_raw = voxels

    voxels = torch.clamp(voxels, 0.0, 1.0)

    if kernel is not None:
        voxels = smoothen_voxels3d(cfg, voxels, kernel)
        if has_rgb:
            if not cfg.pc_rgb_clip_after_conv:
                voxels_rgb = torch.clamp(voxels_rgb, 0.0, 1.0)
            voxels_rgb = convolve_rgb(cfg, voxels_rgb, kernel)

    if scaling_factor is not None:
        sz = scaling_factor.size(0)
        scaling_factor = torch.reshape(scaling_factor, [sz, 1, 1, 1, 1])
        voxels = voxels * scaling_factor
        voxels = torch.clamp(voxels, 0.0, 1.0)

    # if has_rgb:
    #     if cfg.pc_rgb_divide_by_occupancies:
    #         voxels_div = tf.stop_gradient(voxels_raw)
    #         voxels_div = smoothen_voxels3d(cfg, voxels_div, kernel)
    #         voxels_rgb = voxels_rgb / (voxels_div + cfg.pc_rgb_divide_by_occupancies_epsilon)
    #
    #     if cfg.pc_rgb_clip_after_conv:
    #         voxels_rgb = tf.clip_by_value(voxels_rgb, 0.0, 1.0)

    if cfg.ptn_max_projection:
        proj = torch.max(voxels, 1)
        drc_probs = None
        proj_depth = None
    else:
        proj, drc_probs = util.drc_pytorch.drc_projection(voxels, cfg)
        drc_probs = torch.flip(drc_probs, 2)
        proj_depth = util.drc_pytorch.drc_depth_projection(drc_probs, cfg)

    # proj = tf.reverse(proj, [1])
    proj = torch.flip(proj, 1)

    if voxels_rgb is not None:
        voxels_rgb = torch.flip(voxels_rgb, 2)
        proj_rgb = util.drc_pytorch.project_volume_rgb_integral(cfg, drc_probs, voxels_rgb)
    else:
        proj_rgb = None

    output = {
        "proj": proj,
        "voxels": voxels,
        "tr_pc": tr_pc,
        "voxels_rgb": voxels_rgb,
        "proj_rgb": proj_rgb,
        "drc_probs": drc_probs,
        "proj_depth": proj_depth
    }
    return output


def pc_point_dropout(points, rgb, keep_prob):
    # shape = points.shape.as_list()
    num_input_points = points.size(1)
    batch_size = points.size(0)
    num_channels = points.size(2)
    num_output_points = int(num_input_points * keep_prob)

    out_points = torch.empty([batch_size, num_output_points, num_channels])
    for i in range(batch_size):
        cur_ind = np.random.choice(num_input_points, num_output_points, replace=False)
        out_points[i] = points[i][cur_ind]

    # def sampler(num_output_points_np):
    #     all_inds = []
    #     for k in range(batch_size):
    #         ind = np.random.choice(num_input_points, num_output_points_np, replace=False)
    #         ind = np.expand_dims(ind, axis=-1)
    #         ks = np.ones_like(ind) * k
    #         inds = np.concatenate((ks, ind), axis=1)
    #         all_inds.append(np.expand_dims(inds, 0))
    #     return np.concatenate(tuple(all_inds), 0).astype(np.int64)
    #
    # selected_indices = sampler([num_output_points])
    # out_points = tf.gather_nd(points, selected_indices)
    # out_points = tf.reshape(out_points, [batch_size, num_output_points, num_channels])
    # if rgb is not None:
    #     num_rgb_channels = rgb.shape.as_list()[2]
    #     out_rgb = tf.gather_nd(rgb, selected_indices)
    #     out_rgb = tf.reshape(out_rgb, [batch_size, num_output_points, num_rgb_channels])
    # else:
    out_rgb = None
    return out_points, out_rgb


def subsample_points(xyz, num_points):
    idxs = np.random.choice(xyz.shape[0], num_points)
    xyz_s = xyz[idxs, :]
    return xyz_s

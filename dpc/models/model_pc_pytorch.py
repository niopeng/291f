import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models.model_base_pytorch import ModelBase, pool_single_view

# from util.losses import add_drc_loss, add_proj_rgb_loss, add_proj_depth_loss
from util.point_cloud_pytorch import pointcloud_project, pointcloud_project_fast, \
    pc_point_dropout
from util.gauss_kernel_pytorch import gauss_smoothen_image, smoothing_kernel
from util.quaternion_torch import \
    quaternion_multiply as q_mul,\
    quaternion_normalise as q_norm,\
    quaternion_rotate as q_rotate,\
    quaternion_conjugate as q_conj

from nets.net_factory_pytorch import get_network


# slim = tf.contrib.slim


# def tf_repeat_0(input, num):
#     orig_shape = input.shape
#     e = tf.expand_dims(input, axis=1)
#     tiler = [1 for _ in range(len(orig_shape)+1)]
#     tiler[1] = num
#     tiled = tf.tile(e, tiler)
#     new_shape = [-1]
#     new_shape.extend(orig_shape[1:])
#     final = tf.reshape(tiled, new_shape)
#     return final
def tf_repeat_0(input, num):
    return torch.repeat_interleave(input, num, dim=0)



def get_smooth_sigma(cfg, global_step):
    num_steps = cfg.max_number_of_steps
    diff = (cfg.pc_relative_sigma_end - cfg.pc_relative_sigma)
    sigma_rel = cfg.pc_relative_sigma + global_step / num_steps * diff
    # sigma_rel = tf.cast(sigma_rel, tf.float32)
    # sigma_rel = sigma_rel.type(torch.FloatTensor)
    return float(sigma_rel)


def get_dropout_prob(cfg, global_step):
    if not cfg.pc_point_dropout_scheduled:
        return cfg.pc_point_dropout

    exp_schedule = cfg.pc_point_dropout_exponential_schedule
    num_steps = cfg.max_number_of_steps
    keep_prob_start = cfg.pc_point_dropout
    keep_prob_end = 1.0
    start_step = cfg.pc_point_dropout_start_step
    end_step = cfg.pc_point_dropout_end_step
    # global_step = tf.cast(global_step, dtype=tf.float32)

    x = global_step / num_steps
    k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
    b = keep_prob_start - k * start_step
    if exp_schedule:
        alpha = np.log(keep_prob_end / keep_prob_start)
        keep_prob = keep_prob_start * np.exp(alpha * x)
    else:
        keep_prob = k * x + b
    keep_prob = np.clip(keep_prob, keep_prob_start, keep_prob_end)
    # keep_prob = tf.reshape(keep_prob, [])
    return keep_prob


def get_st_global_scale(cfg, global_step):
    num_steps = cfg.max_number_of_steps
    keep_prob_start = 0.0
    keep_prob_end = 1.0
    start_step = 0
    end_step = 0.1
    # global_step = tf.cast(global_step, dtype=tf.float32)
    x = global_step / num_steps
    k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
    b = keep_prob_start - k * start_step
    keep_prob = k * x + b
    # keep_prob = tf.clip_by_value(keep_prob, keep_prob_start, keep_prob_end)
    keep_prob = np.clip(keep_prob, keep_prob_start, keep_prob_end)
    # keep_prob = tf.reshape(keep_prob, [])
    return keep_prob


def align_predictions(outputs, alignment):
    outputs["points_1"] = q_rotate(outputs["points_1"], alignment)
    outputs["poses"] = q_mul(outputs["poses"], q_conj(alignment))
    outputs["pose_student"] = q_mul(outputs["pose_student"], q_conj(alignment))
    return outputs


class Scale_net(nn.Module):
    '''For creating several pose regressors,
       input shape should be [batch_size, z_dim]'''

    def __init__(self, cfg):
        super(Scale_net, self).__init__()

        self.cfg = cfg
        self.layer = nn.Linear(self.z_dim, 1)

    def forward(self, inputs):
        out = F.sigmoid(self.layer(inputs)) * self.cfg.pc_occupancy_scaling_maximum
        return out


# def predict_scaling_factor(cfg, input):
#     if not cfg.pc_learn_occupancy_scaling:
#         return None
#
#     weight_dict = OrderedDict()
#     weight_dict.update({'0.weight': weights})
#     weight_dict.update({'0.bias': biases})
#
#     pred = nn.Sequential(nn.Linear(input, 1))
#
#     init_stddev = 0.025
#     w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)
#
#     with slim.arg_scope(
#             [slim.fully_connected],
#             weights_initializer=w_init,
#             activation_fn=None):
#         pred = slim.fully_connected(input, 1)
#         pred = tf.sigmoid(pred) * cfg.pc_occupancy_scaling_maximum
#
#     return pred

# def predict_scaling_factor(cfg, input, is_training):
#     if not cfg.pc_learn_occupancy_scaling:
#         return None
#
#     weight_dict = OrderedDict()
#     weight_dict.update({'0.weight': weights})
#     weight_dict.update({'0.bias': biases})
#
#     pred = nn.Sequential(nn.Linear(input, 1))
#
#     init_stddev = 0.025
#     w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)
#
#     with slim.arg_scope(
#             [slim.fully_connected],
#             weights_initializer=w_init,
#             activation_fn=None):
#         pred = slim.fully_connected(input, 1)
#         pred = tf.sigmoid(pred) * cfg.pc_occupancy_scaling_maximum
#
#     if is_training:
#         tf.contrib.summary.scalar("pc_occupancy_scaling_factor", tf.reduce_mean(pred))
#
#     return pred


def predict_focal_length(cfg, input, is_training):
    return None
    # if not cfg.learn_focal_length:
    #     return None
    #
    # init_stddev = 0.025
    # w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)
    #
    # with slim.arg_scope(
    #         [slim.fully_connected],
    #         weights_initializer=w_init,
    #         activation_fn=None):
    #     pred = slim.fully_connected(input, 1)
    #     out = cfg.focal_length_mean + tf.sigmoid(pred) * cfg.focal_length_range
    #
    # if is_training:
    #     tf.contrib.summary.scalar("meta/focal_length", tf.reduce_mean(out))
    #
    # return out


class ModelPointCloud(ModelBase):  # pylint:disable=invalid-name
    """Inherits the generic Im2Vox model class and implements the functions."""

    def __init__(self, cfg, global_step=0, is_train=True):
        super(ModelPointCloud, self).__init__(cfg)
        self._gauss_sigma = None
        self._gauss_kernel = None
        self._sigma_rel = None
        self._global_step = global_step
        self.setup_sigma()
        self.setup_misc()
        self._alignment_to_canonical = None
        if cfg.align_to_canonical and cfg.predict_pose:
            self.set_alignment_to_canonical()

        # added code piece
        self.encoder = get_network(cfg.encoder_name).to(self.device)
        self.decoder = get_network(cfg.decoder_name).to(self.device)
        self.posenet = get_network(cfg.posenet_name).to(self.device)
        self.scalenet = Scale_net(cfg).to(self.device)

        if is_train:
            self.encoder.train()
            self.decoder.train()
            self.posenet.train()
            self.scalenet.train()

            optim_params = []
            for k, v in self.encoder.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            for k, v in self.decoder.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            for k, v in self.posenet.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))

            for k, v in self.scalenet.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))

            self.params = optim_params
            self.optimizer = torch.optim.Adam(optim_params, lr=cfg.learning_rate,
                                                weight_decay=cfg.weight_decay, betas=(0.9, 0.999))

    def setup_sigma(self):
        cfg = self.cfg()
        sigma_rel = get_smooth_sigma(cfg, self._global_step)

        # tf.contrib.summary.scalar("meta/gauss_sigma_rel", sigma_rel)
        self._sigma_rel = sigma_rel
        self._gauss_sigma = sigma_rel / cfg.vox_size
        self._gauss_kernel = smoothing_kernel(cfg, sigma_rel)

    def gauss_sigma(self):
        return self._gauss_sigma

    def gauss_kernel(self):
        return self._gauss_kernel

    def setup_misc(self):
        if self.cfg().pose_student_align_loss:
            num_points = 2000
            sigma = 1.0
            values = np.random.normal(loc=0.0, scale=sigma, size=(num_points, 3))
            values = np.clip(values, -3*sigma, +3*sigma)
            self._pc_for_alignloss = torch.tensor(values, dtype=torch.float32, requires_grad=True)
            # self._pc_for_alignloss = tf.Variable(values, name="point_cloud_for_align_loss",
            #                                      dtype=tf.float32)

    def set_alignment_to_canonical(self):
        exp_dir = self.cfg().checkpoint_dir
        stuff = scipy.io.loadmat(f"{exp_dir}/final_reference_rotation.mat")
        alignment = torch.tensor(stuff["rotation"], torch.float32)
        self._alignment_to_canonical = alignment


    def predict(self, images, predict_for_all=False):
        outputs = {}
        cfg = self._params
        self.encoder.eval()
        self.decoder.eval()
        self.posenet.eval()
        self.scalenet.eval()
        with torch.no_grad():
            enc_outputs = self.encoder(images.permute(0, 2, 3, 1))
            ids = enc_outputs['ids']
            outputs['conv_features'] = enc_outputs['conv_features']
            outputs['ids'] = ids
            outputs['z_latent'] = enc_outputs['z_latent']

            # unsupervised case, case where convnet runs on all views, need to extract the first
            if ids.size(0) != cfg.batch_size:
                ids = pool_single_view(cfg, ids, 0)
            outputs['ids_1'] = ids

            key = 'ids' if predict_for_all else 'ids_1'
            decoder_out = self.decoder(outputs[key], outputs)
            pc = decoder_out['xyz']
            outputs['points_1'] = pc
            outputs['rgb_1'] = decoder_out['rgb']
            outputs['scaling_factor'] = self.scalenet(outputs[key])
            outputs['focal_length'] = predict_focal_length(cfg, outputs['ids'])

            if cfg.predict_pose:
                pose_out = self.posenet(enc_outputs['poses'])
                outputs.update(pose_out)

        self.encoder.train()
        self.decoder.train()
        self.scalenet.train()
        self.posenet.train()

        if self._alignment_to_canonical is not None:
            outputs = align_predictions(outputs, self._alignment_to_canonical)

        return outputs


    def optimize_parameters(self, images, predict_for_all=False):
        outputs = {}
        cfg = self._params

        self.optimizer_G.zero_grad()
        enc_outputs = self.encoder(images)
        ids = enc_outputs['ids']
        outputs['conv_features'] = enc_outputs['conv_features']
        outputs['ids'] = ids
        outputs['z_latent'] = enc_outputs['z_latent']

        # unsupervised case, case where convnet runs on all views, need to extract the first
        if ids.size(0) != cfg.batch_size:
            ids = pool_single_view(cfg, ids, 0)
        outputs['ids_1'] = ids

        key = 'ids' if predict_for_all else 'ids_1'
        decoder_out = self.decoder(outputs[key], outputs)
        pc = decoder_out['xyz']
        outputs['points_1'] = pc
        outputs['rgb_1'] = decoder_out['rgb']
        outputs['scaling_factor'] = self.scalenet(outputs[key])
        outputs['focal_length'] = predict_focal_length(cfg, outputs['ids'])

        if cfg.predict_pose:
            pose_out = self.posenet(enc_outputs['poses'])
            outputs.update(pose_out)

        if self._alignment_to_canonical is not None:
            outputs = align_predictions(outputs, self._alignment_to_canonical)

        loss = self.get_loss(images, outputs)

        loss.backward()
        self.optimizer.step()

        return loss


    # def model_predict(self, images, is_training=False, reuse=False, predict_for_all=False, alignment=None):
    #     outputs = {}
    #     cfg = self._params
    #
    #     if is_training:
    #         self.optimizer_G.zero_grad()
    #         enc_outputs = self.encoder(images)
    #         ids = enc_outputs['ids']
    #         outputs['conv_features'] = enc_outputs['conv_features']
    #         outputs['ids'] = ids
    #         outputs['z_latent'] = enc_outputs['z_latent']
    #
    #         # unsupervised case, case where convnet runs on all views, need to extract the first
    #         if ids.shape.as_list()[0] != cfg.batch_size:
    #             ids = pool_single_view(cfg, ids, 0)
    #         outputs['ids_1'] = ids
    #
    #         key = 'ids' if predict_for_all else 'ids_1'
    #         decoder_out = self.decoder(outputs[key], outputs)
    #         pc = decoder_out['xyz']
    #         outputs['points_1'] = pc
    #         outputs['rgb_1'] = decoder_out['rgb']
    #         outputs['scaling_factor'] = predict_scaling_factor(cfg, outputs[key], is_training)
    #         outputs['focal_length'] = predict_focal_length(cfg, outputs['ids'], is_training)
    #
    #         if cfg.predict_pose:
    #             pose_out = self.posenet(enc_outputs['poses'])
    #             outputs.update(pose_out)
    #     else:
    #         self.encoder.eval()
    #         self.decoder.eval()
    #         self.posenet.eval()
    #         with torch.no_grad():
    #             enc_outputs = self.encoder(images)
    #             ids = enc_outputs['ids']
    #             outputs['conv_features'] = enc_outputs['conv_features']
    #             outputs['ids'] = ids
    #             outputs['z_latent'] = enc_outputs['z_latent']
    #
    #             # unsupervised case, case where convnet runs on all views, need to extract the first
    #             if ids.shape.as_list()[0] != cfg.batch_size:
    #                 ids = pool_single_view(cfg, ids, 0)
    #             outputs['ids_1'] = ids
    #
    #             key = 'ids' if predict_for_all else 'ids_1'
    #             decoder_out = self.decoder(outputs[key], outputs)
    #             pc = decoder_out['xyz']
    #             outputs['points_1'] = pc
    #             outputs['rgb_1'] = decoder_out['rgb']
    #             outputs['scaling_factor'] = predict_scaling_factor(cfg, outputs[key], is_training)
    #             outputs['focal_length'] = predict_focal_length(cfg, outputs['ids'], is_training)
    #
    #             if cfg.predict_pose:
    #                 pose_out = self.posenet(enc_outputs['poses'])
    #                 outputs.update(pose_out)
    #
    #         self.encoder.train()
    #         self.decoder.train()
    #         self.posenet.train()
    #
    #
    #     if self._alignment_to_canonical is not None:
    #         outputs = align_predictions(outputs, self._alignment_to_canonical)
    #
    #     return outputs

    def get_dropout_keep_prob(self):
        cfg = self.cfg()
        return get_dropout_prob(cfg, self._global_step)

    def compute_projection(self, inputs, outputs, is_training):
        cfg = self.cfg()
        all_points = outputs['all_points']
        all_rgb = outputs['all_rgb']

        if cfg.predict_pose:
            camera_pose = outputs['poses']
        else:
            if cfg.pose_quaternion:
                camera_pose = inputs['camera_quaternion']
            else:
                camera_pose = inputs['matrices']

        if is_training and cfg.pc_point_dropout != 1:
            dropout_prob = self.get_dropout_keep_prob()
            # if is_training:
            #     tf.contrib.summary.scalar("meta/pc_point_dropout_prob", dropout_prob)
            all_points, all_rgb = pc_point_dropout(all_points, all_rgb, dropout_prob)

        if cfg.pc_fast:
            predicted_translation = outputs["predicted_translation"] if cfg.predict_translation else None
            proj_out = pointcloud_project_fast(cfg, all_points, camera_pose, predicted_translation,
                                               all_rgb, self.gauss_kernel(),
                                               scaling_factor=outputs['all_scaling_factors'],
                                               focal_length=outputs['all_focal_length'])
            proj = proj_out["proj"]
            outputs["projs_rgb"] = proj_out["proj_rgb"]
            outputs["drc_probs"] = proj_out["drc_probs"]
            outputs["projs_depth"] = proj_out["proj_depth"]
        else:
            proj, voxels = pointcloud_project(cfg, all_points, camera_pose, self.gauss_sigma())
            outputs["projs_rgb"] = None
            outputs["projs_depth"] = None

        outputs['projs'] = proj

        batch_size = outputs['points_1'].size(0)
        outputs['projs_1'] = proj[0:batch_size, :, :, :]

        return outputs

    def replicate_for_multiview(self, tensor):
        cfg = self.cfg()
        new_tensor = tf_repeat_0(tensor, cfg.step_size)
        return new_tensor

    def get_model_fn(self, is_training=True, reuse=False, run_projection=True):
        cfg = self._params

        def model(inputs):
            code = 'images' if cfg.predict_pose else 'images_1'
            outputs = self.model_predict(inputs[code], is_training, reuse)
            pc = outputs['points_1']

            if run_projection:
                all_points = self.replicate_for_multiview(pc)
                num_candidates = cfg.pose_predict_num_candidates
                all_focal_length = None
                if num_candidates > 1:
                    all_points = tf_repeat_0(all_points, num_candidates)
                    if cfg.predict_translation:
                        trans = outputs["predicted_translation"]
                        outputs["predicted_translation"] = tf_repeat_0(trans, num_candidates)
                    focal_length = outputs['focal_length']
                    if focal_length is not None:
                        all_focal_length = tf_repeat_0(focal_length, num_candidates)

                outputs['all_focal_length'] = all_focal_length
                outputs['all_points'] = all_points
                if cfg.pc_learn_occupancy_scaling:
                    all_scaling_factors = self.replicate_for_multiview(outputs['scaling_factor'])
                    if num_candidates > 1:
                        all_scaling_factors = tf_repeat_0(all_scaling_factors, num_candidates)
                else:
                    all_scaling_factors = None
                outputs['all_scaling_factors'] = all_scaling_factors
                if cfg.pc_rgb:
                    all_rgb = self.replicate_for_multiview(outputs['rgb_1'])
                else:
                    all_rgb = None
                outputs['all_rgb'] = all_rgb

                outputs = self.compute_projection(inputs, outputs, is_training)

            return outputs

        return model

    def proj_loss_pose_candidates(self, gt, pred, inputs):
        """
        :param gt: [BATCH*VIEWS, IM_SIZE, IM_SIZE, 1]
        :param pred: [BATCH*VIEWS*CANDIDATES, IM_SIZE, IM_SIZE, 1]
        :return: [], [BATCH*VIEWS]
        """
        cfg = self.cfg()
        num_candidates = cfg.pose_predict_num_candidates
        gt = tf_repeat_0(gt, num_candidates) # [BATCH*VIEWS*CANDIDATES, IM_SIZE, IM_SIZE, 1]
        sq_diff = (gt - pred) ** 2
        all_loss = torch.sum(sq_diff, [1, 2, 3]) # [BATCH*VIEWS*CANDIDATES]
        all_loss = torch.reshape(all_loss, [-1, num_candidates]) # [BATCH*VIEWS, CANDIDATES]
        min_loss = torch.argmin(all_loss, dim=1) # [BATCH*VIEWS]
        # tf.contrib.summary.histogram("winning_pose_candidates", min_loss)

        min_loss_mask = torch.FloatTensor(all_loss.size(0), num_candidates)
        min_loss_mask.zero_()
        min_loss_mask.scatter_(1, min_loss, 1)
        # min_loss_mask = tf.one_hot(min_loss, num_candidates) # [BATCH*VIEWS, CANDIDATES]
        num_samples = min_loss_mask.size(0)

        min_loss_mask_flat = torch.reshape(min_loss_mask, [-1]) # [BATCH*VIEWS*CANDIDATES]
        min_loss_mask_final = torch.reshape(min_loss_mask_flat, [-1, 1, 1, 1]) # [BATCH*VIEWS*CANDIDATES, 1, 1, 1]
        loss_tensor = (gt - pred) * min_loss_mask_final
        if cfg.variable_num_views:
            weights = inputs["valid_samples"]
            weights = tf_repeat_0(weights, num_candidates)
            weights = torch.reshape(weights, [weights.size(0), 1, 1, 1])
            loss_tensor *= weights
        # proj_loss = tf.nn.l2_loss(loss_tensor)
        # proj_loss /= tf.to_float(num_samples)
        proj_loss = torch.sum(loss_tensor ** 2) / 2
        proj_loss /= (num_samples * 1.)

        return proj_loss, min_loss

    def add_student_loss(self, inputs, outputs, min_loss, add_summary):
        cfg = self.cfg()
        num_candidates = cfg.pose_predict_num_candidates

        student = outputs["pose_student"]
        teachers = outputs["poses"]
        teachers = torch.reshape(teachers, [-1, num_candidates, 4])

        # indices = min_loss
        # indices = torch.unsqueeze(indices, dim=-1)
        # batch_size = teachers.size(0)
        # batch_indices = torch.range(0, batch_size, 1, dtype=torch.int64)
        # batch_indices = torch.unsqueeze(batch_indices, -1)
        # indices = torch.concat([batch_indices, indices], dim=1)
        # teachers = tf.gather_nd(teachers, indices)
        teachers = teachers[torch.arange(teachers.size(0)), min_loss]
        # use teachers only as ground truth
        # teachers = tf.stop_gradient(teachers)
        teachers.requires_grad_(False)

        if cfg.variable_num_views:
            weights = inputs["valid_samples"]
        else:
            weights = 1.0

        if cfg.pose_student_align_loss:
            ref_pc = self._pc_for_alignloss
            num_ref_points = ref_pc.size(0)
            ref_pc_all = torch.repeat_interleave(torch.unsqueeze(ref_pc, dim=0), [teachers.size(0), 1, 1])
            pc_1 = q_rotate(ref_pc_all, teachers)
            pc_2 = q_rotate(ref_pc_all, student)
            # student_loss = tf.nn.l2_loss(pc_1 - pc_2) / num_ref_points
            student_loss = torch.sum((pc_1 - pc_2) ** 2) / 2
            student_loss /= (num_ref_points * 1.)
        else:
            q_diff = q_norm(q_mul(teachers, q_conj(student)))
            angle_diff = q_diff[:, 0]
            student_loss = torch.sum((1.0 - angle_diff ** 2) * weights)

        num_samples = min_loss.size(0)
        student_loss /= (num_samples * 1.)

        # if add_summary:
        #     tf.contrib.summary.scalar("losses/pose_predictor_student_loss", student_loss)
        student_loss *= cfg.pose_predictor_student_loss_weight

        return student_loss

    def add_proj_loss(self, inputs, outputs, weight_scale, add_summary):
        cfg = self.cfg()
        gt = inputs['masks']
        pred = outputs['projs']
        num_samples = pred.size(0)

        gt_size = gt.size(1)
        pred_size = pred.size(1)
        assert gt_size >= pred_size, "GT size should not be higher than prediction size"
        if gt_size > pred_size:
            if cfg.bicubic_gt_downsampling:
                interp_method = "bicubic"
            else:
                interp_method = "bilinear"
            gt = F.interpolate(gt, size=[pred_size, pred_size], mode=interp_method)

        # if cfg.pc_gauss_filter_gt:
        #     sigma_rel = self._sigma_rel
        #     smoothed = gauss_smoothen_image(cfg, gt, sigma_rel)
        #     if cfg.pc_gauss_filter_gt_switch_off:
        #         gt = torch.where(sigma_rel < 1.0, gt, smoothed)
        #     else:
        #         gt = smoothed

        total_loss = 0
        num_candidates = cfg.pose_predict_num_candidates
        if num_candidates > 1:
            # pass
            proj_loss, min_loss = self.proj_loss_pose_candidates(gt, pred, inputs)
            if cfg.pose_predictor_student:
                student_loss = self.add_student_loss(inputs, outputs, min_loss, add_summary)
                total_loss += student_loss
        else:
            # proj_loss = tf.nn.l2_loss(gt - pred)
            # proj_loss /= tf.to_float(num_samples)
            # l = nn.MSELoss().to(self.device)
            # proj_loss = l(pred, gt)
            proj_loss = torch.sum((gt - pred) ** 2) / 2
            proj_loss /= (num_samples * 1.)

        total_loss += proj_loss

        # if add_summary:
        #     tf.contrib.summary.scalar("losses/proj_loss", proj_loss)

        total_loss *= weight_scale
        return total_loss


    def regularization_loss(self, cfg):
        reg_loss = torch.zeros(1, dtype=torch.float32)
        if cfg.weight_decay > 0:
            for v in self.params:
                reg_loss += torch.sum(v ** 2) / 2
        reg_loss *= cfg.weight_decay
        return reg_loss


    def get_loss(self, inputs, outputs, add_summary=True):
        """Computes the loss used for PTN paper (projection + volume loss)."""
        cfg = self.cfg()
        # g_loss = tf.zeros(dtype=tf.float32, shape=[])
        g_loss = self.add_proj_loss(inputs, outputs, cfg.proj_weight, add_summary)
        g_loss += self.regularization_loss(cfg)
        # if cfg.proj_weight:
        #     g_loss += self.add_proj_loss(inputs, outputs, cfg.proj_weight, add_summary)

        # if cfg.drc_weight:
        #     g_loss += add_drc_loss(cfg, inputs, outputs, cfg.drc_weight, add_summary)
        #
        # if cfg.pc_rgb:
        #     g_loss += add_proj_rgb_loss(cfg, inputs, outputs, cfg.proj_rgb_weight, add_summary, self._sigma_rel)
        #
        # if cfg.proj_depth_weight:
        #     g_loss += add_proj_depth_loss(cfg, inputs, outputs, cfg.proj_depth_weight, self._sigma_rel, add_summary)
        #
        # if add_summary:
        #     tf.contrib.summary.scalar("losses/total_task_loss", g_loss)

        return g_loss

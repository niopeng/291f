import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from util.camera_torch import camera_from_blender, quaternion_from_campos

class Chair_dataset(Dataset):

    def __init__(self, pkl_path, cfg):
        print('begin creating dataset')
        with open(pkl_path,'rb') as f:
            data = pickle.load(f)
            self.image = data["image"]
            self.mask = data["mask"]
            self.name = data['name']
            self.vox = data['vox']
            self.extrinsic = data['extrinsic']
            self.cam_pos = data['cam_pos']
            self.depth = data['depth']
            self.length = len(data['name'])
        self.num_views = cfg.num_views
        self.image_size = cfg.image_size
        print('done creating dataset')

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        sample = {'image': torch.FloatTensor(self.image[idx].reshape((self.num_views, self.image_size, self.image_size, 3))),
                  'mask': torch.FloatTensor(self.mask[idx].reshape((self.num_views, self.image_size, self.image_size, 1)))}
                  #'name': np.array(self.name[idx])}
        if len(self.vox)>0:
            sample['vox'] =torch.FloatTensor(self.vox[idx])
        if len(self.cam_pos)>0:
            sample['extrinsic'] = torch.FloatTensor(self.extrinsic[idx].reshape((self.num_views, 4, 4)))
            sample['cam_pos'] = torch.FloatTensor(self.cam_pos[idx].reshape((self.num_views,3)))
        if len(self.depth) >0:
            sample['depth']= torch.FloatTensor(self.depth[idx].reshape((self.num_views, self.image_size, self.image_size, 1)))

        return sample

def pool_single_view(cfg, tensor, view_idx):
    indices = torch.arange(cfg.batch_size) * cfg.step_size + view_idx
    output = tensor[indices]
    return output


def preprocess(cfg, raw_inputs, random_views=True):
    """Selects the subset of viewpoints to train on."""
    var_num_views = cfg.variable_num_views # False
    step_size = cfg.step_size
    num_views = raw_inputs['image'].shape[1]
    quantity = cfg.batch_size
    if cfg.num_views_to_use == -1:
        max_num_views = num_views
    else:
        max_num_views = cfg.num_views_to_use

    inputs = dict()

    def batch_sampler(all_num_views):
        out = np.zeros((0, 2), dtype=np.int64)
        valid_samples = np.zeros((0), dtype=np.float32)
        for n in range(quantity):
            valid_samples_m = np.ones((step_size), dtype=np.float32)
            if var_num_views:
                num_actual_views = int(all_num_views[n, 0])
                ids = np.random.choice(num_actual_views, min(step_size, num_actual_views), replace=False)
                if num_actual_views < step_size:
                    to_fill = step_size - num_actual_views
                    ids = np.concatenate((ids, np.zeros((to_fill), dtype=ids.dtype)))
                    valid_samples_m[num_actual_views:] = 0.0
            elif random_views:
                ids = np.random.choice(max_num_views, step_size, replace=False)
            else:
                ids = np.arange(0, step_size).astype(np.int64)

            ids = np.expand_dims(ids, axis=-1)
            batch_ids = np.full((step_size, 1), n, dtype=np.int64)
            full_ids = np.concatenate((batch_ids, ids), axis=-1)
            out = np.concatenate((out, full_ids), axis=0)

            valid_samples = np.concatenate((valid_samples, valid_samples_m), axis=0)

        return torch.LongTensor(out), torch.LongTensor(valid_samples)

    num_actual_views = raw_inputs['num_views'] if var_num_views else torch.tensor([0])

    indices, valid_samples = batch_sampler(num_actual_views) #[tf.int64, tf.float32])
    indices = torch.reshape(indices, [step_size * quantity, 2])
    inputs['valid_samples'] = torch.reshape(valid_samples, [step_size * quantity])
    inputs['masks'] = gather_nd(raw_inputs['mask'], indices[:,1],step_size)

    inputs['images'] = gather_nd(raw_inputs['image'], indices[:,1],step_size)
    if cfg.saved_depth:
        inputs['depths'] = gather_nd(raw_inputs['depth'], indices[:,1],step_size)
    inputs['images_1'] = pool_single_view(cfg, inputs['images'], 0)

    def fix_matrix(extr):
        out = np.zeros_like(extr)
        num_matrices = extr.shape[0]
        for k in range(num_matrices):
            out[k, :, :] = camera_from_blender(extr[k, :, :])
        return torch.FloatTensor(out)

    def quaternion_from_campos_wrapper(campos):
        num = campos.shape[0]
        out = np.zeros([num, 4], dtype=np.float32)
        for k in range(num):
            out[k, :] = quaternion_from_campos(campos[k, :])
        return torch.FloatTensor(out)

    if cfg.saved_camera:
        matrices = gather_nd(raw_inputs['extrinsic'], indices[:,1],step_size)
        orig_shape = matrices.shape
        extr_tf = fix_matrix(matrices)
        inputs['matrices'] = torch.reshape(extr_tf, shape=orig_shape)

        cam_pos =  gather_nd(raw_inputs['cam_pos'], indices[:,1],step_size)
        orig_shape = cam_pos.shape
        quaternion = quaternion_from_campos_wrapper(cam_pos)
        inputs['camera_quaternion'] = torch.reshape(quaternion, shape=[orig_shape[0], 4])
    return inputs

def gather_nd(input, indices, step_size):
    input = torch.repeat_interleave(input, step_size, dim=0)
    output = input[torch.arange(input.size(0)), indices]
    return output
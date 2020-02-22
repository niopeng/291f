import startup

import sys
import pickle
import os
import glob
import re
import random

import numpy as np
from scipy.io import loadmat
from imageio import imread

from skimage.transform import resize as im_resize

from util.fs import mkdir_if_missing

import torch
import argparse


parser = argparse.ArgumentParser(description='parameters')

parser.add_argument('--split_dir', metavar='DIR', default='',
                        help='Directory path containing the input rendered images.' )
parser.add_argument('--inp_dir_renders', metavar='DIR', default='',
                        help='Directory path containing the input rendered images.' )
parser.add_argument('--inp_dir_voxels', metavar='DIR', default='',
                        help='Directory path containing the input voxels.' )
parser.add_argument('--out_dir', metavar='DIR', default='',
                        help='Directory path to write the output.' )
parser.add_argument('--synth_set', metavar='NUM', default='03001627',
                        help='picture ID' )

parser.add_argument('--store_camera', action='store_true', help='')
parser.add_argument('--store_voxels', action='store_true', help='')
parser.add_argument('--store_depth', action='store_true', help='')

parser.add_argument('--split_path', metavar='', default='',
                        help='' )

parser.add_argument('--num_views', metavar='N', type=int, default=10,
                    help='Num of viewpoints in the input data.')
parser.add_argument('--image_size', metavar='I', type=int, default=64,
                    help='Input images dimension (pixels) - width & height.')
parser.add_argument('--vox_size', metavar='N', type=int, default=32,
                    help='Voxel prediction dimension.')

parser.add_argument('--tfrecords_gzip_compressed', action='store_true', help='Voxel prediction dimension.')

args = parser.parse_args()


def read_camera(filename):
    cam = loadmat(filename)
    extr = cam["extrinsic"]
    pos = cam["pos"]
    return extr, pos


def loadDepth(dFile, minVal=0, maxVal=10):
    dMap = imread(dFile)
    dMap = dMap.astype(np.float32)
    dMap = dMap*(maxVal-minVal)/(pow(2,16)-1) + minVal
    return dMap


def _dtype_feature(ndarray):
    ndarray = ndarray.flatten()
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    #if dtype_ == np.float64 or dtype_ == np.float32:
    #   # TODO
    #    return torch.tensor(ndarray)
    #elif dtype_ == np.int64:
    #    return torch.tensor(ndarray)
    #else:
    #    raise ValueError("The input should be numpy ndarray. \
    #                      Instaed got {}".format(ndarray.dtype))
    return ndarray


def _string_feature(s):
    s = s.encode('utf-8')
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[s]))
    return s


def create_record(synth_set, split_name, models):
    im_size = args.image_size
    num_views = args.num_views
    num_models = len(models)

    mkdir_if_missing(args.out_dir)

    # address to save the data
    train_filename = "{}/{}_{}.pkl".format(args.out_dir, synth_set, split_name)

    render_dir = os.path.join(args.inp_dir_renders, synth_set)
    voxel_dir = os.path.join(args.inp_dir_voxels, synth_set)
    imagel, maskl, namel,voxl, extl,cam_posl,depthl =[],[],[],[],[],[],[]
    for j, model in enumerate(models):
        print("{}/{}".format(j, num_models))

        if args.store_voxels:
            voxels_file = os.path.join(voxel_dir, "{}.mat".format(model))
            voxels = loadmat(voxels_file)["Volume"].astype(np.float32)

            # this needed to be compatible with the
            # PTN projections
            voxels = np.transpose(voxels, (1, 0, 2))
            voxels = np.flip(voxels, axis=1)

        im_dir = os.path.join(render_dir, model)
        images = sorted(glob.glob("{}/render_*.png".format(im_dir)))

        rgbs = np.zeros((num_views, im_size, im_size, 3), dtype=np.float32)
        masks = np.zeros((num_views, im_size, im_size, 1), dtype=np.float32)
        cameras = np.zeros((num_views, 4, 4), dtype=np.float32)
        cam_pos = np.zeros((num_views, 3), dtype=np.float32)
        depths = np.zeros((num_views, im_size, im_size, 1), dtype=np.float32)

        assert(len(images) >= num_views)

        for k in range(num_views):
            im_file = images[k]
            img = imread(im_file)
            rgb = img[:, :, 0:3]
            mask = img[:, :, [3]]
            mask = mask / 255.0
            if True:  # white background
                mask_fg = np.repeat(mask, 3, 2)
                mask_bg = 1.0 - mask_fg
                rgb = rgb * mask_fg + np.ones(rgb.shape)*255.0*mask_bg
            # plt.imshow(rgb.astype(np.uint8))
            # plt.show()
            rgb = rgb / 255.0
            actual_size = rgb.shape[0]
            if im_size != actual_size:
                rgb = im_resize(rgb, (im_size, im_size), order=3)
                mask = im_resize(mask, (im_size, im_size), order=3)
            rgbs[k, :, :, :] = rgb
            masks[k, :, :, :] = mask

            fn = os.path.basename(im_file)
            img_idx = int(re.search(r'\d+', fn).group())

            if args.store_camera:
                cam_file = "{}/camera_{}.mat".format(im_dir, img_idx)
                cam_extr, pos = read_camera(cam_file)
                cameras[k, :, :] = cam_extr
                cam_pos[k, :] = pos

            if args.store_depth:
                depth_file = "{}/depth_{}.png".format(im_dir, img_idx)
                depth = loadDepth(depth_file)
                d_max = 10.0
                d_min = 0.0
                depth = (depth - d_min) / d_max
                depth_r = im_resize(depth, (im_size, im_size), order=0)
                depth_r = depth_r * d_max + d_min
                depths[k, :, :] = np.expand_dims(depth_r, -1)
        imagel.append(_dtype_feature(rgbs))
        maskl.append(_dtype_feature(masks))
        namel.append(_string_feature(model))
        if args.store_voxels:
            voxl.append(_dtype_feature(voxels))
        if args.store_camera:
            extl.append(_dtype_feature(cameras))
            cam_posl.append(_dtype_feature(cam_pos))
        if args.store_depth:
            depthl.append(_dtype_feature(depths))
        """
        plt.imshow(np.squeeze(img[:,:,0:3]))
        plt.show()
        plt.imshow(np.squeeze(img[:,:,3]).astype(np.float32)/255.0)
        plt.show()
        """
        feature = {"image": imagel,
            "mask": maskl,
            "name":namel,
            "vox": voxl,
            "extrinsic":extl,
            "cam_pos": cam_posl,
            "depth":depthl}

    with open(train_filename, 'wb') as fp:
        #json.dump(feature, fp)
        pickle.dump(feature, fp)



SPLIT_DEF = [("val", 0.05), ("train", 0.95)]


def generate_splits(input_dir):
    files = [f for f in os.listdir(input_dir) if os.path.isdir(f)]
    models = sorted(files)
    random.shuffle(models)
    num_models = len(models)
    models = np.array(models)
    out = {}
    first_idx = 0
    for k, splt in enumerate(SPLIT_DEF):
        fraction = splt[1]
        num_in_split = int(np.floor(fraction * num_models))
        end_idx = first_idx + num_in_split
        if k == len(SPLIT_DEF)-1:
            end_idx = num_models
        models_split = models[first_idx:end_idx]
        out[splt[0]] = models_split
        first_idx = end_idx
    return out


def load_drc_split(base_dir, synth_set):
    filename = os.path.join(base_dir, "{}.file".format(synth_set))
    lines = [line.rstrip('\n') for line in open(filename)]

    k = 3  # first 3 are garbage
    split = {}
    while k < len(lines):
        _,_,name,_,_,num = lines[k:k+6]
        k += 6
        num = int(num)
        split_curr = []
        for i in range(num):
            _, _, _, _, model_name = lines[k:k+5]
            k += 5
            split_curr.append(model_name)
        split[name] = split_curr

    return split


def generate_records(synth_set):
    base_dir = args.split_dir
    split = load_drc_split(base_dir, synth_set)

    for key, value in split.items():
        if key == 'val':
            create_record(synth_set, key, value)


def read_split(filename):
    f = open(filename, "r")
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


if __name__ == '__main__':
    generate_records(args.synth_set)

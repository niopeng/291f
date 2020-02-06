## pytorching working on

import numpy as np
import tensorflow as tf
import torch

#slim = tf.contrib.slim


def _preprocess(images):
    return images * 2 - 1


def model(images, cfg, is_training):
    """Model encoding the images into view-invariant embedding."""
    del is_training  # Unused
    
    #image_size = images.get_shape().as_list()[1]
    # the images.size() is [batch_size, channel, h, w]
    image_size = images.size()[2]
    batch_size = images.size()[0]
    
    target_spatial_size = 4

    f_dim = cfg.f_dim
    fc_dim = cfg.fc_dim
    z_dim = cfg.z_dim
    outputs = dict()
    
    images = _preprocess(images)

    #act_func = tf.nn.leaky_relu
    act_func = torch.nn.LeakyReLU(negative_slope=0.2)

    num_blocks = int(np.log2(image_size / target_spatial_size) - 1)

    # define convolution layers
    conv_list = []
    conv_list.append(torch.nn.Conv2d(image.size()[1], f_dim, (5,5), stride=(2,2)))
    for k in range(num_blocks):
        new_f_dim = f_dim * 2
        conv_list.append(torch.nn.Conv2d(f_dim, new_f_dim, (3,3), stride=(2,2)))
        conv_list.append(torch.nn.Conv2d(new_f_dim, new_f_dim, (3,3), stride=(1,1)))
        f_dim = new_f_dim
    # feed the images into the convolution layers
    hf = images
    for conv_layer in conv_list:
        hf = conv_layer(hf)
        hf = act_func(hf)
    hf = hf.view(batch_size, -1)
    outputs["conv_features"] = hf

    # define fully-connected layers
    fc1 = torch.nn.Linear(hf.size()[1], fc_dim)
    fc2 = torch.nn.Linear(fc_dim, fc_dim)
    fc3 = torch.nn.Linear(fc_dim, z_dim)
    # feed the tensor into the fc layers
    out1 = act_func(fc1(hf))
    out2 = act_func(fc2(out1))
    out3 = fc3(out2)
    out3_act = act_func(out3)
    outputs["z_latent"] = out1
    outputs['ids'] = out3_act
    if cfg.predict_pose:
        outputs['poses'] = out3
    
    return outputs

# idk where is this one being used
def decoder_part(input, cfg):
    """Not sure what's this function for... """
    batch_size = input.size()[0]
    act_func = torch.nn.LeakyReLU(negative_slope=0.2)
    fc_dim = cfg.fc_dim
    z_dim = cfg.z_dim

    fc2 = torch.nn.Linear(input.size()[1], fc_dim)
    fc3 = torch.nn.Linear(fc_dim, z_dim)

    out = act_func(fc2(input))
    out = act_func(fc3(out))
    return out

# def decoder_part(input, cfg):
#     batch_size = input.shape.as_list()[0]
#     fake_input = tf.zeros([batch_size, 128*4*4])
#     act_func = tf.nn.leaky_relu

#     fc_dim = cfg.fc_dim
#     z_dim = cfg.z_dim

#     # this is unused but needed to match the FC layers in the encoder function
#     fc1 = slim.fully_connected(fake_input, fc_dim, activation_fn=act_func)

#     fc2 = slim.fully_connected(input, fc_dim, activation_fn=act_func)
#     fc3 = slim.fully_connected(fc2, z_dim, activation_fn=act_func)
#     return fc3

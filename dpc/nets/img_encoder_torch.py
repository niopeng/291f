## pytorching done

import numpy as np
import tensorflow as tf
import torch

#slim = tf.contrib.slim

class imgEncoder(torch.nn.Module):
    """Model encoding the images into view-invariant embedding."""
    def __init__(self, cfg, channel_number=3, image_size=128):
        super(imgEncoder, self).__init__()
        
        self.cfg = cfg
        self.fc_dim = cfg.fc_dim
        self.f_dim = cfg.f_dim
        self.z_dim = cfg.z_dim
        self.channel_number = channel_number
        self.image_size = image_size
        self.act_func = torch.nn.LeakyReLU(negative_slope=0.2)
        self.conv_list = []
        self.conv_list.append(torch.nn.Conv2d(self.channel_number, self.f_dim, (5,5), stride=(2,2),padding=2))
        target_spatial_size = 4
        num_blocks = int(np.log2(self.image_size / target_spatial_size) - 1)
        f_dim= self.f_dim
        for _ in range(num_blocks):
            new_f_dim = f_dim * 2
            self.conv_list.append(torch.nn.Conv2d(f_dim, new_f_dim, (3,3), stride=(2,2), padding=1))
            self.conv_list.append(torch.nn.Conv2d(new_f_dim, new_f_dim, (3,3), stride=(1,1), padding=1))
            f_dim = new_f_dim
    
        self.fc1 = torch.nn.Linear(256*4*4, self.fc_dim)
        self.fc2 = torch.nn.Linear(self.fc_dim, self.fc_dim)
        self.fc3 = torch.nn.Linear(self.fc_dim, self.z_dim)
        self.pose_fc = torch.nn.Linear(self.fc_dim, self.z_dim)
    
        # aka msra initialization
        for layer in self.conv_list:
            torch.nn.init.kaiming_normal_(layer.weight, a=0.2)
        torch.nn.init.kaiming_normal_(self.fc1.weight, a=0.2)
        torch.nn.init.kaiming_normal_(self.fc2.weight, a=0.2)
        torch.nn.init.kaiming_normal_(self.fc3.weight, a=0.2)
        torch.nn.init.kaiming_normal_(self.pose_fc.weight, a=0.2)


    def _preprocess(self, images):
        return images * 2 - 1
    
    
    def forward(self, images, is_training=True):
        del is_training # Unused
        outputs = dict()
        batch_size = images.size()[0]

        images = self._preprocess(images)
        out = images
        for conv_layer in self.conv_list:
            out = self.act_func(conv_layer(out))
        out = out.view(batch_size, -1)
        outputs["conv_features"] = out

        out = self.act_func(self.fc1(out))
        outputs["z_latent"] = out
        out = self.act_func(self.fc2(out))
        if self.cfg.predict_pose:
            outputs["poses"] = self.pose_fc(out)
        out = self.act_func(self.fc3(out))
        outputs["ids"] = out

        return outputs

# def _preprocess(images):
#     return images * 2 - 1


# def model(images, cfg, is_training):
#     """Model encoding the images into view-invariant embedding."""
#     del is_training  # Unused
    
#     #image_size = images.get_shape().as_list()[1]
#     # the images.size() is [batch_size, channel, h, w]
#     image_size = images.size()[2]
#     batch_size = images.size()[0]
    
#     target_spatial_size = 4

#     f_dim = cfg.f_dim
#     fc_dim = cfg.fc_dim
#     z_dim = cfg.z_dim
#     outputs = dict()
    
#     images = _preprocess(images)

#     #act_func = tf.nn.leaky_relu
#     act_func = torch.nn.LeakyReLU(negative_slope=0.2)

#     num_blocks = int(np.log2(image_size / target_spatial_size) - 1)

#     # define convolution layers
#     conv_list = []
#     conv_list.append(torch.nn.Conv2d(images.size()[1], f_dim, (5,5), stride=(2,2)))
#     for k in range(3): # for k in range(num_blocks):
#         new_f_dim = f_dim * 2
#         conv_list.append(torch.nn.Conv2d(f_dim, new_f_dim, (3,3), stride=(2,2)))
#         conv_list.append(torch.nn.Conv2d(new_f_dim, new_f_dim, (3,3), stride=(1,1)))
#         f_dim = new_f_dim
#     # feed the images into the convolution layers
#     hf = images
#     for conv_layer in conv_list:
#         hf = conv_layer(hf)
#         hf = act_func(hf)
#     hf = hf.view(batch_size, -1)
#     outputs["conv_features"] = hf

#     # define fully-connected layers
#     fc1 = torch.nn.Linear(hf.size()[1], fc_dim)
#     fc2 = torch.nn.Linear(fc_dim, fc_dim)
#     fc3 = torch.nn.Linear(fc_dim, z_dim)
#     # feed the tensor into the fc layers
#     out1 = act_func(fc1(hf))
#     out2 = act_func(fc2(out1))
#     out3 = fc3(out2)
#     out3_act = act_func(out3)
#     outputs["z_latent"] = out1
#     outputs['ids'] = out3_act
#     if cfg.predict_pose:
#         outputs['poses'] = out3
    
#     return outputs

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


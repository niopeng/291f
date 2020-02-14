# pytorching working on
import numpy as np
import torch
import torch.nn.functional as F

class pcDecoder(torch.nn.Module):
    def __init__(self, cfg, afterConvSize=256*4*4):
        super(pcDecoder, self).__init__()

        self.cfg = cfg
        self.fc_dim = cfg.fc_dim
        self.num_points = cfg.pc_num_points
        self.input_size = cfg.z_dim
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.act_func = torch.nn.LeakyReLU(negative_slope=0.2)
        self.fc1 = torch.nn.Linear(self.input_size, self.num_points*3)
        self.deep_fc1 = torch.nn.Linear(afterConvSize, self.fc_dim)
        self.deep_fc2 = torch.nn.Linear(self.fc_dim, self.fc_dim)
        self.deep_fc3 = torch.nn.Linear(self.fc_dim, self.fc_dim)
        self.deep_fc_last = torch.nn.Linear(self.fc_dim, self.num_points*3)
        self.shallow_fc_last = torch.nn.Linear(self.input_size, self.num_points*3)

        torch.nn.init.normal_(self.fc1.weight, std=cfg.pc_decoder_init_stddev)
        torch.nn.init.normal_(self.deep_fc_last.weight, std=cfg.pc_decoder_init_stddev)
        torch.nn.init.normal_(self.shallow_fc_last.weight, std=cfg.pc_decoder_init_stddev)
        torch.nn.init.kaiming_normal_(self.deep_fc1.weight, a=0.2)
        torch.nn.init.kaiming_normal_(self.deep_fc2.weight, a=0.2)
        torch.nn.init.kaiming_normal_(self.deep_fc3.weight, a=0.2)

    def forward(self, inputs, outputs_all, is_training=True):
        batch_size = inputs.size()[0]
        outputs = dict()

        pts_raw = self.fc1(inputs)
        pred_pts = pts_raw.view(batch_size, self.num_points, 3)
        pred_pts = self.tanh(pred_pts)

        if self.cfg.pc_unit_cube:
            pred_pts = pred_pts / 2.0
        
        outputs["xyz"] = pred_pts

        if self.cfg.pc_rgb:
            if self.cfg.pc_rgb_deep_decoder:
                inp = outputs_all["conv_features"]
                out = self.act_func(self.deep_fc1(inp))
                out = self.act_func(self.deep_fc2(out))
                out = self.act_func(self.deep_fc3(out))
                rgb_raw = self.deep_fc_last(out)
            
            else:
                rgb_raw = self.shallow_fc_last(inputs)

            rgb = rgb_raw.view(batch_size, self.num_points, 3)
            rgb = self.sigmoid(rgb)
        else:
            rgb = None
        
        outputs["rgb"] = rgb

        return outputs


# def model(inputs, outputs_all, cfg, is_training):
#     num_points = cfg.pc_num_points
#     init_stddev = cfg.pc_decoder_init_stddev
#     batch_size = inputs.size()[0]
#     fc1 = torch.nn.Linear(inputs.size()[1], num_points*3)
#     # nn.init.normal_ instead of tf.truncated_normal_initializer()
#     torch.nn.init.normal_(fc1.weight, std=init_stddev)

#     pts_raw = fc1(inputs)
#     pred_pts = pts_raw.view(batch_size, num_points, 3)
#     pred_pts = F.tanh(pred_pts)

#     if cfg.pc_unit_cube:
#         pred_pts = pred_pts / 2.0

#     out = dict()
#     out["xyz"] = pred_pts

#     deep_fc1 = torch.nn.Linear(outputs_all["conv_features"].size()[1], cfg.fc_dim)
#     deep_fc2 = torch.nn.Linear(cfg.fc_dim, cfg.fc_dim)
#     deep_fc3 = torch.nn.Linear(cfg.fc_dim, cfg.fc_dim)
#     deep_fc_last = torch.nn.Linear(cfg.fc_dim, num_points*3)
#     shallow_fc_last = torch.nn.Linear(inputs.size()[1], num_points*3)

#     # kaiming_normal_ instead of tf.contrib.layers.variance_scaling_initializer()
#     # aka msra initialization
#     torch.nn.init.kaiming_normal_(deep_fc1.weight, a=0.2)
#     torch.nn.init.kaiming_normal_(deep_fc2.weight, a=0.2)
#     torch.nn.init.kaiming_normal_(deep_fc3.weight, a=0.2)

#     torch.nn.init.normal_(deep_fc_last.weight, std=init_stddev)
#     torch.nn.init.normal_(shallow_fc_last.weight, std=init_stddev)

#     if cfg.pc_rgb:
#         if cfg.pc_rgb_deep_decoder:
#             inp = outputs_all["conv_features"]
#             inp = F.leaky_relu(deep_fc1(inp), negative_slope=0.2)
#             inp = F.leaky_relu(deep_fc2(inp), negative_slope=0.2)
#             inp = F.leaky_relu(deep_fc3(inp), negative_slope=0.2)
#             rgb_raw = deep_fc_last(inp)

#         else:
#             rgb_raw = shallow_fc_last(inputs)

#         rgb = rgb_raw.view(batch_size, num_points, 3)
#         rgb = F.sigmoid(rgb)
#     else:
#         rgb = None
    
#     out["rgb"] = rgb

#     return out

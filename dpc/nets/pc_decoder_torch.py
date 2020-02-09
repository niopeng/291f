# pytorching working on
import numpy as np
import torch
import torch.nn.functional as F


def model(inputs, outputs_all, cfg, is_training):
    num_points = cfg.pc_num_points
    init_stddev = cfg.pc_decoder_init_stddev
    batch_size = inputs.size()[0]
    fc1 = torch.nn.Linear(inputs.size()[1], num_points*3)
    # nn.init.normal_ instead of tf.truncated_normal_initializer()
    torch.nn.init.normal_(fc1.weight, std=init_stddev)

    pts_raw = fc1(inputs)
    pred_pts = pts_raw.view(batch_size, num_points, 3)
    pred_pts = F.tanh(pred_pts)

    if cfg.pc_unit_cube:
        pred_pts = pred_pts / 2.0

    out = dict()
    out["xyz"] = pred_pts

    deep_fc1 = torch.nn.Linear(outputs_all["conv_features"].size()[1], cfg.fc_dim)
    deep_fc2 = torch.nn.Linear(cfg.fc_dim, cfg.fc_dim)
    deep_fc3 = torch.nn.Linear(cfg.fc_dim, cfg.fc_dim)
    deep_fc_last = torch.nn.Linear(cfg.fc_dim, num_points*3)
    shallow_fc_last = torch.nn.Linear(inputs.size()[1], num_points*3)

    # kaiming_normal_ instead of tf.contrib.layers.variance_scaling_initializer()
    # aka msra initialization
    torch.nn.init.kaiming_normal_(deep_fc1.weight, a=0.2)
    torch.nn.init.kaiming_normal_(deep_fc2.weight, a=0.2)
    torch.nn.init.kaiming_normal_(deep_fc3.weight, a=0.2)

    torch.nn.init.normal_(deep_fc_last.weight, std=init_stddev)
    torch.nn.init.normal_(shallow_fc_last.weight, std=init_stddev)

    if cfg.pc_rgb:
        if cfg.pc_rgb_deep_decoder:
            inp = outputs_all["conv_features"]
            inp = F.leaky_relu(deep_fc1(inp), negative_slope=0.2)
            inp = F.leaky_relu(deep_fc2(inp), negative_slope=0.2)
            inp = F.leaky_relu(deep_fc3(inp), negative_slope=0.2)
            rgb_raw = deep_fc_last(inp)

        else:
            rgb_raw = shallow_fc_last(inputs)

        rgb = rgb_raw.view(batch_size, num_points, 3)
        rgb = F.sigmoid(rgb)
    else:
        rgb = None
    
    out["rgb"] = rgb

    return out

# pytorching working on
import torch
import torch.nn as nn
import torch.nn.functional as F

class basePoseEstimator(nn.Module):
    '''For creating several pose regressors, 
       input shape should be [batch_size, z_dim]'''
    def __init__(self, cfg):
        super(basePoseEstimator, self).__init__()

        self.cfg = cfg
        self.num_layers = cfg.pose_candidates_num_layers
        self.f_dim = 32
        self.input_size = cfg.z_dim
        self.act_func = nn.LeakyReLU(negative_slope=0.2)
        layers = []
        for k in range(self.num_layers):
            if k == 0:
                layers.append(nn.Linear(self.input_size, self.f_dim))
                layers.append(self.act_func)
            elif k == self.num_layers - 1:
                layers.append(nn.Linear(self.f_dim, 4))
                layers.append(self.act_func)
            else:
                layers.append(nn.Linear(self.f_dim, self.f_dim))
                layers.append(self.act_func)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, inputs):
        out = inputs
        out = self.layers(out)
#         for layer in self.layers:
#             out = self.act_func(layer(out))
        return out



class poseDecoder(nn.Module):
    '''The shape of output['poses'] is [batch_size, num_candidates, 4] \n
       The shape of output['pose_student'] is [batch_size, 1, 4]'''
    def __init__(self, cfg):
        super(poseDecoder, self).__init__()

        self.cfg = cfg
        self.num_candidates = cfg.pose_predict_num_candidates
        estimators = []
        for _ in range(self.num_candidates):
            estimators.append(basePoseEstimator(self.cfg))
        self.estimators = nn.Sequential(*estimators)
        
        if self.cfg.pose_predictor_student:
            self.student_estimator = basePoseEstimator(self.cfg)
        else:
            self.student_estimator = None

        if self.cfg.predict_translation:
            self.pose_translate_layer = nn.Linear(cfg.z_dim, 3)
            nn.init.normal_(self.pose_translate_layer.weight, std=cfg.predict_translation_init_stddev)
        else:
            self.pose_translate_layer = None

    
    def forward(self, inputs):
        '''The shape of output['poses'] is [batch_size, num_candidates, 4] \n
           The shape of output['pose_student'] is [batch_size, 1, 4]'''
        output = {}
        pose_candidates = [estimator(inputs) for estimator in self.estimators]
        pose_teachers = torch.cat(pose_candidates, dim=1)
        pose_teachers = pose_teachers.view(-1, self.num_candidates, 4)
        
        if self.student_estimator is None:
            pose_student = None
        else:
            pose_student = self.student_estimator(inputs)

        if self.pose_translate_layer is None:
            t = None
        else:
            t = self.pose_translate_layer(inputs)
            if self.cfg.predict_translation_tanh:
                t = nn.Tanh()(t) * self.cfg.predict_translation_scaling_factor
        
        output["poses"] = pose_teachers
        output["pose_student"] = pose_student
        output["predicted_translation"] = t
        
        return output

        
        

# def pose_branch(inputs, cfg):
#     num_layers = cfg.pose_candidates_num_layers
#     f_dim = 32
#     t = inputs
#     for k in range(num_layers):
#         if k == (num_layers - 1):
#             out_dim = 4
#             act_func = None
#         else:
#             out_dim = f_dim
#             act_func = tf.nn.leaky_relu
#         t = slim.fully_connected(t, out_dim, activation_fn=act_func)
#     return t


# def model(inputs, cfg):
#     """predict pose quaternions
#     inputs: [B,Z]
#     """

#     w_init = tf.contrib.layers.variance_scaling_initializer()

#     out = {}
#     with slim.arg_scope(
#             [slim.fully_connected],
#             weights_initializer=w_init):
#         with tf.variable_scope('predict_pose', reuse=tf.AUTO_REUSE):
#             num_candidates = cfg.pose_predict_num_candidates
#             if num_candidates > 1:
#                 outs = [pose_branch(inputs, cfg) for _ in range(num_candidates)]
#                 q = tf.concat(outs, axis=1)
#                 q = tf.reshape(q, [-1, 4])
#                 if cfg.pose_predictor_student:
#                     out["pose_student"] = pose_branch(inputs, cfg)
#             else:
#                 q = slim.fully_connected(inputs, 4, activation_fn=None)

#             if cfg.predict_translation:
#                 trans_init_stddev = cfg.predict_translation_init_stddev
#                 w_trans_init = tf.truncated_normal_initializer(stddev=trans_init_stddev, seed=1)
#                 t = slim.fully_connected(inputs, 3,
#                                          activation_fn=None,
#                                          weights_initializer=w_trans_init)
#                 if cfg.predict_translation_tanh:
#                     t = tf.tanh(t) * cfg.predict_translation_scaling_factor
#             else:
#                 t = None

#     out["poses"] = q
#     out["predicted_translation"] = t

#     return out

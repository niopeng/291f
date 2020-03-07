import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def model(encode_features, cfg):
    
    z_dim = cfg.z_dim
    init_stddev = cfg.pc_decoder_init_stddev
    w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)

    second_encoded = slim.fully_connected(encode_features, z_dim,
                                         activation_fn=None,
                                         weights_initializer=w_init)
    return second_encoded

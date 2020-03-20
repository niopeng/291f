#!/usr/bin/env python

import startup
from tqdm import tqdm
import os
import time

# import tensorflow as tf

from models import model_pc_pytorch

from util.app_config import config as app_config
from util.system import setup_environment
# from util.train import get_trainable_variables, get_learning_rate
# from util.losses import regularization_loss
from util.fs import mkdir_if_missing

import torch
from Chair_dataset import *
from torch.utils.data import DataLoader

def train():
    cfg = app_config

    setup_environment(cfg)

    train_dir = cfg.checkpoint_dir
    mkdir_if_missing(train_dir)

    # tf.logging.set_verbosity(tf.logging.INFO)

    split_name = "val" #train
    dataset_file = os.path.join(cfg.inp_dir, f"{cfg.synth_set}_{split_name}.pkl")
    dataset = Chair_dataset(dataset_file,cfg)
    if cfg.shuffle_dataset:
        torch.manual_seed(7000)
    print("*"*30)
    print('creating dataloader')
    train_loader = DataLoader(dataset=dataset,
                              batch_size=cfg.batch_size,
                              num_workers=8,
                              shuffle=cfg.shuffle_dataset)
    for epoch in tqdm(range(cfg.max_number_of_steps), desc='Epoch', ncols=100):
        train_size = len(train_loader)
        ts = time.time()
        print_now = 0
        for batch_idx, train_data in tqdm(enumerate(train_loader), desc='Batch', total=train_size,
                                      ncols=100):
            # global_step = tf.train.get_or_create_global_step()
            # model = model_pc_pytorch.ModelPointCloud(cfg, global_step)
            model = model_pc_pytorch.ModelPointCloud(cfg)
            inputs = preprocess(cfg, train_data)
            # print('inputs shape')
            # for i in inputs:
            #     print(i, inputs[i].shape)
            # Call Forward of model
            # outputs = model(inputs)
            # task_loss = model.get_loss(inputs, outputs)
#             print(inputs.keys())
            loss = model.optimize_parameters(inputs)
            if print_now % 200 == 0:
                print("Epoch: %d, Step: %d, Loss: %f" % (epoch, print_now, loss.item()))
            print_now += 1
            #reg_loss = regularization_loss(train_scopes, cfg)
            #loss = task_loss + reg_loss

            # break
        # break
    print("Training Complete!")

    '''
    dataset = dataset.map(lambda rec: parse_tf_records(cfg, rec), num_parallel_calls=4) \
        .batch(cfg.batch_size) \
        .prefetch(buffer_size=100) \
        .repeat()

    iterator = dataset.make_one_shot_iterator()
    train_data = iterator.get_next()

    summary_writer = tfsum.create_file_writer(train_dir, flush_millis=10000)

    with summary_writer.as_default(), tfsum.record_summaries_every_n_global_steps(10):
        global_step = tf.train.get_or_create_global_step()
        model = model_pc.ModelPointCloud(cfg, global_step)
        inputs = model.preprocess(train_data, cfg.step_size)

        model_fn = model.get_model_fn(
            is_training=True, reuse=False, run_projection=True)
        outputs = model_fn(inputs)

        # train_scopes
        train_scopes = ['encoder', 'decoder']

        # loss
        task_loss = model.get_loss(inputs, outputs)
        reg_loss = regularization_loss(train_scopes, cfg)
        loss = task_loss + reg_loss

        # summary op
        summary_op = tfsum.all_summary_ops()

        # optimizer
        var_list = get_trainable_variables(train_scopes)
        optimizer = tf.train.AdamOptimizer(get_learning_rate(cfg, global_step))
        train_op = optimizer.minimize(loss, global_step, var_list)

    # saver
    max_to_keep = 2
    saver = tf.train.Saver(max_to_keep=max_to_keep)

    session_config = tf.ConfigProto(
        log_device_placement=False)
    session_config.gpu_options.allow_growth = cfg.gpu_allow_growth
    session_config.gpu_options.per_process_gpu_memory_fraction = cfg.per_process_gpu_memory_fraction

    sess = tf.Session(config=session_config)
    with sess, summary_writer.as_default():
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        tfsum.initialize(graph=tf.get_default_graph())

        global_step_val = 0
        while global_step_val < cfg.max_number_of_steps:
            t0 = time.perf_counter()
            _, loss_val, global_step_val, summary = sess.run([train_op, loss, global_step, summary_op])
            t1 = time.perf_counter()
            dt = t1 - t0
            print(f"step: {global_step_val}, loss = {loss_val:.4f} ({dt:.3f} sec/step)")
            if global_step_val % 5000 == 0:
                saver.save(sess, f"{train_dir}/model", global_step=global_step_val)

    '''
def main(_):
    train()


if __name__ == '__main__':
    main(0)

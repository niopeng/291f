#!/bin/sh

python ../dpc/densify/densify.py \
--shapenet_path=dataset/Pix3D \
--python_interpreter=python \
--synth_set=$1 \
--subset=val \
--output_dir=gt/dense

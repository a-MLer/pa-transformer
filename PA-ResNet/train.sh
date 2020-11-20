#!/bin/bash

python -u tf_cnn_benchmarks.py \
    --data_format=NCHW \
    --batch_size=32 \
    --model=asresnet50 \
    --optimizer=momentum \
    --variable_update=replicated \
    --nodistortions \
    --gradient_repacking=8 \
    --num_gpus=8 \
    --num_epochs=100 \
    --weight_decay=1e-4 \
    --data_dir=${DIRECTORY_OF_IMAGENET_TFRECORD_DATA} \
    --train_dir=${OUTPUT_DIR} \
    --eval_during_training_every_n_epochs=1 \
    --save_model_steps=5005 \
    --num_eval_epochs=1

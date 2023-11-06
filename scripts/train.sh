#!/bin/bash

SEED=0
DATASET="waterbirds"
DATASET_DIR="dataset/waterbirds"
METHOD="erm"
LOG_DIR="logs/"
EPOCHS=250
LR=1e-4
WD=1e-3
BS=4
GA=1
PATIENCE=20

CUDA_VISIBLE_DEVICES=2 nohup python3 -u src/train.py \
--seed ${SEED} \
--dataset ${DATASET} \
--data_path ${DATASET_DIR} \
--method ${METHOD} \
--num_epochs ${EPOCHS} \
--learning_rate ${LR} \
--weight_decay ${WD} \
--batch_size ${BS} \
--grad_acc ${GA} \
--early_stopping_patience ${PATIENCE} \
--output_dir models/ \
--result_dir results/ > ${LOG_DIR}/${DATASET}_subsample_${METHOD}_lr_${LR}_wd_${WD}_bs_${BS}_seed_${SEED}.log &


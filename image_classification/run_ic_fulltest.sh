#!/usr/bin/env bash

strength=$1
warmup=$2
path="."

#arch="res8_fp"
#arch="res8_w8a8"
#arch="res8_w4a8"
#arch="res8_w2a8"
#arch="res8_w248a8_chan"
#arch="res8_w248a8_multiprec"
#arch="res8_w248a248_multiprec"
arch=$3

project="hp-nas_ic"

#tags="warmup"
#tags="fp"
#tags="init_same no_wp reg_w"
tags="init_same wp reg_w softemp"

mkdir -p ${path}/${arch}
mkdir -p ${path}/${arch}/model_${strength}

export WANDB_MODE=offline

if [[ "$4" == "search" ]]; then
    echo Search
    split=0.2
    # NB: add --warmup-8bit if needed
    python3 search.py ${path}/${arch}/model_${strength} -a mix${arch} \
        -d cifar --tiny-test --arch-data-split ${split} \
        --epochs 500 --step-epoch 50 -b 32 \
        --warmup ${warmup} --no-warmup-8bit --patience 100 \
        --lr 0.001 --lra 0.01 --wd 1e-4 \
        --ai same --cd ${strength} --rt weights \
        --seed 42 --gpu 0 \
        --no-gumbel-softmax --temperature 5 --anneal-temp \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/${arch}/model_${strength}/log_search_${strength}.txt
fi

if [[ "$5" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py ${path}/${arch}/model_${strength} -a quant${arch} \
        -d cifar --tiny-test --epochs 500 --step-epoch 50 -b 32 --patience 500 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${arch}/model_${strength}/arch_model_best.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/${arch}/model_${strength}/log_finetune_${strength}.txt
else
    echo From-Scratch
    #pretrained_model="${arch}/model_${strength}/arch_model_best.pth.tar"
    #pretrained_model="warmup_8bit.pth.tar"
    #pretrained_model="warmup_5bit.pth.tar"
    pretrained_model="warmup_fp.pth.tar"
    python3 main.py ${path}/${arch}/model_${strength} -a quant${arch} \
        -d cifar --tiny-test --epochs 500 --step-epoch 50 -b 32 --patience 500 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${pretrained_model} | tee ${path}/${arch}/model_${strength}/log_fromscratch_${strength}.txt
fi
#!/usr/bin/env bash

strength=$1
path="."
arch=$2
input_res=$3
target=$4

project="hp-nas_ic"
tags="init_same softemp"

if [[ "$5" == "now" ]]; then
    timestamp=$(date +"%Y-%m-%d-%T")
else
    timestamp=$5
fi
mkdir -p ${path}/${arch}
mkdir -p ${path}/${arch}/model_${strength}
mkdir -p ${path}/${arch}/model_${strength}/${timestamp}

export WANDB_MODE=offline

# pretrained_model="warmup_${input_res}.pth.tar"
pretrained_model="warmup_64red.pth.tar"
if [[ "$6" == "search" ]]; then
    echo Search
    split=0.0
    # NB: add --warmup-8bit if needed
    python3 search.py ${path}/${arch}/model_${strength}/${timestamp} -a mix${arch} \
        --input-res ${input_res} --no-std-head --val-split 0.1 \
        --epochs 50 -b 100 \
        --ac ${pretrained_model} --patience 10 \
        --lr 0.001 --lra 0.001 --wd 1e-4 \
        --ai same --cd ${strength} --target ${target} \
        --seed 42 --gpu 0 --workers 0 \
        --no-gumbel-softmax --temperature 1 --anneal-temp \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/${arch}/model_${strength}/${timestamp}/log_search_${strength}.txt
fi

if [[ "$7" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py ${path}/${arch}/model_${strength}/${timestamp} -a quant${arch} \
        --input-res ${input_res} --no-std-head --val-split 0.1 \
        --epochs 50 -b 100 --patience 10 \
        --lr 0.0005 --wd 1e-4 \
        --seed 42 --gpu 0 --workers 0 \
        --ac ${arch}/model_${strength}/${timestamp}/arch_model_best.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/${arch}/model_${strength}/${timestamp}/log_finetune_${strength}.txt
else
    echo From-Scratch
    # pretrained_model="warmup_224.pth.tar"
    pretrained_model="warmup_64red.pth.tar"
    python3 main.py ${path}/${arch}/model_${strength}/${timestamp} -a quant${arch} \
        --input-res ${input_res} --no-std-head --val-split 0.1 \
        --epochs 50 --step-epoch 10 -b 100 --patience 10 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 --workers 4 \
        --ac ${pretrained_model} | tee ${path}/${arch}/model_${strength}/${timestamp}/log_fromscratch_${strength}.txt
fi
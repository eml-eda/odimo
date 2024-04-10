#!/usr/bin/env bash

strength=$1
data="/space/risso/imagenet-1k/imagenet"
arch=$2
target=$3

if [[ "$4" == "now" ]]; then
    timestamp=$(date +"%Y-%m-%d-%T")
else
    timestamp=$4
fi
mkdir -p ${arch}
mkdir -p ${arch}/model_${strength}
mkdir -p ${arch}/model_${strength}/${timestamp}

if [[ "$5" == "search" ]]; then
    echo Search
    split=0.0
    # NB: add --warmup-8bit if needed
    python3 search.py ${data} -a mix${arch} \
        --val-split 0.1 \
        --epochs 300 -b 128 \
        --patience 40 \
        --lr 0.001 --lra 0.001 --wd 1e-4 \
        --ai same --cd ${strength} --target ${target} \
        --seed 42 --workers 4 --world-size 2 \
        --no-gumbel-softmax --temperature 1 --anneal-temp
fi

if [[ "$6" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py ${data} -a quant${arch} \
        --epochs 300 -b 128 --patience 40 \
        --lr 0.0005 --wd 1e-4 \
        --seed 42 --workers 4 --world-size 2 \
        --ac ${arch}/model_${strength}/${timestamp}/arch_model_best.pth.tar -ft
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

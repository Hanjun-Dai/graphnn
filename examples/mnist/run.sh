#!/bin/bash

if [ ! -e data ]; then
    mkdir data
fi

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e data/$fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
        mv $fname data/
    fi
done

DATA_ROOT=data

./build/mnist \
    -train_feat $DATA_ROOT/train-images-idx3-ubyte \
    -train_label $DATA_ROOT/train-labels-idx1-ubyte \
    -test_feat $DATA_ROOT/t10k-images-idx3-ubyte \
    -test_label $DATA_ROOT/t10k-labels-idx1-ubyte \
    -device 0 \
    2>&1 | tee log.txt

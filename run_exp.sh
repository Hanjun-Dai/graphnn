#!/bin/bash

DATA_ROOT=$HOME/data/dataset/mnist/raw
GRAPHNN=$HOME/Workspace/cpp/graphnn

$GRAPHNN/build/app/mnist -train_feat $DATA_ROOT/train-images.idx3-ubyte -train_label $DATA_ROOT/train-labels.idx1-ubyte -test_feat $DATA_ROOT/t10k-images.idx3-ubyte -test_label $DATA_ROOT/t10k-labels.idx1-ubyte -device 0

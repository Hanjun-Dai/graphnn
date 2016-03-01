#!/bin/bash

DATA=ENZYMES

DATA_ROOT=data/$DATA
RESULT_ROOT=results

tool=kernel_mean_field

LV=3
CONV_SIZE=64
FP_LEN=0
n_hidden=64
bsize=50
learning_rate=0.001
max_iter=200000
cur_iter=0
dev_id=0
fold=1
save_dir=$RESULT_ROOT/$tool-lv-$LV-conv-$CONV_SIZE-fp-$FP_LEN-bsize-$bsize-fold-$fold

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi


build/$tool \
	       -graph $DATA_ROOT/${DATA}.txt \
               -train_idx $DATA_ROOT/10fold_idx/train_idx-${fold}.txt \
               -test_idx $DATA_ROOT/10fold_idx/test_idx-${fold}.txt \
               -lr $learning_rate \
               -device $dev_id \
               -maxe $max_iter \
               -svdir $save_dir \
               -hidden $n_hidden \
               -int_save 10000 \
               -int_test 1000 \
               -l2 0.00 \
               -m 0.9 \
               -lv $LV \
               -conv $CONV_SIZE \
               -fp $FP_LEN \
               -b $bsize \
               -cur_iter $cur_iter \
               2>&1 | tee $save_dir/log.txt 


#!/bin/bash

maxMAP=0.8
minTruePos=5
GPU=3
LR=1e-3

for step in 1 2 
do
    for LRDNI in 1e-4 2e-4 5e-4
    do
        for NEIGH in 100 200 300 400 500
        do

        python train_ssr.py --gpu $GPU --lr $LR --lr-dni $LRDNI --nNeigh $NEIGH --maxMAP $maxMAP --minTruePos $minTruePos --nStep $step

        done
    done 
done 

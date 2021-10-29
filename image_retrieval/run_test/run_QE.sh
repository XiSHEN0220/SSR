#!/bin/bash

## nb of neighs = 300
for neigh in 1 3 5 7 9
do

python test_ssr_QE.py --SSR-dir cache/sfm120k_Step1_Rerank300_LrDni0.00010_maxMAP0.8_MinTruePos5_mAP_68.125/ --gpu 0 --QE-name AQE --QE-neigh $neigh

python test_ssr_QE.py --SSR-dir cache/sfm120k_Step1_Rerank300_LrDni0.00010_maxMAP0.8_MinTruePos5_mAP_68.125/ --gpu 0 --QE-name AQEwD --QE-neigh $neigh

python test_ssr_QE.py --SSR-dir cache/sfm120k_Step1_Rerank300_LrDni0.00010_maxMAP0.8_MinTruePos5_mAP_68.125/ --gpu 0 --QE-name alphaQE --QE-neigh $neigh

python test_ssr_QE.py --SSR-dir cache/sfm120k_Step1_Rerank300_LrDni0.00010_maxMAP0.8_MinTruePos5_mAP_68.125/ --gpu 0 --QE-name DQE --QE-neigh $neigh


done
    


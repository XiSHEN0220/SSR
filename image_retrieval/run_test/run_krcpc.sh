#!/bin/bash

## nb of neigh = 100
for l in 0.1 0.3 0.5 
do
python test_ssr_kreciprocal.py --SSR-dir cache/sfm120k_Step1_Rerank100_LrDni0.00020_maxMAP0.8_MinTruePos5_mAP_66.575 --k1 40 --k2 20 --lambda-value $l


python test_ssr_kreciprocal.py --SSR-dir cache/sfm120k_Step1_Rerank100_LrDni0.00020_maxMAP0.8_MinTruePos5_mAP_66.575 --k1 80 --k2 40 --lambda-value $l


python test_ssr_kreciprocal.py --SSR-dir cache/sfm120k_Step1_Rerank100_LrDni0.00020_maxMAP0.8_MinTruePos5_mAP_66.575 --k1 160 --k2 80 --lambda-value $l
done
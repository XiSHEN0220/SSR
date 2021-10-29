# Landmark datasets

## Data

We provide training and evaluation code on two datasets: rOxford5K and rParis6K.  

Please refer to this [link](https://github.com/XiSHEN0220/SSR/tree/main/image_retrieval/data) to download rOxford5K, rParis6K, SFM120K (for training).


## Training 

Training on SFM120K features can be launched with:

````
bash run_train/run.sh
````

## Evaluating QE + SSR

To evaluate QE + SSR, one can run: 
````
bash run_test/run_QE.sh
````

Note that the parameter `--SSR-dir` should be modified.

## Evaluating k-reciprocal + SSR

To evaluate k-reciprocal + SSR, one can run: 
````
bash run_test/run_krcpc.sh
````

Note that the parameter `--SSR-dir` should be modified.










# Preparing data


## Landmarks

Training features : [SFM120K](https://github.com/filipradenovic/cnnimageretrieval-pytorch#networks-with-projection-fc-layer-after-global-pooling) 
Test features include ([Oxford5K](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) and [Paris6K](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)). 


Features are extracted with the model pretrained on google-landmarks-2018. Model is released by [cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch#networks-with-projection-fc-layer-after-global-pooling) named gl18-[tl-resnet101-gem-w](http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth). 

Please refer to [their testing script](https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/examples/test_e2e.py) to extract features if needed.


One download from dropbox: 

````
bash download_landmark.sh
````

The structure should be

````
landmark/gl18_resnet101_roxford5k.pkl
landmark/gl18_resnet101_rparis6k.pkl
landmark/gl18_resnet101_sfm120k_.pkl
````

  

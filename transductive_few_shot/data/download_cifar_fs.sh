wget https://www.dropbox.com/s/wuxb1wlahado3nq/cifar-fs-splits.zip?dl=0
mv cifar-fs-splits.zip?dl=0 cifar-fs-splits.zip
unzip cifar-fs-splits.zip
rm cifar-fs-splits.zip

python get_cifarfs.py
mv cifar-fs-splits/val1000* cifar-fs/
rm cifar-fs/val1000Episode_5_way_1_shot.json cifar-fs/val1000Episode_5_way_5_shot.json
rm -r cifar-fs-splits


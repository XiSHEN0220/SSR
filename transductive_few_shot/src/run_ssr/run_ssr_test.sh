# miniIN + conv
nEpisode=2000
gpu=0
nStep=3


DATASET="miniImageNet"
CKPT="cache/Conv64_MiniImageNet_Step3_EpisodeFix_Suppk5_Qryk75_Hidden4096/outputs/netSSRBest.pth"
NSUPP=1
ARCH="ConvNet_4_64"

python few_shot_test.py --nEpisode $nEpisode --gpu $gpu --ckptPth $CKPT --dataset $DATASET --architecture $ARCH --nStep $nStep --nSupport $NSUPP

NSUPP=5
python few_shot_test.py --nEpisode $nEpisode --gpu $gpu --ckptPth $CKPT --dataset $DATASET --architecture $ARCH --nStep $nStep --nSupport $NSUPP



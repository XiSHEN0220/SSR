import os
import subprocess
import sys

gpu = sys.argv[1]
path = sys.argv[2]

nEpisode=2000

exps = os.listdir(path)
for exp in exps:
    ckptPth = os.path.join(path, exp)

    if not os.path.isdir(ckptPth):
        continue

    print(f'Testing {exp}')
    kws = exp.split('_')

    arch = kws[0]
    data = kws[1]
    nsteps = int(kws[2][-1])
    lr = float(kws[3][5:])

    if data == 'MiniImageNet':
        data = "miniImageNet"

    if arch == 'WRN':
        arch = 'WRN_28_10'
    elif arch == 'Conv64':
        arch = 'ConvNet_4_64'

    command = f"python few_shot_test.py --nEpisode {nEpisode} --gpu {gpu} --ckptPth {ckptPth} --dataset {data} --architecture {arch} --nStep {nsteps} --lr-dni {lr} --nSupport 1"
    print(command)
    subprocess.call(command, shell=True)

    command = f"python few_shot_test.py --nEpisode {nEpisode} --gpu {gpu} --ckptPth {ckptPth} --dataset {data} --architecture {arch} --nStep {nsteps} --lr-dni {lr} --nSupport 5"
    print(command)
    subprocess.call(command, shell=True)


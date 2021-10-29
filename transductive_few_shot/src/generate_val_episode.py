import os
import json 
import numpy as np 
import argparse 
parser = argparse.ArgumentParser(description='PyTorch Classification')


parser.add_argument('--imgDir', type = str, default='data/Mini-ImageNet/val', help='imgDir')
parser.add_argument('--nEpisode', type = int, default=1000, help='number of episodes')
parser.add_argument('--nSupport', type = int, default=1, help='number of support')
parser.add_argument('--nQuery', type = int, default=15, help='number of query')
parser.add_argument('--nClsEpisode', type = int, default=5, help='number of cls in each episode')

parser.add_argument('--outJson', type = str, default = '../data/Mini-ImageNet/val1000Episode_5_way_1_shot.json', help='output json file')

args = parser.parse_args()
print (args)



clsList = os.listdir(args.imgDir)
episode = []

for i in range(args.nEpisode) : 
    tmp = {'Support':[], 'Query':[]}
    clsEpisode = np.random.choice(clsList, args.nClsEpisode, replace=False) 
    for cls in clsEpisode : 
        clsPath = os.path.join(args.imgDir, cls)
        imgList = os.listdir(clsPath)
        imgCls = np.random.choice(imgList, args.nQuery + args.nSupport, replace=False)
        tmp['Support'].append([cls + '/' + imgCls[j] for j in range(args.nSupport)])
        tmp['Query'].append([cls + '/' + imgCls[j] for j in range(args.nSupport, args.nQuery + args.nSupport)])
     
        
    episode.append(tmp)
    
    

with open(args.outJson, 'w') as f : 
    json.dump(episode, f)


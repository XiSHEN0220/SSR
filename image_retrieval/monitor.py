import matplotlib.pyplot as plt
import numpy as np 
import argparse 
import os 
import json 

parser = argparse.ArgumentParser(description='Train classification but eval on retrieval task')

parser.add_argument('--input', type=str, nargs='+', help='input directorys')
parser.add_argument('--label', type=str, nargs='+', help='labels')
parser.add_argument('--output', type=str,  help='save name')
parser.add_argument('--yaxis', type=str, choices=['BestMAP', 'trainLoss', 'trainMAP', 'valMAP', 'rOxfordM', 'rOxfordH', 'rPairsM', 'rPairsH'], help='which plot?')

args = parser.parse_args()
print (args)

plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(10, 10))
for i, input in enumerate(args.input) : 
    with open(os.path.join(input, 'history.json'), 'r') as f :
        h = json.load(f)
        
    y = h[args.yaxis]
    x = 5 * (np.arange(len(y)) + 1)
    
    color = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(x, y, color=color, linewidth=3, marker = 'x', markersize = 8, label=args.label[i])
plt.xlabel('K iters')
plt.ylabel(args.yaxis)
plt.grid(True)
plt.legend(loc='best')
    
plt.show()
if args.output : 
    plt.savefig(args.output,bbox_inches='tight')

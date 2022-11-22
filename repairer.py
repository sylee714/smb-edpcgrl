import gym
import gym_pcgrl
import time 
import numpy as np
from gym_pcgrl.envs.probs.MarioLevelRepairer.CNet.model import CNet
from gym_pcgrl.envs.probs.generator2 import Generator
from gym_pcgrl.envs.probs.MarioLevelRepairer.GA.repairer import Repairer
# from stable_baselines.common.env_checker import check_env

# https://www.youtube.com/watch?v=dLP-2Y6yu70&ab_channel=sentdex
# if __name__ == '__main__':
#     from gym_pcgrl.envs.probs.MarioLevelRepairer.CNet.model import CNet
#     from gym_pcgrl.envs.probs.generator2 import Generator
#     from gym_pcgrl.envs.probs.MarioLevelRepairer.GA.repairer import Repairer

def saveLevelAsText(level, path):
    map={'X':0, 'S':1, '-':2, '?':3, 'Q':4, 'E':5,'<':6,'>':7,'[':8,']':9,'o':10,'B':11,'b':12}
    map2=['X','S','-', '?', 'Q', 'E','<','>','[',']','o','B','b']
    with open(path+".txt",'w') as f:
        for i in range(len(level)):
            str=''
            for j in range(len(level[0])):
                str+=map2[level[i][j]]
            f.write(str+'\n')

def readTextLevel(path):
    map={'X':0, 'S':1, '-':2, '?':3, 'Q':4, 'E':5,'<':6,'>':7,'[':8,']':9,'o':10,'B':11,'b':12}
    result = []
    arr = None
    with open(path) as f:
        data = f.readlines()
        h, w = len(data), len(data[0])-1
        arr = np.empty(shape=(h,w), dtype=int)
        for i in range(h):
            for j in range(w):
                arr[i][j]=map[data[i][j]]
    return arr

# def parseMarioMap(file_path):
#     new_map = np.zeros([14, 168], dtype = int)
#     # Using readlines()
#     mapFile = open(file_path, 'r')
#     lines = mapFile.readlines()

#     for line in lines:
#         for sym in line:
#             if sym != "\n":


# parseMarioMap("mario_map.txt")

mario_map = readTextLevel("mario_map.txt")
print(mario_map)

repairer = Repairer(0) # passing in cuda_id=0 for only cpu

# nothing gets repaired
for i in range(6):
    before = mario_map[:, i * 28 : (i+1) * 28]
    new_piece = repairer.repair(mario_map[:, i * 28 : (i+1) * 28])

    repaired = False    
    for i in range(14):
        for j in range(28):
            if before[i, j] != new_piece[i, j]:
                print("repaired")
                repaired = True
                break
        if repaired:
            break

    print("-------------------------------")
import numpy as np
import math
import os 
from statistics import mean

def readTextLevel(path):
    # map={'X':0, 'S':1, '-':2, '?':3, 'Q':4, 'E':5,'<':6,'>':7,'[':8,']':9,'o':10,'B':11,'b':12}
    # result = []
    arr = None
    with open(path) as f:
        data = f.readlines()
        h, w = len(data), len(data[0])-1
        print("h: ", h)
        print("w: ", w)
        arr = np.empty(shape=(h,w), dtype=int)
        for i in range(h):
            for j in range(w):
                arr[i][j]=data[i][j]
    return arr

def splitMap(map):
    sections = []
    # sectionMaps = []
    for i in range(6):
        # sections.append(map[:, i*28:(i+1)*28])
        sections.append(lv2Map(map[:, i*28:(i+1)*28]))
    return sections

def calKLFromMap(mpa, mpb, w=0.5, eps=0.001):
    result = 0
    keys = set(mpa.keys()) | set(mpb.keys())
    suma = sum([mpa[e] for e in mpa.keys()])
    sumb = sum([mpb[e] for e in mpb.keys()])
    for e in keys:
        a = ((eps + mpa[e]) / (suma + len(keys) * eps)) if (e in mpa.keys()) else (eps / (suma + len(keys) * eps));
        b = ((eps + mpb[e]) / (sumb + len(keys) * eps)) if (e in mpb.keys()) else (eps / (sumb + len(keys) * eps))
        result += w * a * math.log2(a / b) + (1 - w) * b * math.log2(b / a)
    return result

def calKL(mpa, mpb, eps=0.001):
    result = 0
    keys = set(mpa.keys()) | set(mpb.keys())
    suma = sum([mpa[e] for e in mpa.keys()])
    sumb = sum([mpb[e] for e in mpb.keys()])
    for e in keys:
        a = ((eps + mpa[e]) / (suma + len(keys) * eps)) if (e in mpa.keys()) else (eps / (suma + len(keys) * eps));
        b = ((eps + mpb[e]) / (sumb + len(keys) * eps)) if (e in mpb.keys()) else (eps / (sumb + len(keys) * eps))
        result += a * math.log2(a / b) 
    return result

def lv2Map(lv, fh=2, fw=2):
    mp={}
    h, w = lv.shape
    for i in range(h-fh+1):
        for j in range(w-fw+1):
            k = tuple((lv[i:i+fh, j:j+fw]).flatten())
            mp[k] = (mp[k]+1) if (k in mp.keys()) else 1
    return mp

def computeFun(blocks):
    funs = []
    for i in range(5):
        funs.append(calKLFromMap(blocks[i], blocks[i+1]))
    return funs

def computeHistory(blocks):
    visited = []
    history = []
    for block in blocks:
        score = 0
        for vist in visited:
            score = score + calKLFromMap(vist, block)
        if visited:
            history.append(score/len(visited))
        visited.append(block)
    return history

playableDirPath = "maps/playable/"
unplayableDirPath = "maps/unplayable/"

mapFiles = os.listdir(playableDirPath)

# go thru every map file
# for file in mapFiles:
#     # read the map file
#     lvlMap = readTextLevel(playableDirPath + mapFiles[0])
#     # split the map into 6 blocks and convert into map tiles to be used for KL
#     sections = splitMap(lvlMap)

# read the map file
lvlMap = readTextLevel(playableDirPath + mapFiles[0])
# split the map into 6 blocks and convert into map tiles to be used for KL
blocks = splitMap(lvlMap)
funs = computeFun(blocks)
history = computeHistory(blocks)
print(mean(funs))
print(mean(history))





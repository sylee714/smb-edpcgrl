import numpy as np
import math

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

piece0 = np.full((4, 28), 3)
piece1 = np.full((4, 28), 3)
# piece0 = np.random.randint(7, size=(14, 28))
# piece1 = np.random.randint(7, size=(14, 28))

# piece1[:, 10:13] = [[0, 0, 0], [0, 0, 0]]
piece1[:, 20:22] = [[0, 0], [0, 0]]
print(piece0)
print(piece1)

mp0 = lv2Map(piece0)
mp1 = lv2Map(piece1)
print(mp0)
print(mp1)

print(calKL(mp0, mp1))

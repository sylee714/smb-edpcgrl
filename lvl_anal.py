from gettext import npgettext
import numpy as np

def lv2Map(lv, fh=2, fw=2):
    mp={}
    h, w = lv.shape
    for i in range(h-fh+1):
        for j in range(w-fw+1):
            k = str(tuple((lv[i:i+fh, j:j+fw]).flatten()))
            mp[k] = (mp[k]+1) if (k in mp.keys()) else 1
    return mp

sym2num = {
    "S": 3,
    "g": 2,
    "t": 6,
    "#": 1,
    "@": 4,
    "1": 4,
    "k": 2,
    "F": 0,
    "U": 3,
    "-": 0,
    "!": 4,
    "C": 5,
    "M": 0,
    "X": 1,
    "E": 2,
    "2": 4,
    "Q": 5,
    "T": 6,
    "K": 2,
    "L": 4,
    "?": 4
}

# lvl = open("lvl-4.txt", "r")

# symbol_set = set()
# lvl_num = []

# for line in lvl:
#     lvl_line = ""
#     for symbol in line:
#         if symbol != "\n" and not symbol in symbol_set:
#             symbol_set.add(symbol)
#         if symbol != "\n":
#             lvl_line += str(sym2num[symbol])
#     lvl_line += "\n"
#     lvl_num.append(lvl_line)

# lvl.close()

# print(len(symbol_set))        
# print(symbol_set)
# print(lvl_num)

# f = open("lvl-4-num.txt", "w")
# f.writelines(lvl_num)
# f.close()

lvl = open("lvl-1-num.txt", "r")

lvl_num = []

for line in lvl:
    lvl_line = []
    for symbol in line:
        if symbol != "\n":
            lvl_line.append(int(symbol))
    lvl_num.append(lvl_line)

lvl.close()


lvl_num = np.array(lvl_num)

mp0 = lv2Map(lvl_num)

print(len(mp0))
for block in mp0:
    print("{} : {}".format(block, mp0[block]))

lvl = open("lvl-4-num.txt", "r")

lvl_num = []

for line in lvl:
    lvl_line = []
    for symbol in line:
        if symbol != "\n":
            lvl_line.append(int(symbol))
    lvl_num.append(lvl_line)

lvl.close()


lvl_num = np.array(lvl_num)

mp1 = lv2Map(lvl_num)

print(len(mp1))
for block in mp1:
    print("{} : {}".format(block, mp1[block]))

for block in mp1:
    if not block in mp0:
        mp0[block] = mp1[block]
    else:
        mp0[block] += mp1[block]

print("after merging")
print(len(mp0))
# for block in mp0:
#     print("{} : {}".format(block, mp0[block]))

sorted_mp0 = dict(sorted(mp0.items(), key=lambda item: item[1]))
for block in sorted_mp0:
    print("{} : {}".format(block, sorted_mp0[block]))

import json
  
with open('valid_tile_patterns.json', 'w') as convert_file:
     convert_file.write(json.dumps(sorted_mp0))

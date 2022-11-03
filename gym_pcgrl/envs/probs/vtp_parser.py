import numpy as np
import json
  
def parse(file_path):
    valid_patterns = {}

    # Opening JSON file
    f = open(file_path)
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    
    # Iterating through the json
    # list
    for ele in data:
        temp = ele
        ele = ele.strip("()")
        ele = ele.replace(" ", "")
        tiles = ele.split(",")
        k = (int(tiles[0]), int(tiles[1]), int(tiles[2]), int(tiles[3]))
        valid_patterns[k] = data[temp]

    # Closing file
    f.close()

    return valid_patterns

# vp = parse("valid_tile_patterns.json")

# print(vp)



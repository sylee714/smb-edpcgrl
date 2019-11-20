"""
A helper module that can be used by all problems
"""
import numpy as np

"""
Private function to get a list of all tile locations on the map that have any of
the tile_values

Parameters:
    map (any[][]): the current map
    tile_values (any[]): an array of all the tile values that the method is searching for

Returns:
    (int,int)[]: a list of (x,y) position on the map that have a certain value
"""
def _get_certain_tiles(map, tile_values):
    tiles = []
    for y in range(len(map)):
        for x in range(len(map[y])):
            if map[y][x] in tile_values:
                tiles.append((x, y))
    return tiles

"""
Private function that runs flood fill algorithm on the current color map

Parameters:
    x (int): the starting x position of the flood fill algorithm
    y (int): the starting y position of the flood fill algorithm
    color_map (int[][]): the color map that is being colored
    map (any[][]): the current tile map to check
    color_index (int): the color used to color in the color map
    passable_values (any[]): the current values that can be colored over

Returns:
    int: the number of tiles that has been colored
"""
def _flood_fill(x, y, color_map, map, color_index, passable_values):
    num_tiles = 0
    queue = [(x, y)]
    while len(queue) > 0:
        (cx, cy) = queue.pop(0)
        if color_map[cy][cx] != -1 or map[cy][cx] not in passable_values:
            continue
        num_tiles += 1
        color_map[cy][cx] = color_index
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
                continue
            queue.append((nx, ny))
    return num_tiles

"""
Calculates the number of regions in the current map with passable_values

Parameters:
    map (any[][]): the current map being tested
    passable_values (any[]): an array of all the passable tile values

Returns:
    int: number of regions in the map
"""
def calc_num_regions(map, passable_values):
    empty_tiles = _get_certain_tiles(map, passable_values)
    region_index=0
    color_map = np.full((len(map), len(map[0])), -1)
    for (x,y) in empty_tiles:
        num_tiles = _flood_fill(x, y, color_map, map, region_index + 1, passable_values)
        if num_tiles > 0:
            region_index += 1
        else:
            continue
    return region_index

"""
Private function that runs dikjstra algorithm and return the map

Parameters:
    x (int): the starting x position for dikjstra algorithm
    y (int): the starting y position for dikjstra algorithm
    map (any[][]): the current map being tested
    passable_values (any[]): an array of all the passable tile values

Returns:
    int[][]: returns the dikjstra map after running the dijkstra algorithm
"""
def _run_dikjstra(x, y, map, passable_values):
    dikjstra_map = np.full((len(map), len(map[0])),-1)
    visited_map = np.zeros((len(map), len(map[0])))
    queue = [(x, y, 0)]
    while len(queue) > 0:
        (cx,cy,cd) = queue.pop(0)
        if map[cy][cx] not in passable_values or (dikjstra_map[cy][cx] >= 0 and dikjstra_map[cy][cx] <= cd):
            continue
        visited_map[cy][cx] = 1
        dikjstra_map[cy][cx] = cd
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
                continue
            queue.append((nx, ny, cd + 1))
    return dikjstra_map, visited_map

"""
Calculate the longest path on the map

Parameters:
    map (any[][]): the current map being tested
    passable_values (any[]): an array of all passable tiles in the map

Returns:
    int: the longest path in tiles in the current map
"""
def calc_longest_path(map, passable_values):
    empty_tiles = _get_certain_tiles(map, passable_values)
    final_visited_map = np.zeros((len(map), len(map[0])))
    final_value = 0
    for (x,y) in empty_tiles:
        if final_visited_map[y][x] > 0:
            continue
        dikjstra_map, visited_map = _run_dikjstra(x, y, map, passable_values)
        final_visited_map += visited_map
        (mx,my) = np.unravel_index(np.argmax(dikjstra_map, axis=None), dikjstra_map.shape)
        dikjstra_map, _ = _run_dikjstra(mx, my, map, passable_values)
        max_value = np.max(dikjstra_map)
        if max_value > final_value:
            final_value = max_value
    return final_value

"""
Calculate the number of tiles that have certain values in the map

Returns:
    int: get number of tiles in the map that have certain tile values
"""
def calc_certain_tile(map, tile_values):
    return len(_get_certain_tiles(map, tile_values))

"""
Calculate the number of reachable tiles of a certain values from a certain starting value
The starting value has to be one on the map

Parameters:
    map (any[][]): the current map
    start_value (any): the start tile value it has to be only one on the map
    passable_values (any[]): the tile values that can be passed in the map
    reachable_values (any[]): the tile values that the algorithm trying to reach

Returns:
    int: number of tiles that has been reached of the reachable_values
"""
def calc_num_reachable_tile(map, start_value, passable_values, reachable_values):
    (sx,sy) = _get_certain_tiles(map, [start_value])[0]
    dikjstra_map, _ = _run_dikjstra(sx, sy, map, passable_values)
    tiles = _get_certain_tiles(map, reachable_values)
    total = 0
    for (tx,ty) in tiles:
        if dikjstra_map[ty][tx] >= 0:
            total += 1
    return total

"""
Generate random map based on the input Parameters

Parameters:
    random (numpy.random): random object to help generate the map
    width (int): the generated map width
    height (int): the generated map height
    prob (dict(int,float)): the probability distribution of each tile value

Returns:
    int[][]: the random generated map
"""
def gen_random_map(random, width, height, prob):
    map = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            total = 0
            randv = random.rand()
            for v in prob:
                total += prob[v]
                if randv < total:
                    map[y][x] = int(v)
                    break
    return map

"""
A method to convert the map to use the tile names instead of tile numbers

Parameters:
    map (numpy.int[][]): a numpy 2D array of the current map
    tiles (string[]): a list of all the tiles in order

Returns:
    string[][]: a 2D map of tile strings instead of numbers
"""
def get_string_map(map, tiles):
    int_to_string = dict((i, s) for i, s in enumerate(tiles))
    result = []
    for y in range(map.shape[0]):
        result.append([])
        for x in range(map.shape[1]):
            result[y].append(int_to_string[int(map[y][x])])
    return result

"""
A method to convert the probability dictionary to use tile numbers instead of tile names

Parameters:
    prob (dict(string,float)): a dictionary of the probabilities for each tile name
    tiles (string[]): a list of all the tiles in order

Returns:
    Dict(int,float): a dictionary of tile numbers to probability values (sum to 1)
"""
def get_int_prob(prob, tiles):
    string_to_int = dict((s, i) for i, s in enumerate(tiles))
    result = {}
    total = 0
    for t in tiles:
        result[string_to_int[t]] = prob[t]
        total += prob[t]
    for i in result:
        result[i] /= total
    return result
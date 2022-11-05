from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, get_floor_dist, get_type_grouping, get_changes
from gym_pcgrl.envs.probs.smb.engine import State,BFSAgent,DFSAgent,AStarAgent
from gym_pcgrl.envs.probs.MarioLevelRepairer.CNet.model import CNet
from gym_pcgrl.envs.probs.generator2 import Generator
from gym_pcgrl.envs.probs.MarioLevelRepairer.GA.repairer import Repairer
import random
import time
import subprocess
import math

from gym_pcgrl.envs.probs.utils import *
from gym_pcgrl.envs.helper import *
from gym_pcgrl.envs.probs.utils import *
from collections import deque

from gym_pcgrl.envs.probs.vtp_parser import *

rootpath = os.path.abspath(os.path.dirname(__file__)) + "/"

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

# This method saves the passed in level map using the Mariopuzzle symbols
def saveLevelAsText(level, path):
    map={'X':0, 'S':1, '-':2, '?':3, 'Q':4, 'E':5,'<':6,'>':7,'[':8,']':9,'o':10,'B':11,'b':12}
    map2=['X','S','-', '?', 'Q', 'E','<','>','[',']','o','B','b']
    with open(path+".txt",'w') as f:
        for i in range(len(level)):
            str=''
            for j in range(len(level[0])):
                str+=map2[level[i][j]]
            f.write(str+'\n')

class SMBProblem(Problem):
    def __init__(self):
        super().__init__()

        # -----PCGRL Reward Method-----
        self._solver_power = 100000

        self._min_empty = 1100
        self._min_enemies = 12
        self._max_enemies = 37
        self._min_jumps = 25

        self._rewards = {
            "dist-floor": 2,
            "disjoint-tubes": 1,
            "enemies": 1,
            "empty": 1,
            "noise": 4,
            "jumps": 2,
            "jumps-dist": 2,
            "dist-win": 5,
            "completion-rate": 5,
            "fun": 15,
            "hist-dev": 20,
            "tile-diff": 1
        }
        # -----PCGRL Reward Method-----

        # Need this for now to pass in to the reset method
        self._prob = {"empty":0.75, "solid":0.1, "enemy":0.01, "brick":0.04, "question":0.01, "coin":0.02, "tube": 0.02}
        self._noise_rate = 0.3

        # self._width = 140 # original = 114; the width does not include the left 3 cols and right 3 cols
        self._width = 168
        self._height = 14
        self._total_num_of_tiles = self._width * self._height
        
        self._border_size = (3, 0)

        self.initial_state = None
        self.nz = 32
        self.generator = Generator(random.randint(1, 10000000 - 1))
        self.repairer = Repairer(0) # passing in cuda_id=0 for only cpu

        self.win_h, self.win_w = 14, 28
        self._num_of_tiles_per_block = self.win_h * self.win_w
        self._num_of_blocks = self._width // self.win_w
        self.sy, self.sx = 14, 7
        self.ny, self.nx = 0, 4

        self._start_block_num = 1
        self._cur_block_num = 0
        # self._cur_block_num = self._start_block_num # to tell which block iteration is on

        # take one block from the total since we are not changing the very first initial block
        remaining_tiles = self._total_num_of_tiles - self._num_of_tiles_per_block
        # calculate the end block number
        self._end_block_num = remaining_tiles//self._num_of_tiles_per_block

        # print("end block: ", self._end_block_num)

        # termination condition
        # self._last_iteration = remaining_tiles
        # 7 iterations per block since we are moving 2 rows at a time
        # self._last_iteration = self._end_block_num * 4
        self._last_iteration = 4

        # bool to track if it generates unplayable blocks 10 times in a row
        self.unplayable = False

        # print("Number of tiles per block: ", self._num_of_tiles_per_block)
        # print("End Block: ", self._end_block_num )
        # print("Last iteration: ", self._last_iteration)

        # self.valid_patterns = parse(rootpath + "valid_tile_patterns.json")
        
        # self.F_que = deque(maxlen=100)
        # self.H_que = deque(maxlen=100)
        # max. number of previous blocks to consider for the historical feature
        self.novel_k = 4
        self.history_stack = deque(maxlen=(self.novel_k + 1))

    def reset(self, start_stats):
        super().reset(start_stats)
        self.generator = Generator(random.randint(1, 10000000 - 1))
        self._cur_block_num = 0
        # self._cur_block_num = self._start_block_num # to tell which block iteration is on
        self.fun = 0
        self.hist_dev = 0
        self.kl_val = 0
        self.valid_pattern_counts = 0
        self.history_stack.clear()
        self.unplayable = False

    # Generate a random vector, which is used to generate the initial block
    def sample_random_vector(self, size):
        return np.clip(np.random.randn(size), -1, 1)

    # Converts Mariopuzzle map tiles to PCGRL map tiles
    # String Version
    def convertMP2PCGRL_str(self, map):
        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j] == 0:
                    map[i][j] = "brick"
                elif map[i][j] == 1:
                    map[i][j] = "solid"
                elif map[i][j] == 2:
                    map[i][j] = "empty"
                elif map[i][j] == 3:
                    map[i][j] = "question"
                elif map[i][j] == 4: 
                    map[i][j] = "question"
                elif map[i][j] == 5:
                    map[i][j] = "enemy"
                elif map[i][j] == 6:
                    map[i][j] = "tube"
                elif map[i][j] == 7: 
                    map[i][j] = "tube"
                elif map[i][j] == 8:
                    map[i][j] = "tube"
                elif map[i][j] == 9:
                    map[i][j] = "tube"
                elif map[i][j] == 10:
                    map[i][j] = "coin"
                elif map[i][j] == 11:
                    map[i][j] = "empty"
                elif map[i][j] == 12:
                    map[i][j] = "empty"

    # Converts Mariopuzzle map tiles to PCGRL map tiles
    # Numerical version
    def convertMP2PCGRL_num(self, map):
        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j] == 0:
                    map[i][j] = 3
                elif map[i][j] == 1:
                    map[i][j] = 1
                elif map[i][j] == 2:
                    map[i][j] = 0
                elif map[i][j] == 3:
                    map[i][j] = 4
                elif map[i][j] == 4: 
                    map[i][j] = 4
                elif map[i][j] == 5:
                    map[i][j] = 2
                elif map[i][j] == 6:
                    map[i][j] = 6
                elif map[i][j] == 7:
                    map[i][j] = 6
                elif map[i][j] == 8:
                    map[i][j] = 6
                elif map[i][j] == 9:
                    map[i][j] = 6
                elif map[i][j] == 10:
                    map[i][j] = 5
                elif map[i][j] == 11:
                    map[i][j] = 0
                elif map[i][j] == 12:
                    map[i][j] = 0
    
    # Converts PCGRL str map tiles to Mariopuzzle num map tiles
    def convertPCGRL_str2MP_num(self, map):
        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j] == "empty":
                    map[i][j] = 2
                elif map[i][j] == "solid":
                    map[i][j] = 1
                elif map[i][j] == "solid_above":
                    map[i][j] = 1
                elif map[i][j] == "enemy":
                    map[i][j] = 5
                elif map[i][j] == "brick": 
                    map[i][j] = 0
                elif map[i][j] == "question":
                    map[i][j] = 3
                elif map[i][j] == "coin":
                    map[i][j] = 10
                elif map[i][j] == "top_left":
                    map[i][j] = 6
                elif map[i][j] == "top_right":
                    map[i][j] = 7
                elif map[i][j] == "tube_left":
                    map[i][j] = 8
                elif map[i][j] == "tube_right":
                    map[i][j] = 9

    # Reads the file that has completion rate after running Mario AI Framework.
    def readMarioAIResultFile(self, path):
        f = open(path, "r")
        content = f.read()
        
        return float(content)

    # Adds the start and end point to Mario Puzzle Map to match the PCGRL's map
    def addStartEndPoints(self, map):
        start_point = np.zeros([self._height, 3],dtype = int)

        for row in range(len(start_point)):
            for col in range(len(start_point[row])):
                if row == len(start_point) - 1 or row == len(start_point) - 2:
                    start_point[row, col] = 0
                else:
                    start_point[row, col] = 2

        end_point = np.zeros([self._height, 3],dtype = int)
        for row in range(len(end_point)):
            for col in range(len(end_point[row])):
                if row == len(end_point) - 1 or row == len(end_point) - 2:
                    end_point[row, col] = 0
                else:
                    end_point[row, col] = 2

        return np.concatenate((start_point, map, end_point), axis=1)

    # modifies the initial block by adding noises
    def add_noise(self, piece):
        new_piece = np.zeros([self._height, self.win_w], dtype = int)
        new_piece[:, :] = piece[:, :]

        iterations = int(self._noise_rate * self._num_of_tiles_per_block)
        for i in range(iterations):
            x = random.randint(0, self.win_w - 1)
            y = random.randint(0, self.win_h - 1)
            # self._prob = {"empty":0.75, "solid":0.1, "enemy":0.01, "brick":0.04, "question":0.01, "coin":0.02, "tube": 0.02}
            # np.random.choice(np.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2]
            # return ["empty", "solid", "enemy", "brick", "question", "coin", "tube"]
            tile = np.random.choice(np.arange(0, len(self.get_tile_types())), p=[0.75, 0.1, 0.01, 0.09, 0.01, 0.02, 0.02])
            new_piece[y][x] = tile

        return new_piece

    # modify this method to initialize all the blocks
    def init_map(self, map):

        # temp_map = np.zeros([self._height, 28], dtype = int)
        new_piece = np.zeros([self._height, self.win_w], dtype = int)
        
        playable = False

        # count to reset the game if it generates unplayable segments 10 times in the same block
        count = 0

        # Keep generate the block till it's playable
        # print("Generating the initial block")
        while not playable and count < 10:
            # random state
            self.state = self.sample_random_vector(self.nz)

            # generate a new piece
            st = time.time()
            piece = self.generator.generate(self.state)

            # repair broken tiles
            st = time.time()
            new_piece = self.repairer.repair(piece)

            # Pass in the generated piece to the Mario AI to check if the new piece is playable from the start
            full_map = self.addStartEndPoints(new_piece)
            saveLevelAsText(full_map, rootpath + "mario_current_map")
            subprocess.call(['java', '-jar', rootpath + "Mario-AI-Framework.jar", rootpath + "mario_current_map.txt"])
            completion_rate = self.readMarioAIResultFile(rootpath + "mario_result.txt")

            # print("Completion Rate: {}".format(completion_rate))

            # check it's playable
            if completion_rate == 1.0:
                playable = True

            if not playable:
                count += 1
        # print("--------------------------------")

        # if generated a playable initial block, add it to the map
        if playable:
            self.convertMP2PCGRL_num(new_piece)
            self.init_block = lv2Map(new_piece[:, :])
            self.history_stack.append(self.init_block)
            
            map[:, :28] = new_piece[:, :]

            # copy the intial block and add some noises
            for i in range(1, self._num_of_blocks):
                map[:, i*28 : (i+1)*28] = self.add_noise(new_piece)[:, :]

        # if not, mark this episode as unplayable
        else:
            self.unplayable = True


    # This method is to repair a block after the representation finishes updating the working block.
    def repair_block(self, map, block_num):
        # 1. Need to convert PCGRL tiles to MP tiles
        # 2. Pass the block to the repairer
        # 3. Swap the original block with the repaired one

        # First, need to convert the num tiles to str tiles in PCG-RL, since some tiles like tubes are not distinguishable
        # Then, convert the pcg-rl str tiles to mariopuzzle num-tiles
        new_map = self.convertPCGRL_num2PCGRL_str(get_string_map(map, self.get_tile_types()))
        new_map = np.array(new_map)

        # convert the pcg-rl str tiles to mariopuzzle num-tiles
        self.convertPCGRL_str2MP_num(new_map)
        pass

    def get_tile_types(self):
        return ["empty", "solid", "enemy", "brick", "question", "coin", "tube"]

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        # -----PCGRL Reward Method-----
        self._min_empty = kwargs.get('min_empty', self._min_empty)
        self._min_enemies = kwargs.get('min_enemies', self._min_enemies)
        self._max_enemies = kwargs.get('max_enemies', self._max_enemies)
        self._min_jumps = kwargs.get('min_jumps', self._min_jumps)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]
        # -----PCGRL Reward Method-----

    # Converts PCGRL num tiles to PCGRL str tiles
    def convertPCGRL_num2PCGRL_str(self, map):
        new_map = []

        for y in range(len(map)):
            new_map.append([])

            # This distinguishes between solid and solid above & different parts of tube
            for x in range(len(map[y])):
                value = map[y][x]
                if map[y][x] == "solid" and y < self._height - 2:
                    value = "solid_above"
                if map[y][x] == "tube":
                    if y >= 1 and map[y-1][x] != "tube":
                        value = "top"
                    if x >= 1 and map[y][x-1] != "tube":
                        value += "_left"
                    else:
                        value += "_right"
                new_map[y].append(value)

        return new_map

    # Convert the current map as runnable to render by adding Mario at the front and the finish pole at the end.
    def _get_runnable_lvl(self, map):
        # size of map = 140 x 14 -> without 3 left/right cols for the player and the pole
        # size of new map = 146 x 14
        new_map = []

        for y in range(len(map)):
            new_map.append([])
            # This add 3 cols at the front
            for x in range(3):
                if y < self._height - 2:
                    new_map[y].append("empty")
                else:
                    new_map[y].append("solid")

            # This distinguishes between solid and solid above & different parts of tube
            for x in range(len(map[y])):
                value = map[y][x]
                if map[y][x] == "solid" and y < self._height - 2:
                    value = "solid_above"
                if map[y][x] == "tube":
                    if y >= 1 and map[y-1][x] != "tube":
                        value = "top"
                    if x >= 1 and map[y][x-1] != "tube":
                        value += "_left"
                    else:
                        value += "_right"
                new_map[y].append(value)
            
            # This add 3 cols at the end
            for x in range(3):
                if y < self._height - 2:
                    new_map[y].append("empty")
                else:
                    new_map[y].append("solid")

        new_map[-3][1] = "player"
        new_map[-3][-2] = "solid_above"
        for y in range(3, len(map) - 3):
            new_map[y][-2] = "pole"
        new_map[1][-2] = "pole_top"
        new_map[2][-2] = "pole_flag"
        new_map[2][-3] = "flag"

        return new_map

    def _run_game(self, map):
        gameCharacters=" # ## #"
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvlString = ""
        for i in range(len(map)):
            if i < self._height - 3:
                lvlString += "   "
            elif i == self._height - 3:
                lvlString += " @ "
            else:
                lvlString += "###"
            for j in range(len(map[i])):
                string = map[i][j]
                lvlString += string_to_char[string]
            if i < self._height - 3:
                lvlString += " | "
            elif i == self._height - 3:
                lvlString += " # "
            else:
                lvlString += "###"
            lvlString += "\n"

        state = State()
        state.stringInitialize(lvlString.split("\n"))

        aStarAgent = AStarAgent()

        sol,solState,iters = aStarAgent.getSolution(state, 1, self._solver_power)
        if solState.checkWin():
            return 0, solState.getGameStatus()
            
        sol,solState,iters = aStarAgent.getSolution(state, 0, self._solver_power)
        if solState.checkWin():
            return 0, solState.getGameStatus()
                
        return solState.getHeuristic(), solState.getGameStatus()

    # Computed the current stats of the map
    # str_map is used to run the A* agent on it
    # num_map is used to calculate Fun and Historical Deviation values
    def get_stats(self, str_map=None, num_map=None, cur_block=0):
        # convert the str_map to a numpy arry
        new_map = np.array(str_map)

        # print("current block: ", cur_block)

        # update the current block number
        self._cur_block_num = cur_block

        # calculate the start x of the current block
        now_x = self.win_w * self._cur_block_num

        # calculate the end x of the current block
        end_x = now_x + self.win_w
        
        map_stats = {
            "block-num": self._cur_block_num,
            "dist-win": 0,
            "jumps": 0,
            "jumps-dist": 0,
        }

        # run A* agent to play the game
        map_stats["dist-win"], play_stats = self._run_game(new_map[:, : end_x])
        map_stats["status"] = play_stats["status"] 

        map_stats["jumps"] = play_stats["jumps"]
        prev_jump = 0
        value = 0
        for l in play_stats["jump_locs"]:
            value = max(value, l[0] - prev_jump)
            prev_jump = l[0]
        value = max(value, self._width - prev_jump)
        map_stats["jumps-dist"] = value

        # ------ Playability ------
        self.completion_rate = (end_x - map_stats["dist-win"]) / end_x
        map_stats["completion-rate"] = self.completion_rate
        # ------ Playability ------

        # ------ Fun ------
        kl_val = KLWithSlideWindow(num_map, 
                                    (0, now_x, self.win_h, self.win_w), 
                                    self.sx, self.nx, 
                                    self.sy, self.ny)
        self.kl_val = kl_val
        # self.fun = self.kl_fn(kl_val)
        self.fun = kl_val
        # map_stats["kl-val"] = self.kl_val
        map_stats["fun"] = self.fun
        # ------ Fun ------

        # ------ Historical Deviation ------
        # pop the prev one if still on the same block number
        if len(self.history_stack) > self._cur_block_num:
            self.history_stack.pop()

        # calculate historical deviation
        piece_map = lv2Map(num_map[ : , now_x : end_x])
        self.hist_dev = self.cal_novelty(piece_map)
        map_stats["hist-dev"] = self.hist_dev
        # map_stats["hist-dev-kl-fn"] = self.kl_fn(self.hist_dev)
        self.history_stack.append(piece_map)
        # ------ Historical Deviation ------

        # ------ Title Locations ------
        # prev_tile_locations = get_tile_locations(new_map[ : , now_x - self.win_w : now_x], self.get_tile_types())
        # cur_tile_locations = get_tile_locations(new_map[ : , now_x : end_x], self.get_tile_types())

        # prev_tile_counts = {}
        # for tile in self.get_tile_types():
        #     prev_tile_counts[tile] = calc_certain_tile(prev_tile_locations, [tile])
        # map_stats["prev-tile-counts"] = prev_tile_counts

        # cur_tile_counts = {}
        # for tile in self.get_tile_types():
        #     cur_tile_counts[tile] = calc_certain_tile(cur_tile_locations, [tile])            
        # map_stats["cur-tile-counts"] = cur_tile_counts

        # total_tile_diff = 0
        # for tile in self.get_tile_types():
        #     total_tile_diff += abs(cur_tile_counts[tile] - prev_tile_counts[tile])
        # map_stats["total-tile-diff"] = total_tile_diff
        # ------ Title Locations ------


        return map_stats

    # count the number of valid pattern in a newly generated piece
    def eval_valid_patterns(self, piece):
        counts = 0
        fh = 2
        fw = 2
        h, w = piece.shape
        for i in range(h-fh+1):
            for j in range(w-fw+1):
                k = tuple((piece[i:i+fh, j:j+fw]).flatten())
                if k in self.valid_patterns:
                    counts += 1
        return counts

    # Get the block number based on the number of iterations
    def get_cur_block_num(self, iter):
        if iter % 4 == 1:
            return 1
        elif iter % 4 == 2:
            return 2
        elif iter % 4 == 3:
            return 3
        else:
            return 4

    # Computes the reward value
    def get_reward(self, new_stats=None, old_stats=None):
        reward = 0    

        # print("old his-dev: ", old_stats["hist-dev"])
        # print("new his-dev: ", new_stats["hist-dev"])
        # print("optimal range: ", get_range_reward(new_stats["hist-dev"], old_stats["hist-dev"], 0.26, 0.94))
        # print("old")
        # print("tile-diff: ", old_stats["total-tile-diff"])
        # print("new")
        # print("tile-diff: ", new_stats["total-tile-diff"])
        # print("optimal range: ", get_range_reward(new_stats["total-tile-diff"], old_stats["total-tile-diff"], 5, 30))

        # sqrt-ing the tile-diff value, so need to keep track of the sign
        # tile_diff_sign = 1 if get_range_reward(new_stats["total-tile-diff"], old_stats["total-tile-diff"], 5, 30) >= 0 else -1

        rewards = {
            "completion-rate": get_range_reward(new_stats["completion-rate"], old_stats["completion-rate"], 0, 1.0),
            "fun": get_range_reward(new_stats["fun"], old_stats["fun"], 0.26, 0.94),
            "hist-div": get_range_reward(new_stats["hist-dev"], old_stats["hist-dev"], 0.26, 0.94)
            # "tile-diff": abs(get_range_reward(new_stats["total-tile-diff"], old_stats["total-tile-diff"], 5, 30))
        }

        # print("reward: ", rewards["hist-div"] * self._rewards["hist-dev"])
        # print("reward: ", rewards["tile-diff"] * self._rewards["tile-diff"])

        # compare the number of tiles between the prev block and the cur block?
        # randomly choose some sub-blocks from the prev block? 
        # copy the prev block but randomly add noises?

        return rewards["hist-div"] * self._rewards["hist-dev"] + rewards["fun"] * self._rewards["fun"] + rewards["completion-rate"] * self._rewards["completion-rate"]
                # math.sqrt(rewards["tile-diff"]) * self._rewards["tile-diff"] * tile_diff_sign

    # Fun reward function
    # lower bound = 0.26
    # upper bound = 0.94
    def kl_fn(self, val):
        if (val < 0.26):
            return -(val-0.26)**2
        if (val > 0.94):
            return -(val-0.94)**2
        # return 0
        # return (val*10)**2
        return val

    def cal_novelty(self, piece):
        score = []
        for x in self.history_stack:
            score.append(calKLFromMap(x, piece))

        score.sort()
        sum = 0

        # novel_k = 4
        # only consider top 4 most similar blocks
        siz = min(len(score), self.novel_k)
        for i in range(siz):
            sum += score[i]

        if siz > 0:
            sum /= siz
    
        return sum

    def add_then_norm(self, value, history):
        history.append(value)
        maxv = max(history)
        minv = min(history)
        if maxv == minv:
            return 0
        else:
            return (value-minv)/(maxv-minv)
    
    # Check the termination conditions
    # 1. Level is not playable
    # 2. Reached the end iteration
    # 3. Failed to produce a playable initial map
    def get_episode_over(self, new_stats=None, old_stat=None):
        return new_stats["completion-rate"] != 1.0 or self.unplayable

    # Return the debug information
    def get_debug_info(self, new_stats, old_stats):
        return {
            "dist-win": new_stats["dist-win"],
            "jumps": new_stats["jumps"],
            "jumps-dist": new_stats["jumps-dist"],
            "status": new_stats["status"],
            "completion-rate": new_stats["completion-rate"],
            # "kl-val": new_stats["kl-val"],
            "fun": new_stats["fun"],
            "hist-dev": new_stats["hist-dev"]
            # "hist-dev-kl-fn": new_stats["hist-dev-kl-fn"]
        }

    def render(self, map):
        new_map = self._get_runnable_lvl(map)

        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/smb/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/smb/solid_floor.png").convert('RGBA'),
                "solid_above": Image.open(os.path.dirname(__file__) + "/smb/solid_air.png").convert('RGBA'),
                "enemy": Image.open(os.path.dirname(__file__) + "/smb/enemy.png").convert('RGBA'),
                "brick": Image.open(os.path.dirname(__file__) + "/smb/brick.png").convert('RGBA'),
                "question": Image.open(os.path.dirname(__file__) + "/smb/question.png").convert('RGBA'),
                "coin": Image.open(os.path.dirname(__file__) + "/smb/coin.png").convert('RGBA'),
                "top_left": Image.open(os.path.dirname(__file__) + "/smb/top_left.png").convert('RGBA'),
                "top_right": Image.open(os.path.dirname(__file__) + "/smb/top_right.png").convert('RGBA'),
                "tube_left": Image.open(os.path.dirname(__file__) + "/smb/tube_left.png").convert('RGBA'),
                "tube_right": Image.open(os.path.dirname(__file__) + "/smb/tube_right.png").convert('RGBA'),
                "pole_top": Image.open(os.path.dirname(__file__) + "/smb/poletop.png").convert('RGBA'),
                "pole": Image.open(os.path.dirname(__file__) + "/smb/pole.png").convert('RGBA'),
                "pole_flag": Image.open(os.path.dirname(__file__) + "/smb/flag.png").convert('RGBA'),
                "flag": Image.open(os.path.dirname(__file__) + "/smb/flagside.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/smb/player.png").convert('RGBA')
            }
        self._border_size = (0, 0)
        img = super().render(new_map)
        self._border_size = (3, 0)

        return img

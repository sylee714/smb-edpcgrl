from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, get_floor_dist, get_type_grouping, get_changes
from gym_pcgrl.envs.probs.smb.engine import State,BFSAgent,AStarAgent
from gym_pcgrl.envs.probs.MarioLevelRepairer.CNet.model import CNet
from gym_pcgrl.envs.probs.generator2 import Generator
from gym_pcgrl.envs.probs.MarioLevelRepairer.GA.repairer import Repairer
import random
import time
import subprocess
from gym_pcgrl.envs.probs.utils import *
from gym_pcgrl.envs.helper import *
from gym_pcgrl.envs.probs.utils import *
from collections import deque
rootpath = os.path.abspath(os.path.dirname(__file__)) + "/"

class SMBProblem(Problem):
    def __init__(self):
        super().__init__()

        # -----PCGRL Reward Method-----
        self._solver_power = 10000

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
            "dist-win": 5
        }
        # -----PCGRL Reward Method-----

        # Need this for now to pass in to the reset method
        self._prob = {"empty":0.75, "solid":0.1, "enemy":0.01, "brick":0.04, "question":0.01, "coin":0.02, "tube": 0.02}

        self._width = 140 # original = 114; the width does not include the left 3 cols and right 3 cols
        self._height = 14
        self._total_num_of_tiles = self._width * self._height
        
        self._border_size = (3, 0)

        self.initial_state = None
        self.nz = 32
        self.generator = Generator(random.randint(1, 10000000))
        self.repairer = Repairer(0) # passing in cuda_id=0 for only cpu

        self.novel_k = 4

        self.win_h, self.win_w = 14, 28
        self._num_of_tiles_per_block = self.win_h * self.win_w
        self.sy, self.sx = 14, 7
        self.ny, self.nx = 0, 3

        self._start_block_num = 1
        self._cur_block_num = self._start_block_num # to tell which block iteration is on

        # take one block from the total since we are not changing the very first initial block
        remaining_tiles = self._total_num_of_tiles - self._num_of_tiles_per_block
        # calculate the end block number
        self._end_block_num = remaining_tiles//self._num_of_tiles_per_block

        # termination condition
        self._last_iteration = remaining_tiles

        # print("Number of tiles per block: ", self._num_of_tiles_per_block)
        # print("End Block: ", self._end_block_num )
        # print("Last iteration: ", self._last_iteration)
        
        # self.F_que = deque(maxlen=self._total_num_of_tiles)
        # self.H_que = deque(maxlen=self._total_num_of_tiles)
        self.history_stack = deque(maxlen=(self.novel_k + 1))


    def reset(self, start_stats):
        super().reset(start_stats)
        self._cur_block_num = self._start_block_num # to tell which block iteration is on

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

    # This method saves the passed in level map using the Mariopuzzle symbols
    def saveLevelAsText(self, level, path):
        map={'X':0, 'S':1, '-':2, '?':3, 'Q':4, 'E':5,'<':6,'>':7,'[':8,']':9,'o':10,'B':11,'b':12}
        map2=['X','S','-', '?', 'Q', 'E','<','>','[',']','o','B','b']
        with open(path+".txt",'w') as f:
            for i in range(len(level)):
                str=''
                for j in range(len(level[0])):
                    str+=map2[level[i][j]]
                f.write(str+'\n')

    def readMarioAIResultFile(self, path):
        f = open(path, "r")
        content = f.read()
        print(content)
        return float(content)

    # modify this method to initialize all the blocks
    def init_map(self, map):
        # add 3 cols at the start for Super Mario ?
        # add 3 cols at the end for the finish pole ?

        temp_map = np.zeros([self._height, self._width], dtype = int)
        
        # need to generate 5 blocks in total
        for i in range(int(self._width / self.win_w)):
            
            playable = False

            print("Generating block ", i)
            # Keep generate the block till it's playable
            while not playable:
                if self.initial_state != None:
                    self.state = self.initial_state
                else:
                    self.state = self.sample_random_vector(self.nz)

                st = time.time()
                piece = self.generator.generate(self.state)
                st = time.time()
                new_piece = self.repairer.repair(piece)

                # Copy the new piece to the temp map
                temp_map[:, i * 28 : (i + 1) * 28] = new_piece

                # Pass in the generated piece to the Mario AI to check
                # if the new piece is playable
                self.saveLevelAsText(temp_map[: , : (i + 1) * 28], rootpath + "mario_current_map")
                subprocess.call(['java', '-jar', rootpath + "Mario-AI-Framework.jar", rootpath + "mario_current_map.txt"])
                completion_rate = self.readMarioAIResultFile(rootpath + "mario_result.txt")
                print("Block {} Completion Rate: {}".format(i, completion_rate))
                if completion_rate == 1.0:
                    playable = True
                
            print("--------------------------------")


        self.convertMP2PCGRL_num(temp_map)

        self.history_stack.append(lv2Map(temp_map[0:self.win_h, 0:self.win_w]))
        
        map[:, :] = temp_map[:, :]

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
    def get_stats(self, map=None):
        map_stats = {
            "block-num": self._cur_block_num
        }

        # convert to numpy for easy slicing
        new_map = np.array(map)

        # Consider the whole map?
        # or only the certain blocks based on the number of iterations
        # if we consider only the certain blocks based on the number of iterations,
        # dynamically calculate the min and max numbers of the empty tiles and the enemy tiles.
        # Then, we also need a logtic to convert "self._cur_block_num" to valid indices. 
        # Ex. map[:, : cur_block * win_w + win_w] 
        # -----PCGRL Reward Method-----
        map_locations = get_tile_locations(new_map[:, : self._cur_block_num * self.win_w + self.win_w], self.get_tile_types())
        map_stats = {
            "dist-floor": get_floor_dist(new_map[:, : self._cur_block_num * self.win_w + self.win_w], ["enemy"], ["solid", "brick", "question", "tube_left", "tube_right"]),
            "disjoint-tubes": get_type_grouping(new_map[:, : self._cur_block_num * self.win_w + self.win_w], ["tube"], [(-1,0),(1,0)],1,1),
            "enemies": calc_certain_tile(map_locations, ["enemy"]),
            "empty": calc_certain_tile(map_locations, ["empty"]),
            "noise": get_changes(new_map[:, : self._cur_block_num * self.win_w + self.win_w], False) + get_changes(new_map[:, : self._cur_block_num * self.win_w + self.win_w], True),
            "jumps": 0,
            "jumps-dist": 0,
            "dist-win": 0
        }
        map_stats["dist-win"], play_stats = self._run_game(new_map[:, : self._cur_block_num * self.win_w + self.win_w])
        map_stats["jumps"] = play_stats["jumps"]
        prev_jump = 0
        value = 0
        for l in play_stats["jump_locs"]:
            value = max(value, l[0] - prev_jump)
            prev_jump = l[0]
        value = max(value, self._width - prev_jump)
        map_stats["jumps-dist"] = value
        # -----PCGRL Reward Method-----

        return map_stats

    def get_cur_block_num(self, iterations):
        for i in range(self._end_block_num + 1):
            if (self._num_of_tiles_per_block * i) + 1 <= iterations and iterations <= self._num_of_tiles_per_block * (i + 1):
                return i+1

    # Computes the reward value
    def get_reward(self, new_stats=None, old_stats=None, map=None, iterations=0):
        # -----ED-PCGRL Reward Method-----
        self.current_iteration = iterations
    
        reward = 0

        self._prev_block_num = self._cur_block_num
        self._cur_block_num = self.get_cur_block_num(self.current_iteration)

        # Calculate the X start position based on the current block number
        now_x = 0 + self.win_w * self._cur_block_num

        # Convert the map to MarioPuzzle tile to run the Mario-AI framework
        # First, need to convert the num tiles to str tiles in PCG-RL, since some tiles like tubes are not distinguishable
        # Then, convert the pcg-rl str tiles to mariopuzzle num-tiles

        # Convert the pcg-rl num tiles to str tiles
        new_map = self.convertPCGRL_num2PCGRL_str(get_string_map(map, self.get_tile_types()))

        # convert the pcg-rl str tiles to mariopuzzle num-tiles
        self.convertPCGRL_str2MP_num(new_map)

        # convert to numpy array for easier index slicing
        new_map = np.array(new_map)

        # run the Mario-AI framework
        self.saveLevelAsText(new_map[:, max(0, now_x-3*self.win_w): now_x+self.win_w], rootpath + "mario_current_map")
        subprocess.call(['java', '-jar', rootpath + "Mario-AI-Framework.jar", rootpath + "mario_current_map.txt"])
        self.completion_rate = self.readMarioAIResultFile(rootpath + "mario_result.txt")
        reward += self.completion_rate

        # for the map, use the originally passed in map
        # calculate the diversity
        kl_val = KLWithSlideWindow(
            map, (0, now_x, self.win_h, self.win_w), self.sx, self.nx, self.sy, self.ny)
        self.kl_val = kl_val

        # need to clear the F_que when we move to the next block section
        # calculate fun 
        # rew_F = self.add_then_norm(self.kl_fn(kl_val), self.F_que)
        rew_F = self.kl_fn(kl_val)
        self._rew_F = rew_F
        # print("rew_F: ", rew_F)
        reward += rew_F

        # calculate historical deviation
        piece_map = lv2Map(map[:, now_x : now_x + self.win_w])
        novelty = self.cal_novelty(piece_map)

        # if we are in the same block, pop the previous one and add it.
        if self._prev_block_num == self._cur_block_num:
            self.history_stack.pop()
        self.history_stack.append(piece_map)
        # rew_H = self.add_then_norm(novelty, self.H_que)
        rew_H = novelty
        self._rew_H = rew_H
        # print("rew_H: ", rew_H)
        reward += rew_H

        # -----ED-PCGRL Reward Method-----

        # To dynamically change the min and max of empty and enemy tiles 
        # based on the current block number
        # ratio = self._cur_block_num / (self._end_block_num + 1)

        # -----PCGRL Reward Method-----
        # longer path is rewarded and less number of regions is rewarded
        # rewards = {
        #     "dist-floor": get_range_reward(new_stats["dist-floor"], old_stats["dist-floor"], 0, 0),
        #     "disjoint-tubes": get_range_reward(new_stats["disjoint-tubes"], old_stats["disjoint-tubes"], 0, 0),
        #     "enemies": get_range_reward(new_stats["enemies"], old_stats["enemies"], int(self._min_enemies * ratio), int(self._max_enemies * ratio)),
        #     "empty": get_range_reward(new_stats["empty"], old_stats["empty"], int(self._min_empty * ratio), np.inf),
        #     "noise": get_range_reward(new_stats["noise"], old_stats["noise"], 0, 0),
        #     "jumps": get_range_reward(new_stats["jumps"], old_stats["jumps"], int(self._min_jumps * ratio), np.inf),
        #     "jumps-dist": get_range_reward(new_stats["jumps-dist"], old_stats["jumps-dist"], 0, 0),
        #     "dist-win": get_range_reward(new_stats["dist-win"], old_stats["dist-win"], 0, 0)
        # }

        # #calculate the total reward
        return reward 
            # + rewards["dist-floor"] * self._rewards["dist-floor"] +\
            # rewards["disjoint-tubes"] * self._rewards["disjoint-tubes"] +\
            # rewards["enemies"] * self._rewards["enemies"] +\
            # rewards["empty"] * self._rewards["empty"] +\
            # rewards["noise"] * self._rewards["noise"] +\
            # rewards["jumps"] * self._rewards["jumps"] +\
            # rewards["jumps-dist"] * self._rewards["jumps-dist"] +\
            # rewards["dist-win"] * self._rewards["dist-win"]
        # -----PCGRL Reward Method-----

    # Fun reward function
    # lower bound = 0.26
    # upper bound = 0.94
    def kl_fn(self, val):
        if (val < 0.26):
            return -(val-0.26)**2
        if (val > 0.94):
            return -(val-0.94)**2
        return 0

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
    
    def get_episode_over(self, new_stats=None, old_stat=None):
        return self.completion_rate < 1.0 or self.current_iteration == self._last_iteration

    def get_debug_info(self, new_stats, old_stats):
        return {
            # "dist-floor": new_stats["dist-floor"],
            # "disjoint-tubes": new_stats["disjoint-tubes"],
            # "enemies": new_stats["enemies"],
            # "empty": new_stats["empty"],
            # "noise": new_stats["noise"],
            # "jumps": new_stats["jumps"],
            # "jumps-dist": new_stats["jumps-dist"],
            # "dist-win": new_stats["dist-win"],
            "kl_val": self.kl_val,
            "completion_rate": self.completion_rate,
            "rew_F": self._rew_F,
            "rew_H": self._rew_H 
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

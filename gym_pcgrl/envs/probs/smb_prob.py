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
        self._width = 140 # original = 114; the width does not include the left 3 cols and right 3 cols
        self._height = 14
        self._prob = {"empty":0.75, "solid":0.1, "enemy":0.01, "brick":0.04, "question":0.01, "coin":0.02, "tube": 0.02}
        self._border_size = (3, 0)

        self._solver_power = 10000

        self._min_empty = 900
        self._min_enemies = 10
        self._max_enemies = 30
        self._min_jumps = 20

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

        self.initial_state = None
        self.nz = 32
        self.generator = Generator(random.randint(1, 10000000))
        self.repairer = Repairer(0) # passing in cuda_id=0 for only cpu
        self._start_block_num = 1
        self._end_block_num = 5
        self._cur_block_num = 1 # to tell which block iteration is on

        self.win_h, self.win_w = 14, 28
        self.sy, self.sx = 14, 7
        self.ny, self.nx = 0, 3

        self.F_que = deque(maxlen=1000)
        self.H_que = deque(maxlen=1000)


    def reset(self, start_stats):
        super().reset(start_stats)
        self._cur_block_num = self._start_block_num # to tell which block iteration is on


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
        return float(f.read())

    def update_rep_map_with_init_block(self, map):
        playable = False

        # Keep generate the initial block till it's playable
        while not playable:
            if self.initial_state != None:
                self.state = self.initial_state
            else:
                self.state = self.sample_random_vector(self.nz)

            st = time.time()
            piece = self.generator.generate(self.state)
            st = time.time()
            new_piece = self.repairer.repair(piece)

            # Pass in the generated piece to the Mario AI to check
            # if the new piece is playable
            self.saveLevelAsText(new_piece, rootpath + "mario_current_map")
            subprocess.call(['java', '-jar', rootpath + "Mario-AI-Framework.jar", rootpath + "mario_current_map.txt"])
            completion_rate = self.readMarioAIResultFile(rootpath + "\mario_result.txt")
            print("Initial Block Completion Rate: ", completion_rate)
            if completion_rate == 1.0:
                playable = True
            else:
                pass

        self.convertMP2PCGRL_num(new_piece)

        # print("Initial Block")
        # print(new_piece)

        map[:, :28] = new_piece

    def get_tile_types(self):
        return ["empty", "solid", "enemy", "brick", "question", "coin", "tube"]

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._min_empty = kwargs.get('min_empty', self._min_empty)
        self._min_enemies = kwargs.get('min_enemies', self._min_enemies)
        self._max_enemies = kwargs.get('max_enemies', self._max_enemies)
        self._min_jumps = kwargs.get('min_jumps', self._min_jumps)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

    def _convert_num_to_str_tiles(self, map):
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
        # Assign a char symbol to each tile
        gameCharacters=" # ## #"
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvlString = ""
        # Convert the string map to a char map
        for i in range(len(map)):
            # first 3 cols of rows 0-10 are empty
            if i < self._height - 3:
                lvlString += "   "
            # first 3 cols of row 11 are for the player tile
            elif i == self._height - 3:
                lvlString += " @ "
            # first 3 cols of rows 12-14 are solids
            else:
                lvlString += "###"
            # Go thru each entry in the map and copy and convert it
            for j in range(len(map[i])):
                # print(j)
                string = map[i][j]
                lvlString += string_to_char[string]
            # last 3 cols of rows 0-10 are for the pole tiles
            if i < self._height - 3:
                lvlString += " | "
            # last 3 cols of row 11 are for the pole base
            elif i == self._height - 3:
                lvlString += " # "
            # first 3 cols of rows 12-14 are solids
            else:
                lvlString += "###"
            # add a new line
            lvlString += "\n"

        # print(lvlString)
        # print("Length of lvlString: ", len(lvlString))

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

    def get_stats(self, map=None):
        # map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            # "dist-floor": get_floor_dist(map, ["enemy"], ["solid", "brick", "question", "tube_left", "tube_right"]),
            # "disjoint-tubes": get_type_grouping(map, ["tube"], [(-1,0),(1,0)],1,1),
            # "enemies": calc_certain_tile(map_locations, ["enemy"]),
            # "empty": calc_certain_tile(map_locations, ["empty"]),
            # "noise": get_changes(map, False) + get_changes(map, True),
            # "jumps": 0,
            # "jumps-dist": 0,
            # "dist-win": 0,
            "block-num": self._cur_block_num
        }
        # map_stats["dist-win"], play_stats = self._run_game(map)
        # map_stats["jumps"] = play_stats["jumps"]
        # prev_jump = 0
        # value = 0
        # for l in play_stats["jump_locs"]:
        #     value = max(value, l[0] - prev_jump)
        #     prev_jump = l[0]
        # value = max(value, self._width - prev_jump)
        # map_stats["jumps-dist"] = value
        return map_stats

    def get_reward(self, new_stats=None, old_stats=None, map=None): 

        reward, done = 0, False

        # Calculate the X start position based on the current block number
        now_x = 0 + 28 * self._cur_block_num

        # Convert the map to MarioPuzzle tile to run the Mario-AI framework
        # First, need to convert the num tiles to str tiles in PCG-RL, since some tiles like tubes are not distinguishable
        # Then, convert the pcg-rl str tiles to mariopuzzle num-tiles

        # Convert the pcg-rl num tiles to str tiles
        new_map = self._convert_num_to_str_tiles(get_string_map(map, self.get_tile_types()))
        new_map = np.array(new_map)

        # convert the pcg-rl str tiles to mariopuzzle num-tiles
        self.convertPCGRL_str2MP_num(new_map)
        # print(new_map)
        self.saveLevelAsText(new_map[:, max(0, now_x-3*self.win_w): now_x+self.win_w], rootpath + "mario_current_map")
        subprocess.call(['java', '-jar', rootpath + "Mario-AI-Framework.jar", rootpath + "mario_current_map.txt"])
        completion_rate = self.readMarioAIResultFile(rootpath + "\mario_result.txt")
        reward += completion_rate
        print("completion rate: ", completion_rate)

        # calculate the diversity
        kl_val = KLWithSlideWindow(
            self.lv, (0, now_x, self.win_h, self.win_w), self.sx, self.nx, self.sy, self.ny)
        
        # need to clear the F_que when we move to the next block section
        # calculate fun 
        rew_F = self.add_then_norm(self.kl_fn(kl_val), self.F_que)
        reward += rew_F

        # calculate historical deviation
        piece_map = lv2Map(new_map[:, now_x : now_x + self.win_w])
        novelty = self.cal_novelty(piece_map)
        rew_H = self.add_then_norm(novelty, self.H_que)
        reward += rew_H
        
        #calculate the total reward
        return 0

    def cal_novelty(self, piece):
        score = []
        for x in self.pop:
            score.append(calKLFromMap(x, piece))
        score.sort()
        sum = 0
        # novel_k = 10
        # only consider top 10 most similar blocks
        siz = min(len(score), self.novel_k)
        for i in range(siz):
            sum += score[i]
        if siz > 0:
            sum /= siz
        return sum

    def add_then_norm(self, value, history):
        if not self.norm:
            return value
        history.append(value)
        maxv = max(history)
        minv = min(history)
        if maxv == minv:
            return 0
        else:
            return (value-minv)/(maxv-minv)
    
    # def get_reward(self, iterations, action, new_stats, old_stats):
    #     print("Iterations: ", iterations)

    #     # Return a huge negative reward if an action if out of range
    #     if iterations >= 0 or iterations <= 999:
    #         pass
    #     elif iterations >= 1000 or iterations <= 1999:
    #         pass
    #     elif iterations >= 2000 or iterations <= 2999:
    #         pass
    #     elif iterations >= 3000 or iterations <= 3999:
    #         pass
    #     elif iterations >= 4000 or iterations <= 4999:
    #         pass

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["dist-win"] <= 0

    def get_debug_info(self, new_stats, old_stats):
        return {
            "dist-floor": new_stats["dist-floor"],
            "disjoint-tubes": new_stats["disjoint-tubes"],
            "enemies": new_stats["enemies"],
            "empty": new_stats["empty"],
            "noise": new_stats["noise"],
            "jumps": new_stats["jumps"],
            "jumps-dist": new_stats["jumps-dist"],
            "dist-win": new_stats["dist-win"]
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

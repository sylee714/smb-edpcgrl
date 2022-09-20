from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, get_floor_dist, get_type_grouping, get_changes
from gym_pcgrl.envs.probs.smb.engine import State,BFSAgent,AStarAgent


class SMB2Problem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 114
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
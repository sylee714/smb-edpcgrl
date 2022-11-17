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

repairer = Repairer(0) # passing in cuda_id=0 for only cpu

new_piece = repairer.repair(piece)
import gym
import gym_pcgrl
import time 
import numpy as np

# https://www.youtube.com/watch?v=dLP-2Y6yu70&ab_channel=sentdex
if __name__ == '__main__':
    from gym_pcgrl.envs.probs.MarioLevelRepairer.CNet.model import CNet
# env = gym.make('zelda-narrow-v0')
# env.reset()

# # model = PPO2(CnnLnLstmPolicy, env, nminibatches=1, verbose=1)
# # model.learn(total_timesteps=10000)

# # episodes = 10
# # for ep in range(episodes):
# #     obs = env.reset()
# #     done = False
# while True:    
#     env.render()
#     obs, reward, done, info = env.step(env.action_space.sample())

# env.close()

# Checking how it gets printed 
# def get_tile_types():
#     return ["empty", "solid", "enemy", "brick", "question", "coin", "tube"]

# gameCharacters=" # ## #"
# string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(get_tile_types()))
# print(string_to_char)

# height = 14
# width = 30

# rows = []
# cols = []

# for i in range(height):
#     cols = []
#     for j in range(width):
#         cols.append(" ")
#     rows.append(cols)

# for row in rows:
#     print(row)

# lvlString = ""
# for i in range(height):
#         if i < height - 3:
#             lvlString += "   "
#         elif i == height - 3:
#             lvlString += " @ "
#         else:
#             lvlString += "###"
#         for j in range(width):
#             # string = map[i][j]
#             # lvlString += string_to_char[string]
#             lvlString += " "
#         if i < height - 3:
#             lvlString += " | "
#         elif i == height - 3:
#             lvlString += " # "
#         else:
#             lvlString += "###"
#         lvlString += "\n"

# print(lvlString)
# print(len(lvlString))

# add "time.sleep()" to slow the frame rate

# --------------------------------

    env = gym.make('smb-wide-v0')

    # Observation and action space 
    obs_space = env.observation_space
    action_space = env.action_space
    # print("The observation space: {}".format(obs_space))
    # print("The action space: {}".format(action_space))

    obs = env.reset()
    for t in range(1):
        action = env.action_space.sample()
        # print("action: ", action)
        # print("action type: ", type(action))
        obs, reward, done, info = env.step(env.action_space.sample()) # [0, 0, 2]
        # print("obs: ", obs)
        env.render('human')
        time.sleep(1)
        # if done:
        #     print("Episode finished after {} timesteps".format(t+1))
        #     break
    



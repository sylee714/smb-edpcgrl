import gym
import gym_pcgrl
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy

# https://www.youtube.com/watch?v=dLP-2Y6yu70&ab_channel=sentdex

env = gym.make('smb-narrow-v0')
env.reset()

# model = PPO2(CnnLnLstmPolicy, env, nminibatches=1, verbose=1)
# model.learn(total_timesteps=10000)

# episodes = 10
# for ep in range(episodes):
#     obs = env.reset()
#     done = False
while True:    
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample())

env.close()

# def get_tile_types():
#     return ["empty", "solid", "enemy", "brick", "question", "coin", "tube"]

# gameCharacters=" # ## #"
# string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(get_tile_types()))
# print(string_to_char)

# height = 14
# width = 30

# rows = []
# # cols = []

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

"""
Run a trained agent and get generated maps
"""
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs
from gym_pcgrl.envs.probs.MarioLevelRepairer.CNet.model import CNet

def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10
    kwargs['render'] = True

    agent = PPO2.load(model_path)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    obs = env.reset()
    dones = False
    # for i in range(kwargs.get('trials', 1)):
    for i in range(1):
        while not dones:
            action, _ = agent.predict(obs)
            obs, _, dones, info = env.step(action)
            # if kwargs.get('verbose', False):
            #     print(info[0])
        # time.sleep(0.2)   

################################## MAIN ########################################
game = 'smb'
representation = 'wide'
# model_path = 'models/{}/{}/best_model.pkl'.format(game, representation)
model_path = 'models/mingkie2222/best_model.pkl'
kwargs = {
    'change_percentage': 0.1,
    # 'change_percentage': 0.9,
    # 'trials': 1,
    'verbose': True
}

# if __name__ == '__main__':
# infer(game, representation, model_path, **kwargs)
env_name = '{}-{}-v0'.format(game, representation)
if game == "binary":
    model.FullyConvPolicy = model.FullyConvPolicyBigMap
    kwargs['cropped_size'] = 28
elif game == "zelda":
    model.FullyConvPolicy = model.FullyConvPolicyBigMap
    kwargs['cropped_size'] = 22
elif game == "sokoban":
    model.FullyConvPolicy = model.FullyConvPolicySmallMap
    kwargs['cropped_size'] = 10
kwargs['render'] = True

agent = PPO2.load(model_path)
env = make_vec_envs(env_name, representation, None, 1, **kwargs)
obs = env.reset()
dones = False
# for i in range(kwargs.get('trials', 1)):
for i in range(1):
    while not dones:
        action, _ = agent.predict(obs)
        obs, _, dones, info = env.step(action)

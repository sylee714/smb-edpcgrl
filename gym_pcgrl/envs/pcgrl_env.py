from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
import numpy as np
import gym
from gym import spaces
import PIL

"""
The PCGRL GYM Environment
"""
class PcgrlEnv(gym.Env):
    """
    The type of supported rendering
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """
    def __init__(self, prob="binary", rep="narrow"):
        self._prob_str = prob
        self._prob = PROBLEMS[prob]()
        self._rep = REPRESENTATIONS[rep]()
        self._rep_stats = None
        self._change_rate = 0.3
        self._iteration = 0
        self._cur_block = 1
        self._changes = 0

        if self._prob_str == "smb":
            # excluding the initial block
            self._max_changes = max(int(self._change_rate * (self._prob._width - 28) * self._prob._height), 1)
            self._max_changes_per_block = self._max_changes // 5
        else:
            self._max_changes = max(int(self._change_rate * self._prob._width * self._prob._height), 1)

        if self._prob_str == "smb":
            # excluding the initial block
            self._max_iterations = self._max_changes * (self._prob._width - 28) * self._prob._height
            self._max_iterations_per_block = self._max_iterations // 5
        else:
            self._max_iterations = self._max_changes * self._prob._width * self._prob._height

        self.seed()
        self.viewer = None

        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int[]: An array of 1 element (the used seed)
    """
    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """
    def reset(self):

        self._iteration = 0
        self._cur_block = 1
        self._changes = 0

        # Initial map gets generated in Representation and it's stored in Rep
        # So, update the initial map with the generated segment
        self._rep.reset(self._prob._width, self._prob._height, 
                        get_int_prob(self._prob._prob, 
                        self._prob.get_tile_types()), 
                        self._prob.win_w, self._prob.win_h)

        self._prob.reset(self._rep_stats)

        if self._prob_str == "smb":
            self._prob.init_map(self._rep._map)
        
        self._rep_stats = self._prob.get_stats(str_map=get_string_map(self._rep._map, self._prob.get_tile_types()), 
                                                    num_map=self._rep._map,
                                                    cur_block=self._cur_block)
        
        observation = self._rep.get_observation()
        return observation

    """
    Get the border tile that can be used for padding

    Returns:
        int: the tile number that can be used for padding
    """
    def get_border_tile(self):
        return self._prob.get_tile_types().index(self._prob._border_tile)

    """
    Get the number of different type of tiles that are allowed in the observation

    Returns:
        int: the number of different tiles
    """
    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    """
    Adjust the used parameters by the problem or representation

    Parameters:
        change_percentage (float): a value between 0 and 1 that determine the
        percentage of tiles the algorithm is allowed to modify. Having small
        values encourage the agent to learn to react to the input screen.
        **kwargs (dict(string,any)): the defined parameters depend on the used
        representation and the used problem
    """
    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            if self._prob_str == "smb":
                # excluding the initial block
                self._max_changes = max(int(percentage * (self._prob._width - 28) * self._prob._height), 1)
            else:
                self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)

        if self._prob_str == "smb":
            # excluding the initial block
            self._max_iterations = self._max_changes * (self._prob._width - 28) * self._prob._height
        else:
            self._max_iterations = self._max_changes * self._prob._width * self._prob._height

        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())

    """
    Advance the environment using a specific action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """
    def step(self, action):
        self._iteration += 1

        #save copy of the old stats to calculate the reward
        self._old_stats = self._rep_stats
        
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action, cur_block=self._cur_block)
        
        # if there is a change, get the new stats
        if change > 0:
            self._changes += change
            self._rep_stats = self._prob.get_stats(str_map=get_string_map(self._rep._map, self._prob.get_tile_types()), 
                                                    num_map=self._rep._map,
                                                    cur_block=self._cur_block)
        
        observation = self._rep.get_observation()
        reward = self._prob.get_reward(new_stats=self._rep_stats, old_stats=self._old_stats)

        # move to the next block when changes >= change limit
        self._cur_block = 1 + (self._changes // self._max_changes_per_block)

        done = self._prob.get_episode_over(new_stats=self._rep_stats) or self._changes >= self._max_changes or self._iteration >= self._max_iterations or self._cur_block >= 6

        info = self._prob.get_debug_info(self._rep_stats, self._old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        info["max_iterations_per_block"] = self._max_iterations_per_block
        info["max_changes_per_block"] = self._max_changes_per_block

        return observation, reward, done, info

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """
    def render(self, mode='human'):
        tile_size=16
        img = self._prob.render(get_string_map(self._rep._map, self._prob.get_tile_types()))
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)
            return self.viewer.isopen

    """
    Close the environment
    """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

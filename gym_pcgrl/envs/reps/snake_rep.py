from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict

"""
This representation is solely inteded to be used with SMB
The snake representation where the agent starts at (x=28, y=0) and changes the tile value at each update.
It goes to left till x=55 and goes moves down and goes to right till x=0. It repeats this pattern until
it reaches (0, 28), which covers the second block.
Then, for the next block it starts at (x=56, y=0) and follows the same pattern.
For the remaining blocks, it does the same thing.
"""
class SnakeRepresentation(Representation):
    """
    Initialize all the parameters used by that representation
    """
    def __init__(self):
        super().__init__()
        self._x = 28
        self._y = 13
        self._iteration = 0

    """
    Gets the action space used by the narrow representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Discrete: the action space used by that narrow representation which
        correspond to which value for each tile type
    """
    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete(num_tiles)

    """
    Resets the current representation where it resets the parent and the current
    modified location

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
    """
    def reset(self, width, height, prob, win_width=0, win_height=0):
        super().reset(width, height, prob)
        self._x = 28
        self._y = 13
        self._iteration = 0

        self._width = width
        self._height = height

        self._win_width = win_width
        self._win_height = win_height

        self._up_point_list = []
        # need to change the logic to find the up points not down points
        for i in range(height):
            if i % 2 == 0:
                self._up_point_list.append((0, i))
            else:
                self._up_point_list.append((win_width-1, i))

    """
    Get the observation space used by the narrow representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Dict: the observation space used by that representation. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))
        })

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._x, self._y], dtype=np.uint8),
            "map": self._map.copy()
        })

    """
    Update the wide representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        change = 0
        # Check if it reached the end point of the last block
        # if self._x != self._width - self._win_width or self._y != self._height - 1:
            # if the it's the same tile, return True -> 1; otherwise, return False -> 0
        change = [0,1][self._map[self._y][self._x] != action]
        self._map[self._y][self._x] = action
        
        # Update the x and y
        # Check if it reached the end point of the last block
        # if self._x != self._width - self._win_width or self._y != self._height - 1:
        if self._x != self._width - self._win_width or self._y != 0:
            # If it reached the end point of the current block, then move to the next block
            # if self._x % self._win_width == 0 and self._y % self._win_height == self._win_height - 1:
            if self._x % self._win_width == 0 and self._y % self._win_height == 0:
                self._y = 13
                self._x += self._win_width
            else:
                # if it's a up point, then move up
                if (self._x % self._win_width, self._y % self._win_height) in self._up_point_list:
                    self._y -= 1
                else:
                    # move to left
                    if self._y % 2 == 0:
                        self._x -= 1
                    # move to right
                    else:
                        self._x += 1

        return change, self._x, self._y

    """
    Modify the level image with a red rectangle around the tile that is
    going to be modified

    Parameters:
        lvl_image (img): the current level_image without modifications
        tile_size (int): the size of tiles in pixels used in the lvl_image
        border_size ((int,int)): an offeset in tiles if the borders are not part of the level

    Returns:
        img: the modified level image
    """
    def render(self, lvl_image, tile_size, border_size):
        x_graphics = Image.new("RGBA", (tile_size,tile_size), (0,0,0,0))
        for x in range(tile_size):
            x_graphics.putpixel((0,x),(255,0,0,255))
            x_graphics.putpixel((1,x),(255,0,0,255))
            x_graphics.putpixel((tile_size-2,x),(255,0,0,255))
            x_graphics.putpixel((tile_size-1,x),(255,0,0,255))
        for y in range(tile_size):
            x_graphics.putpixel((y,0),(255,0,0,255))
            x_graphics.putpixel((y,1),(255,0,0,255))
            x_graphics.putpixel((y,tile_size-2),(255,0,0,255))
            x_graphics.putpixel((y,tile_size-1),(255,0,0,255))
        lvl_image.paste(x_graphics, ((self._x+border_size[0])*tile_size, (self._y+border_size[1])*tile_size,
                                        (self._x+border_size[0]+1)*tile_size,(self._y+border_size[1]+1)*tile_size), x_graphics)

        self._iteration = self._iteration + 1
        lvl_image.save("snake_rep_images/lvl_img_{}.png".format(self._iteration))

        return lvl_image
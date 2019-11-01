import os
from PIL import Image
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.probs.helper import calc_certain_tile, calc_num_regions
from gym_pcgrl.envs.probs.mdungeon.engine import State,BFSAgent,AStarAgent

class MDungeonProblem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 7
        self._height = 11
        self._prob = {"empty":0.4, "solid": 0.4, "player":0.02, "exit":0.02, "potion":0.03, "treasure":0.03, "goblin":0.05, "ogre": 0.05}

        self._max_enemies = 6
        self._max_potions = 2
        self._max_treasures = 3

        self._target_col_enemies = 0.5
        self._target_solution = 20

        self._rewards = {
            "player": 5,
            "exit": 5,
            "potions": 1,
            "treasures": 1,
            "enemies": 5,
            "regions": 5,
            "col-enemies": 2,
            "dist-win": 1,
            "sol-length": 1
        }

    def get_tile_types(self):
        return ["empty", "solid", "player", "exit", "potion", "treasure", "goblin", "ogre"]

    def adjust_param(self, **kwargs):
        self._width, self._height = kwargs.get('width', self._width), kwargs.get('height', self._height)
        self._prob["empty"] = kwargs.get('empty_prob', self._prob["empty"])
        self._prob["solid"] = kwargs.get('solid_prob', self._prob["solid"])
        self._prob["player"] = kwargs.get('player_prob', self._prob["player"])
        self._prob["exit"] = kwargs.get('exit_prob',self._prob["exit"])
        self._prob["potion"] = kwargs.get('potion_prob', self._prob["potion"])
        self._prob["treasure"] = kwargs.get('treasure_prob', self._prob["treasure"])
        self._prob["goblin"] = kwargs.get('goblin_prob', self._prob["goblin"])
        self._prob["ogre"] = kwargs.get('ogre_prob', self._prob["ogre"])

        self._max_enemies = kwargs.get('max_enemies', self._max_enemies)
        self._max_potions = kwargs.get('max_potions', self._max_potions)
        self._max_treasures = kwargs.get('max_treasures', self._max_treasures)

        self._target_col_enemies = kwargs.get('min_col_enemies', self._target_col_enemies)
        self._target_solution = kwargs.get('target_solution', self._target_solution)
        self._rewards = {
            "player": kwargs.get("reward_player", self._rewards["player"]),
            "exit": kwargs.get("reward_exit", self._rewards["exit"]),
            "potions": kwargs.get("reward_potions", self._rewards["potions"]),
            "treasures": kwargs.get("reward_treasures", self._rewards["treasures"]),
            "enemies": kwargs.get("reward_enemies", self._rewards["enemies"]),
            "regions": kwargs.get("reward_regions", self._rewards["regions"]),
            "col-enemies": kwargs.get("reward_col_enemies", self._rewards["col-enemies"]),
            "dist-win": kwargs.get("reward_dist_win", self._rewards["dist-win"]),
            "sol-length": kwargs.get("reward_sol_length", self._rewards["sol-length"])
        }

    def _run_game(self, map):
        gameCharacters=" #@H*$go"
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvlString = ""
        for x in range(self._width+2):
            lvlString += "#"
        lvlString += "\n"
        for i in range(len(map)):
            for j in range(len(map[i])):
                string = map[i][j]
                if j == 0:
                    lvlString += "#"
                lvlString += string_to_char[string]
                if j == self._width-1:
                    lvlString += "#\n"
        for x in range(self._width+2):
            lvlString += "#"
        lvlString += "\n"

        state = State()
        state.stringInitialize(lvlString.split("\n"))

        aStarAgent = AStarAgent()
        bfsAgent = BFSAgent()

        sol,solState,iters = aStarAgent.getSolution(state, 1, 5000)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = aStarAgent.getSolution(state, 0.5, 5000)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = aStarAgent.getSolution(state, 0, 5000)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()
        sol,solState,iters = bfsAgent.getSolution(state, 5000)
        if solState.checkWin():
            return 0, len(sol), solState.getGameStatus()

        return solState.getHeuristic(), 0, solState.getGameStatus()

    def get_stats(self, map):
        map_stats = {
            "player": calc_certain_tile(map, ["player"]),
            "exit": calc_certain_tile(map, ["exit"]),
            "potions": calc_certain_tile(map, ["potion"]),
            "treasures": calc_certain_tile(map, ["treasure"]),
            "enemies": calc_certain_tile(map, ["goblin","ogre"]),
            "regions": calc_num_regions(map, ["empty","player","exit","potion","treasure","goblin","ogre"]),
            "col-potions": 0,
            "col-treasures": 0,
            "col-enemies": 0,
            "dist-win": self._width * self._height,
            "sol-length": 0
        }
        if map_stats["player"] == 1:
            if map_stats["regions"] == 1:
                map_stats["dist-win"], map_stats["sol-length"], play_stats = self._run_game(map)
                map_stats["col-potions"] = play_stats["col_potions"]
                map_stats["col-treasures"] = play_stats["col_treasures"]
                map_stats["col-enemies"] = play_stats["col_enemies"]
        return map_stats

    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": 0,
            "exit": 0,
            "potions": 0,
            "treasures": 0,
            "enemies": 0,
            "regions": 0,
            "col-enemies": 0,
            "dist-win": 0,
            "sol-length": 0
        }
        #calculate the player reward (only one player)
        rewards["player"] = old_stats["player"] - new_stats["player"]
        if rewards["player"] > 0 and new_stats["player"] == 0:
            rewards["player"] *= -1
        elif rewards["player"] < 0 and new_stats["player"] == 1:
            rewards["player"] *= -1
        #calculate the exit reward (only one exit)
        rewards["exit"] = old_stats["exit"] - new_stats["exit"]
        if rewards["exit"] > 0 and new_stats["exit"] == 0:
            rewards["exit"] *= -1
        elif rewards["exit"] < 0 and new_stats["exit"] == 1:
            rewards["exit"] *= -1
        #calculate enemies reward (between 1 and max_enemies)
        rewards["enemies"] = old_stats["enemies"] - new_stats["enemies"]
        if rewards["enemies"] < 0 and old_stats["enemies"] == 0:
            rewards["enemies"] *= -1
        elif new_stats["enemies"] >= 1 and new_stats["enemies"] <= self._max_enemies and\
                old_stats["enemies"] >= 1 and old_stats["enemies"] <= self._max_enemies:
            rewards["enemies"] = 0
        #calculate potions reward (less than max potions)
        rewards["potions"] = old_stats["potions"] - new_stats["potions"]
        if new_stats["potions"] <= self._max_potions and old_stats["potions"] <= self._max_potions:
            rewards["potions"] = 0
        #calculate treasure reward (less than max treasures)
        rewards["treasures"] = old_stats["treasures"] - new_stats["treasures"]
        if new_stats["treasures"] < self._max_treasures and old_stats["treasures"] <= self._max_treasures:
            rewards["treasures"] = 0
        #calculate regions reward (only one region)
        rewards["regions"] = old_stats["regions"] - new_stats["regions"]
        if new_stats["regions"] == 0 and old_stats["regions"] > 0:
            rewards["regions"] = -1
        #calculate ratio of killed enemies
        new_col = new_stats["col-enemies"] / max(new_stats["enemies"], 1)
        old_col = old_stats["col-enemies"] / max(old_stats["enemies"], 1)
        rewards["col-enemies"] = new_col - old_col
        #calculate distance remaining to win
        rewards["dist-win"] = old_stats["dist-win"] - new_stats["dist-win"]
        #calculate solution length
        rewards["sol-length"] = new_stats["sol-length"] - old_stats["sol-length"]
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["exit"] * self._rewards["exit"] +\
            rewards["enemies"] * self._rewards["enemies"] +\
            rewards["treasures"] * self._rewards["treasures"] +\
            rewards["potions"] * self._rewards["potions"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["col-enemies"] * self._rewards["col-enemies"] +\
            rewards["dist-win"] * self._rewards["dist-win"] +\
            rewards["sol-length"] * self._rewards["sol-length"]

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["sol-length"] >= self._target_solution and\
                new_stats["enemies"] > 0 and\
                new_stats["col-enemies"] / max(1,new_stats["enemies"]) > self._target_col_enemies

    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "exit": new_stats["exit"],
            "potions": new_stats["potions"],
            "treasures": new_stats["treasures"],
            "enemies": new_stats["enemies"],
            "regions": new_stats["regions"],
            "col-potions": new_stats["col-potions"],
            "col-treasures": new_stats["col-treasures"],
            "col-enemies": new_stats["col-enemies"],
            "dist-win": new_stats["dist-win"],
            "sol-length": new_stats["sol-length"]
        }

    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/mdungeon/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/mdungeon/solid.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/mdungeon/player.png").convert('RGBA'),
                "exit": Image.open(os.path.dirname(__file__) + "/mdungeon/exit.png").convert('RGBA'),
                "potion": Image.open(os.path.dirname(__file__) + "/mdungeon/potion.png").convert('RGBA'),
                "treasure": Image.open(os.path.dirname(__file__) + "/mdungeon/treasure.png").convert('RGBA'),
                "goblin": Image.open(os.path.dirname(__file__) + "/mdungeon/goblin.png").convert('RGBA'),
                "ogre": Image.open(os.path.dirname(__file__) + "/mdungeon/ogre.png").convert('RGBA'),
            }
        return super().render(map)
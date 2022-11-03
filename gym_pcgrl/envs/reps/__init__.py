from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.reps.narrow_cast_rep import NarrowCastRepresentation
from gym_pcgrl.envs.reps.narrow_multi_rep import NarrowMultiRepresentation
from gym_pcgrl.envs.reps.wide_rep import WideRepresentation
from gym_pcgrl.envs.reps.turtle_rep import TurtleRepresentation
from gym_pcgrl.envs.reps.turtle_cast_rep import TurtleCastRepresentation
from gym_pcgrl.envs.reps.snake_rep import SnakeRepresentation
from gym_pcgrl.envs.reps.up_right_rep import UpRightRepresentation
from gym_pcgrl.envs.reps.exp_rep import ExperienceRepresentation

# all the representations should be defined here with its corresponding class
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    "narrowcast": NarrowCastRepresentation,
    "narrowmulti": NarrowMultiRepresentation,
    "wide": WideRepresentation,
    "turtle": TurtleRepresentation,
    "turtlecast": TurtleCastRepresentation,
    "snake": SnakeRepresentation,
    "up-right": UpRightRepresentation,
    "exp": ExperienceRepresentation
}

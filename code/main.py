import numpy as np

from GridWorld import GridWorld
from Agent import Agent

np.set_printoptions(suppress=True)

if __name__ == "__main__":
    np.random.seed(486)

    rewards = np.array([
        [-0.04, -0.04, -0.04, -0.04],
        [-0.04, -0.04, -0.04, -1.00],
        [-0.04, -0.04, -0.04, 1.000]
    ])

    start_state = (0, 0)
    goal_states = [(2, 3), (1, 3)]
    walls = [(1, 1), (0, 2)]

    env = GridWorld(world_height=3, world_width=4,
                    prob_direct=0.8, prob_lateral=0.1,
                    rewards=rewards,
                    start_state=start_state,
                    goal_states=goal_states,
                    walls=walls)

    agent = Agent(env)

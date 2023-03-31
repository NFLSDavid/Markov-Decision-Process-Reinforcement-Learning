import numpy as np

from GridWorld import GridWorld
from Agent import Agent
import matplotlib.pyplot as plt

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
    V, _ = agent.value_iteration(0.99)
    pi = agent.find_policy(V)
    print(V)
    # print(pi)
    V_adp, all_Vs = agent.passive_adp(pi, 0.99, adp_iters=1000)
    print(V_adp)
    all_Vs = np.array(all_Vs)


    # print(all_Vs.shape)

    # print(all_Vs[1])

    def plot(i, j):
        data = all_Vs[::, i, j]
        # print(data)
        plt.plot(data, label="V({},{})".format(i, j))


    plot(0, 0)
    plot(0, 3)
    plot(2, 2)
    plot(1, 2)
    plt.title("Passive ADP")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

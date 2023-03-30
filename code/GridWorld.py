import numpy as np
from typing import Tuple, List, Union
from utils import Action, Move
from GridWorldTemplate import GridWorldTemplate


class GridWorld(GridWorldTemplate):

    def __init__(self, world_height: int, world_width: int,
                 prob_direct: float, prob_lateral: float, rewards: np.ndarray,
                 start_state: Tuple[int, int], goal_states: List[Tuple[int, int]],
                 walls: Union[None, List[Tuple[int, int]]] = None):
        """
        Constructor function.

        :param world_height: number of rows in world
        :param world_width: number of columns in world
        :param prob_direct: probability of going towards the intended direction
        :param prob_lateral: probability of going lateral to the intended direction
        :param rewards: a 2D Numpy array of rewards when entering each state
        :param start_state: state to start in
        :param goal_states: states that terminate the simulation
        :param walls: coordinates of states to be considered as walls
        """
        super().__init__(world_height, world_width,
                         prob_direct, prob_lateral, rewards,
                         start_state, goal_states, walls=walls)

    def fill_T(self) -> None:
        """
        Initializes and populates transition probabilities T(s'|s,a) for all possible (s, a, s').
        The computed transition probabilities are stored in self.T.

        Usage:
        >> env = GridWorld(2, 3, ...)
        >> print(env.T)  # None
        >> env.fill_T()
        >> print(env.T.shape)
        (2, 3, 4, 2, 3)
        """
        self.T = np.zeros((self.state_dim[0], self.state_dim[1], self.action_dim, self.state_dim[0], self.state_dim[1]))
        for i in range(self.state_dim[0]):
            for j in range(self.state_dim[1]):
                for action in Action.space():
                    for res_action in (action, (action + 1) % 4, (action + 3) % 4):
                        dest_i, dest_j = Move.step((i, j), res_action)
                        if dest_i < 0 or dest_j < 0 or \
                                self.state_dim[0] <= dest_i or self.state_dim[1] <= dest_j or \
                                (dest_i, dest_j) in self.walls:
                            dest_i, dest_j = i, j
                        if res_action == action:
                            self.T[i][j][action][dest_i][dest_j] += self.prob_direct
                        else:
                            self.T[i][j][action][dest_i][dest_j] += self.prob_lateral
        return

    def make_move(self,
                  state: Tuple[int, int],
                  action: int) -> \
            Tuple[Tuple[int, int], float]:
        """
        Takes a single step in the environment under stochasticity based on the transition probabilities.

        Usage:
        >> env  # created; prob_direct = 0.5, prob_lateral = 0.25
        >> env.make_move((0, 0), Action.DOWN)
        ((1, 0), -0.04)
        >> env.make_move((0, 0), Action.DOWN)  # same start state but moved laterally because of randomness
        ((0, 1), -0.04)

        :param state: starting state
        :param action: action to taken
        :return: next state and observed reward entering the next state
        """
        res_action = np.random.choice([action, (action + 1) % 4, (action + 3) % 4],
                                      p=[self.prob_direct, self.prob_lateral, self.prob_lateral])
        dest_i, dest_j = Move.step(state, res_action)
        if dest_i < 0 or dest_j < 0 or \
                self.state_dim[0] <= dest_i or self.state_dim[1] <= dest_j or \
                (dest_i, dest_j) in self.walls:
            dest_i, dest_j = state
        return (dest_i, dest_j), self.R[dest_i][dest_j]

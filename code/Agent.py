import venv

import numpy as np
from typing import Tuple, Union, List, TypeVar
from utils import Action, Move
from AgentTemplate import AgentTemplate

GridWorld_Type = TypeVar("GridWorld")


class Agent(AgentTemplate):

    def __init__(self, env: GridWorld_Type):
        """
        Constructor function.

        :param env: environment to put the agent in
        """
        super().__init__(env)

    def value_iteration(self,
                        gamma: float,
                        tolerance: float = 0.001,
                        max_iter: int = np.inf) -> \
            Tuple[np.ndarray, int]:
        """
        Performs value iteration on the loaded environment. The tolerance parameter specifies the
        maximum absolute difference of all entries between the current V and the previous V
        before the iteration can stop. The max_iter parameter is a hard stopping criterion
        where the iteration must stop once the maximum number of iterations is reached. The tolerance
        and max_iter parameter should be used as a conjunction.

        Usage:
        >> V, i = agent.value_iteration(0.99)

        :param gamma: discount factor
        :param tolerance: terminating condition for the iteration loop
        :param max_iter: maximum allowable iterations (should be set to np.inf unless you're getting infinite loops)
        :return: optimal value array and number of iterations
        """
        idx = 0
        V = np.zeros(self.env.state_dim)
        for i in self.env.goal_states:
            V[i] = self.env.R[i]
        while idx < max_iter:
            cur_tolerance = 0
            updated_V = np.zeros(self.env.state_dim)
            pi = self.find_policy(V)
            for i in range(self.env.state_dim[0]):
                for j in range(self.env.state_dim[1]):
                    if (i, j) in self.env.walls:
                        continue
                    elif (i, j) in self.env.goal_states:
                        updated_V[i][j] = self.env.R[i][j]
                    else:
                        tmp_sum = 0
                        for dest_i in range(self.env.state_dim[0]):
                            for dest_j in range(self.env.state_dim[1]):
                                tmp_sum += (self.env.T[i][j][pi[i][j]][dest_i][dest_j] * V[dest_i][dest_j])
                        updated_V[i][j] = self.env.R[i][j] + gamma * tmp_sum
                    cur_tolerance = max(cur_tolerance, abs(updated_V[i][j] - V[i][j]))
            idx += 1
            V = updated_V
            if cur_tolerance <= tolerance:
                break
        return V, idx

    def find_policy(self, V: np.ndarray) -> np.ndarray:
        """
        Finds the best action to take at each state based on the state values obtained,
        i.e. computing pi(s) = argmax_a(sum_{s'}(P(s'|s, a) V(s'))) for all s

        :param V: state value array to extract the policy from
        :return: policy array
        """
        pi = np.zeros(self.env.state_dim, dtype=int)
        self.env.fill_T()
        for i in range(self.env.state_dim[0]):
            for j in range(self.env.state_dim[1]):
                if (i, j) not in self.env.walls:
                    if (i, j) not in self.env.goal_states:
                        pi[i][j] = np.argmax([np.sum([self.env.T[i][j][action][dest_i][dest_j] * V[dest_i][dest_j]
                                                      for dest_i in range(self.env.state_dim[0])
                                                      for dest_j in range(self.env.state_dim[1])])
                                              for action in Action.space()])
        return pi

    def passive_adp(self,
                    policy: np.ndarray,
                    gamma: float,
                    adp_iters: int = 1000) -> \
            Tuple[np.ndarray, List[np.ndarray]]:
        """
        Performs passive ADP on a given policy using simulations from the environment.

        Note:
        * You should be using the GridWorld.simulate method which requires the GridWorld.make_move
        to be fully implemented and working.
        * You are not allowed to use the true transition probabilities from the environment (i.e. env.T) but
        instead you should learn an estimate of the transition probabilities through experiences
        from GridWorld.simulate.

        Usage:
        >> V, _ = agent.value_iteration(0.99)
        >> pi = agent.find_policy(V)
        >> V_adp, all_Vs = agent.passive_adp(pi, 0.99)

        :param policy: policy to determine action to take for each state
        :param gamma: discount factor
        :param adp_iters: number of the passive ADP iterations
        :return: state value array based on the given policy, and all Vs per iteration
        """
        V_adp = np.zeros(self.env.state_dim)
        all_Vs = [V_adp]
        Rewards = np.zeros(self.env.state_dim)
        action_num = len(Action.space())
        x: int = self.env.state_dim[0]
        y: int = self.env.state_dim[1]
        N = np.zeros((x, y, action_num, x, y))
        P = np.zeros((x, y, action_num, x, y))
        for idx in range(adp_iters):
            updated_V = V_adp.copy()
            for state in self.env.goal_states:
                updated_V[state] = V_adp[state]
            cur_state, callback_fn = self.env.simulate()
            done = False
            while not done:
                # step 2:
                cur_action = policy[cur_state]
                time_step, next_state, reward, done = callback_fn(cur_action)

                # step 3:
                Rewards[next_state] = reward

                # step 4:
                N[cur_state][cur_action][next_state] += 1
                sa_num = N[cur_state][cur_action].sum()
                for i in range(x):
                    for j in range(y):
                        P[cur_state][cur_action][i][j] = N[cur_state][cur_action][i][j] / sa_num
                cur_state = next_state

                # step 5:
                for i in range(x):
                    for j in range(y):
                        if (i, j) in self.env.walls:
                            continue
                        elif (i, j) in self.env.goal_states:
                            updated_V[i][j] = Rewards[i][j]
                        else:
                            tmp_sum = 0
                            for dest_i in range(x):
                                for dest_j in range(y):
                                    tmp_sum += (P[i][j][policy[i][j]][dest_i][dest_j] * V_adp[dest_i][dest_j])
                            updated_V[i][j] = Rewards[i][j] + gamma * tmp_sum

                V_adp = updated_V.copy()
            all_Vs.append(V_adp)
        return V_adp, all_Vs

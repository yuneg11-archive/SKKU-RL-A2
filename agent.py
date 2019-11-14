import numpy as np
import random


class Agent:
    def __init__(self, Q, mode="mc_control", nA=6, alpha=0.01, gamma=0.99):
        self.Q = Q
        self.mode = mode
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        if mode == "mc_control":
            self.step = self.step_mc_control
        elif mode == "q_learning":
            self.step = self.step_q_learning
            self.alpha = 0.2  # Optimal
            self.gamma = 0.8  # Optimal

    def select_action(self, state, eps):
        """
        Params
        ======
        - state: the current state of the environment
        - eps: the threshold value to decide exploration or exploitation

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > eps:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(self.nA)

    def step_mc_control(self, state, action, reward, next_state, done):
        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        pass

    def step_q_learning(self, state, action, reward, next_state, done):
        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])

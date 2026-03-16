import numpy as np
import random
from .config import TrainConfig, Action


class QAgent:
    def __init__(self, num_states: int, num_actions: int, cfg: TrainConfig):
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.epsilon_decay = cfg.epsilon_decay
        self.min_epsilon = cfg.epsilon_min

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return Action(random.randint(0, self.num_actions - 1))
        return Action(int(np.argmax(self.q_table[state])))

    def learn(self, state, action, reward, next_state, terminated):
        if isinstance(action, Action):
            action = action.value
        old_value = self.q_table[state, action]
        next_max = 0.0 if terminated else np.max(self.q_table[next_state])
        self.q_table[state, action] = old_value + self.alpha * (
            reward + self.gamma * next_max - old_value
        )

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save(self, filename="q_table.npy"):
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}")

    def load(self, filename="q_table.npy"):
        self.q_table = np.load(filename)
        self.epsilon = self.min_epsilon
        print(f"Q-table loaded from {filename}")

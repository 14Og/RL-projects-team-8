import numpy as np
import random

class QAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.num_states = num_states
        self.num_actions = num_actions
        
        # 1. Инициализируем Q-таблицу нулями
        # Размер: 20580 строк на 7 столбцов
        self.q_table = np.zeros((num_states, num_actions))
        
        # 2. Гиперпараметры
        self.alpha = alpha      # Скорость обучения
        self.gamma = gamma      # Важность будущих наград
        self.epsilon = epsilon  # Шанс случайного хода (исследование)
        self.epsilon_decay = 0.99995 # Как быстро уменьшается любопытство
        self.min_epsilon = 0.01

    def choose_action(self, state):
        """Выбирает действие: либо случайно, либо по таблице."""
        # Стратегия Epsilon-Greedy
        if random.uniform(0, 1) < self.epsilon:
            # Исследуем: выбираем любое случайное действие
            return random.randint(0, self.num_actions - 1)
        else:
            # Эксплуатируем: выбираем действие с максимальным Q-значением для этого состояния
            # Если в строке несколько одинаковых максимумов (например, все нули), 
            # argmax выберет первый попавшийся.
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """Обновляет знания агента на основе полученного опыта."""
        
        # 1. Что мы думаем сейчас о текущем действии
        old_value = self.q_table[state, action]
        
        # 2. Какую максимальную награду мы можем получить из следующего состояния
        # Мы смотрим в строку следующего состояния и ищем там самый большой Q
        next_max = np.max(self.q_table[next_state])
        
        # 3. Формула Q-обучения (Уравнение Беллмана)
        # Мы обновляем старое значение, добавляя к нему ошибку (разницу между тем,
        # что получили реально + будущее, и тем, что думали раньше).
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        
        # Записываем обновленное значение обратно в таблицу
        self.q_table[state, action] = new_value

    def decay_epsilon(self):
        """Постепенно уменьшает шанс случайного хода."""
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filename="q_table.npy"):
        """Сохраняет обученную таблицу в файл."""
        np.save(filename, self.q_table)
        print(f"Table saved to {filename}")

    def load_q_table(self, filename="q_table.npy"):
        """Загружает таблицу из файла."""
        self.q_table = np.load(filename)
        self.epsilon = self.min_epsilon # После загрузки мы обычно хотим играть, а не учиться
        print(f"Table loaded from {filename}")
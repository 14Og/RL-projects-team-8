import matplotlib.pyplot as plt
import numpy as np
from env import TaxiEnv
from agent import QAgent

def plot_res(rewards, window=100):
    """Функция для отрисовки графика наград."""
    plt.figure(figsize=(10, 5))
    
    # Считаем скользящее среднее, чтобы сгладить график
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(rewards, alpha=0.3, label="Награда за эпизод", color='blue')
    plt.plot(moving_avg, label=f"Среднее за {window} игр", color='red', linewidth=2)
    
    plt.title("Прогресс обучения Такси (Q-Learning)")
    plt.xlabel("Эпизод")
    plt.ylabel("Суммарная награда")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Настройки ---
env = TaxiEnv()
num_states = 7 * 7 * 5 * 4 * (env.max_fuel + 1)
num_actions = 7

agent = QAgent(num_states, num_actions, alpha=0.1, gamma=0.95, epsilon=1.0)

num_episodes = 200000
all_rewards = [] # Для графика
all_success = [] 

print("Начинаем обучение...")

for episode in range(1, num_episodes + 1):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        
        state = next_state
        episode_reward += reward
        if reward == 60:
            all_success.append(1)
        elif done: # Если игра закончилась, но награда не 60 (смерть или таймаут)
            all_success.append(0)
    
    agent.decay_epsilon()
    all_rewards.append(episode_reward)

    # Вывод статуса каждые 1000 игр
    if episode % 10000 == 0 or episode == 1:
        recent_success_rate = np.mean(all_success[-2000:]) * 100
        avg_r = np.mean(all_rewards[-1000:])
        print(f"Эпизод: {episode} | Средняя награда: {avg_r:.1f} | Success: {recent_success_rate:.1f}% | Eps: {agent.epsilon:.3f}")
        agent.save_q_table(f"q_table_step_{episode}.npy")

# Сохраняем результат
agent.save_q_table("q_table_final.npy")

# Выводим график
print("Обучение закончено! Рисую график...")
plot_res(all_rewards)
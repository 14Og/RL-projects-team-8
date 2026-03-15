import os
from env import TaxiEnv

def print_env(env):
    """Простая отрисовка в консоли для теста."""
    grid = [[" " for _ in range(5)] for _ in range(5)]
    
    # Расставляем точки
    for i, (r, c) in enumerate(env.locs):
        grid[r][c] = ["R", "G", "Y", "B"][i]
    
    # Где такси
    tr, tc = env.taxi_pos
    grid[tr][tc] = "X" if env.pass_idx < 4 else "@" # @ - такси с пассажиром
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Target: {['R', 'G', 'Y', 'B'][env.dest_idx]}")
    print(f"Passenger: {'IN TAXI' if env.pass_idx == 4 else ['R', 'G', 'Y', 'B'][env.pass_idx]}")
    print("-" * 11)
    for row in grid:
        print("|" + "|".join(row) + "|")
    print("-" * 11)
    print("Use: w/a/s/d for move, p for Pickup, o for Dropoff, q to Quit")

env = TaxiEnv()
done = False

while not done:
    print_env(env)
    cmd = input("Action: ").lower()
    
    action = -1
    if cmd == 's': action = 0
    elif cmd == 'w': action = 1
    elif cmd == 'd': action = 2
    elif cmd == 'a': action = 3
    elif cmd == 'p': action = 4
    elif cmd == 'o': action = 5
    elif cmd == 'q': break
    
    if action != -1:
        state, reward, done = env.step(action)
        print(f"Reward: {reward}, State ID: {state}")
        if done:
            print("SUCCESS! Passenger delivered.")
            break
import numpy as np

class TaxiEnv:
    def __init__(self):
        self.rows = 7
        self.cols = 7
        self.locs = [(0, 0), (0, 6), (6, 0), (6, 6)]
        self.gas_station = (3, 3) # Заправка в центре
        self.max_fuel = 16
        self.count_steps = 0
        self.walls = [
            ((0, 1), 2), ((1, 1), 2), ((2, 1), 2),
            ((4, 0), 2), ((5, 0), 2), ((6, 0), 2),
            ((3, 3), 2), ((4, 3), 2), ((5, 3), 2),
            ((0, 4), 2), ((1, 4), 2), ((2, 4), 2),
        ]
        self.reset()

    def reset(self):
        self.count_steps = 0
        self.taxi_pos = [np.random.randint(0, 7), np.random.randint(0, 7)]
        self.pass_idx = np.random.randint(0, 4)
        self.dest_idx = np.random.randint(0, 4)
        while self.dest_idx == self.pass_idx:
            self.dest_idx = np.random.randint(0, 4)
        
        self.fuel = self.max_fuel # Полный бак при старте
        return self.get_state()

    def get_state(self):
        # Новая формула кодирования:
        # (((row*7 + col)*5 + pass_idx)*4 + dest_idx)*21 + fuel
        state = self.taxi_pos[0]
        state = state * 7 + self.taxi_pos[1]
        state = state * 5 + self.pass_idx
        state = state * 4 + self.dest_idx
        state = state * (self.max_fuel + 1) + self.fuel
        return state

    def step(self, action):
            row, col = self.taxi_pos
            reward = -1
            done = False
            
            # 1. Тратим бензин (кроме действия заправки)
            if action != 6:
                self.fuel -= 1

            # 2. Логика движения (0-3)
            if action in [0, 1, 2, 3]:
                blocked = False
                if (tuple(self.taxi_pos), action) in self.walls: blocked = True
                # Проверка "обратной" стороны стен
                if action == 2 and ((row, col+1), 3) in self.walls: blocked = True
                if action == 3 and ((row, col-1), 2) in self.walls: blocked = True
                if action == 0 and ((row+1, col), 1) in self.walls: blocked = True
                if action == 1 and ((row-1, col), 0) in self.walls: blocked = True

                if not blocked:
                    if action == 0 and row < 6: self.taxi_pos[0] += 1
                    elif action == 1 and row > 0: self.taxi_pos[0] -= 1
                    elif action == 2 and col < 6: self.taxi_pos[1] += 1
                    elif action == 3 and col > 0: self.taxi_pos[1] -= 1
                else:
                    reward = -3

            # 3. Pickup (4)
            elif action == 4:
                if self.pass_idx < 4 and self.taxi_pos == list(self.locs[self.pass_idx]):
                    self.pass_idx = 4
                    reward = 15
                    #print("Passenger picked up!")
                else: reward = -10
                
            # 4. Dropoff (5)
            elif action == 5:
                if self.pass_idx == 4 and self.taxi_pos == list(self.locs[self.dest_idx]):
                    self.pass_idx = self.dest_idx
                    reward = 60
                    done = True
                    #print("Success delivery!")
                else: reward = -10

            # 5. ЗАПРАВКА (6)
            elif action == 6:
                if tuple(self.taxi_pos) == self.gas_station:
                    self.fuel = self.max_fuel
                    reward = -1
                else:
                    reward = -10

            # 6. Проверка смерти
            if self.fuel < 0:
                reward = -50
                done = True
                self.fuel = 0

            self.count_steps += 1
            if self.count_steps >= 50:
                reward = -50
                done = True
                self.fuel = 0

            return self.get_state(), reward, done

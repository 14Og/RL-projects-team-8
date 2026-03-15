import pygame
import sys
import numpy as np
from env import TaxiEnv

CELL_SIZE = 80
WIDTH, HEIGHT = 7 * CELL_SIZE, 7 * CELL_SIZE + 80

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GRAY = (150, 150, 150)

class TaxiPygame:

    def __init__(self, env, q_table):
        pygame.init()

        self.env = env
        self.q_table = q_table

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Taxi Q-Learning Agent")

        self.font = pygame.font.SysFont("Arial", 20)
        self.clock = pygame.time.Clock()

        self.state = self.env.reset()

    def get_action(self, state):
        return np.argmax(self.q_table[state])

    def draw(self):
        self.screen.fill(WHITE)

        # grid
        for i in range(8):
            pygame.draw.line(self.screen,(230,230,230),(0,i*CELL_SIZE),(7*CELL_SIZE,i*CELL_SIZE))
            pygame.draw.line(self.screen,(230,230,230),(i*CELL_SIZE,0),(i*CELL_SIZE,7*CELL_SIZE))

        # gas station
        gr, gc = self.env.gas_station
        gas_rect = (gc*CELL_SIZE+10, gr*CELL_SIZE+10, CELL_SIZE-20, CELL_SIZE-20)
        pygame.draw.rect(self.screen,(100,100,100),gas_rect)

        txt = self.font.render("GAS",True,WHITE)
        self.screen.blit(txt,(gc*CELL_SIZE+25,gr*CELL_SIZE+25))

        # walls
        for (pos, action) in self.env.walls:
            r,c = pos
            if action == 2:
                pygame.draw.line(self.screen,BLACK,
                ((c+1)*CELL_SIZE,r*CELL_SIZE),
                ((c+1)*CELL_SIZE,(r+1)*CELL_SIZE),6)

        # locations
        colors = [(200,0,0),(0,200,0),(255,165,0),(0,180,180)]
        for i,pos in enumerate(self.env.locs):
            pygame.draw.rect(self.screen,colors[i],
            (pos[1]*CELL_SIZE+5,pos[0]*CELL_SIZE+5,CELL_SIZE-10,CELL_SIZE-10),3)

        # passenger
        if self.env.pass_idx < 4:
            p_pos = self.env.locs[self.env.pass_idx]
            pygame.draw.circle(self.screen,BLUE,
            (p_pos[1]*CELL_SIZE+40,p_pos[0]*CELL_SIZE+40),12)

        # taxi
        tr,tc = self.env.taxi_pos
        pygame.draw.rect(self.screen,YELLOW,
        (tc*CELL_SIZE+15,tr*CELL_SIZE+15,CELL_SIZE-30,CELL_SIZE-30))

        if self.env.pass_idx == 4:
            pygame.draw.circle(self.screen,BLUE,
            (tc*CELL_SIZE+40,tr*CELL_SIZE+40),8)

        # fuel bar
        fuel_ratio = self.env.fuel / self.env.max_fuel
        fuel_color = (0,200,0) if fuel_ratio > 0.3 else (200,0,0)

        pygame.draw.rect(self.screen,GRAY,(10,7*CELL_SIZE+40,200,20))
        pygame.draw.rect(self.screen,fuel_color,(10,7*CELL_SIZE+40,200*fuel_ratio,20))

        fuel_txt = self.font.render(f"Fuel: {self.env.fuel}/{self.env.max_fuel}",True,BLACK)
        self.screen.blit(fuel_txt,(10,7*CELL_SIZE+15))

        target_name = ['R','G','Y','B'][self.env.dest_idx]
        info_txt = self.font.render(f"Deliver to: {target_name}",True,BLACK)

        self.screen.blit(info_txt,(230,7*CELL_SIZE+40))

        pygame.display.flip()

    def run(self):

        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # действие агента
            action = self.get_action(self.state)

            new_state, reward, done = self.env.step(action)
            self.state = new_state

            self.draw()

            if done:
                print("Episode finished | reward:", reward)
                pygame.time.delay(1000)
                self.state = self.env.reset()

            self.clock.tick(5)   # скорость агента


if __name__ == "__main__":

    env = TaxiEnv()

    # загрузка Q таблицы
    q_table = np.load("q_table_step_120000.npy")

    ui = TaxiPygame(env, q_table)
    ui.run()
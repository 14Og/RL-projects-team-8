import pygame
import sys

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)      # Такси
RED = (255, 0, 0)         # Точка R
GREEN = (0, 255, 0)       # Точка G
CYAN = (0, 255, 255)      # Точка B (Blue)
ORANGE = (255, 165, 0)    # Точка Y (Yellow)
BLUE = (0, 0, 255)        # Пассажир (если не в такси)

class TaxiRenderer:
    def __init__(self, cell_size=100):
        pygame.init()
        self.cell_size = cell_size
        self.width = 5 * cell_size
        self.height = 5 * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Taxi Q-Learning (Custom Env)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24, bold=True)

    def draw(self, env):
        # 1. Очистка экрана
        self.screen.fill(WHITE)

        # 2. Отрисовка сетки
        for i in range(6):
            pygame.draw.line(self.screen, GRAY, (0, i*self.cell_size), (self.width, i*self.cell_size))
            pygame.draw.line(self.screen, GRAY, (i*self.cell_size, 0), (i*self.cell_size, self.height))

        # 3. Отрисовка стен (как в оригинальном Taxi-v3)
        # Стены вертикальные: (строка, колонка_после_которой_стена)
        walls = [(0,1), (1,1), (3,0), (4,0), (3,2), (4,2)]
        for (r, c) in walls:
            start_pos = ((c+1) * self.cell_size, r * self.cell_size)
            end_pos = ((c+1) * self.cell_size, (r+1) * self.cell_size)
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 6)

        # 4. Отрисовка целевых точек (R, G, Y, B)
        # locs = [(0,0), (0,4), (4,0), (4,3)]
        points = [
            (env.locs[0], RED, "R"),
            (env.locs[1], GREEN, "G"),
            (env.locs[2], ORANGE, "Y"),
            (env.locs[3], CYAN, "B")
        ]
        for pos, color, label in points:
            rect = (pos[1]*self.cell_size + 5, pos[0]*self.cell_size + 5, self.cell_size-10, self.cell_size-10)
            pygame.draw.rect(self.screen, color, rect, 2) # Только рамка
            text = self.font.render(label, True, color)
            self.screen.blit(text, (pos[1]*self.cell_size + 40, pos[0]*self.cell_size + 35))

        # 5. Отрисовка пассажира
        if env.pass_idx < 4: # Если он не в такси
            p_pos = env.locs[env.pass_idx]
            center = (p_pos[1]*self.cell_size + 50, p_pos[0]*self.cell_size + 50)
            pygame.draw.circle(self.screen, BLUE, center, 15)

        # 6. Отрисовка такси
        t_row, t_col = env.taxi_pos
        taxi_rect = (t_col*self.cell_size + 20, t_row*self.cell_size + 20, self.cell_size - 40, self.cell_size - 40)
        pygame.draw.rect(self.screen, YELLOW, taxi_rect)
        pygame.draw.rect(self.screen, BLACK, taxi_rect, 2) # Обводка

        # Если пассажир в такси, рисуем его внутри такси
        if env.pass_idx == 4:
            center = (t_col*self.cell_size + 50, t_row*self.cell_size + 50)
            pygame.draw.circle(self.screen, BLUE, center, 10)

        pygame.display.flip()
        
    def handle_events(self):
        """Обработка закрытия окна, чтобы Pygame не завис."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def wait(self, fps=10):
        self.clock.tick(fps)
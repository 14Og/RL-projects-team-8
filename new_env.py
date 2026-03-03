
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# --- КЛАССЫ (3 сегмента) ---

class State:
    def __init__(self, angles: np.ndarray):
        assert angles.shape == (3,)
        self._angles = angles.copy()
        self._joints = self._calculate_joint_positions(angles)

    @property
    def angles(self) -> np.ndarray: return self._angles
    @property
    def joints(self) -> np.ndarray: return self._joints

    @staticmethod
    def _calculate_joint_positions(angles: np.ndarray) -> np.ndarray:
        seg = np.zeros((4, 2))
        a1, a2, a3 = np.deg2rad(angles)
        L1, L2, L3 = 1.0, 1.0, 1.0
        
        def se2(x, y, t):
            return np.array([[np.cos(t), -np.sin(t), x], [np.sin(t), np.cos(t), y], [0, 0, 1]])

        T1 = se2(0, 0, a1)
        T2 = se2(L1, 0, a2)
        T3 = se2(L2, 0, a3)
        
        seg[1, :] = (T1 @ np.array([L1, 0, 1]))[:2]
        seg[2, :] = (T1 @ T2 @ np.array([L2, 0, 1]))[:2]
        seg[3, :] = (T1 @ T2 @ T3 @ np.array([L3, 0, 1]))[:2]
        return seg

class ManipulatorEnv:
    def __init__(self, initial_state: State, target: np.ndarray, obstacles: np.ndarray = None):
        self._obstacles = obstacles if obstacles is not None else np.empty((0, 3))
        self._state = initial_state
        self._target = np.array(target)
        self._ray_len = 0.7
        self._collision_threshold = 0.05

    def check_collision(self, state: State) -> bool:
        joints = state.joints
        for obs in self._obstacles:
            obs_p, r = obs[:2], obs[2] + self._collision_threshold
            for i in range(3):
                p1, p2 = joints[i], joints[i+1]
                vec = p2 - p1
                t = np.clip(np.dot(obs_p - p1, vec) / (np.dot(vec, vec) + 1e-6), 0, 1)
                if np.linalg.norm(obs_p - (p1 + t * vec)) <= r: return True
        return False

    def get_point_lidar(self, pos: np.ndarray, n_rays: int = 16):
        readings = np.full(n_rays, 1.0)
        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        for i, ang in enumerate(angles):
            d_vec = np.array([np.cos(ang), np.sin(ang)])
            min_d = self._ray_len
            for obs in self._obstacles:
                oc = pos - obs[:2]
                b = 2.0 * np.dot(d_vec, oc)
                c = np.dot(oc, oc) - obs[2]**2
                disc = b**2 - 4*c
                if disc >= 0:
                    t = (-b - np.sqrt(disc)) / 2.0
                    if 0 <= t <= min_d: min_d = t
            readings[i] = min_d / self._ray_len
        return readings

# --- НАСТРОЙКА СЦЕНЫ ---

obs_list = np.array([
    [1.2, 1.0, 0.3], 
    [-0.8, 1.2, 0.3],
    [0.5, -1.2, 0.4]
])
target_pos = np.array([1.5, 2.0])
env = ManipulatorEnv(State(np.array([0.0, 0.0, 0.0])), target=target_pos, obstacles=obs_list)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect('equal'); ax.grid(True)

# Объекты
link_line, = ax.plot([], [], 'b-', linewidth=5, alpha=0.8, zorder=5)
joints_pt, = ax.plot([], [], 'ko', markersize=6, zorder=6)
target_pt, = ax.plot([target_pos[0]], [target_pos[1]], 'gx', markersize=12, markeredgewidth=3)

# Пул линий лидара: 3 сустава * 16 лучей = 24 линии
lidar_lines = [ax.plot([], [], '--', alpha=0.4, linewidth=1)[0] for _ in range(48)]

for obs in obs_list:
    ax.add_patch(plt.Circle((obs[0], obs[1]), obs[2], color='black', alpha=0.3))

def update(frame):
    # Плавное "змеиное" движение
    a1 = frame # Базовое вращение
    a2 = np.sin(np.deg2rad(frame * 3)) * 70 # Колебание локтя 1
    a3 = np.cos(np.deg2rad(frame * 2)) * 50 # Колебание локтя 2
    
    new_state = State(np.array([a1, a2, a3]))
    env._state = new_state
    
    # Цвет при столкновении
    collision = env.check_collision(new_state)
    link_line.set_color('red' if collision else 'blue')
    
    # Манипулятор
    pts = new_state.joints
    link_line.set_data(pts[:, 0], pts[:, 1])
    joints_pt.set_data(pts[:, 0], pts[:, 1])
    
    # Лидары (J1, J2, J3)
    line_idx = 0
    for j_idx in range(1, 4):
        pos = pts[j_idx]
        vals = env.get_point_lidar(pos)
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        for d_norm, ang in zip(vals, angles):
            d = d_norm * env._ray_len
            end = pos + np.array([np.cos(ang), np.sin(ang)]) * d
            lidar_lines[line_idx].set_data([pos[0], end[0]], [pos[1], end[1]])
            lidar_lines[line_idx].set_color('red' if d_norm < 1.0 else 'green')
            line_idx += 1
            
    return [link_line, joints_pt] + lidar_lines

# Анимация
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 3), interval=40, blit=True)

plt.close()
HTML(ani.to_jshtml())

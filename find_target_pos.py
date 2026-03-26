import numpy as np
import math
import matplotlib.pyplot as plt

def get_all_ik_solutions(x, y, L1, L2, L3, num_samples=100):
    """
    Возвращает список всех возможных конфигураций [th1, th2, th3], 
    чтобы конец робота оказался в точке (x, y).
    """
    valid_configurations = []
    
    # Перебираем все возможные углы ориентации последнего звена (от -Pi до Pi)
    phi_angles = np.linspace(-np.pi, np.pi, num_samples)
    
    for phi in phi_angles:
        # 1. Находим координаты "запястья" (Joint 3)
        x_w = x - L3 * np.cos(phi)
        y_w = y - L3 * np.sin(phi)
        
        # Расстояние от базы до запястья
        D2 = x_w**2 + y_w**2
        D = np.sqrt(D2)
        
        # 2. Проверяем, может ли 2-звенный механизм дотянуться до запястья
        if D > (L1 + L2) or D < abs(L1 - L2):
            continue # Слишком далеко или слишком близко для текущего phi
            
        # 3. Считаем угол th2 по теореме косинусов
        cos_th2 = (D2 - L1**2 - L2**2) / (2 * L1 * L2)
        # Ограничиваем от ошибок округления
        cos_th2 = np.clip(cos_th2, -1.0, 1.0) 
        
        # У нас всегда два решения для th2: "локоть вниз" и "локоть вверх"
        th2_sol1 = np.arccos(cos_th2)
        th2_sol2 = -np.arccos(cos_th2)
        
        # 4. Считаем th1 и th3 для обоих решений
        for th2 in [th2_sol1, th2_sol2]:
            k1 = L1 + L2 * np.cos(th2)
            k2 = L2 * np.sin(th2)
            
            th1 = np.arctan2(y_w, x_w) - np.arctan2(k2, k1)
            th3 = phi - th1 - th2
            
            # Приводим углы к диапазону [-pi, pi]
            th1 = (th1 + np.pi) % (2 * np.pi) - np.pi
            th2 = (th2 + np.pi) % (2 * np.pi) - np.pi
            th3 = (th3 + np.pi) % (2 * np.pi) - np.pi
            
            valid_configurations.append(np.array([th1, th2, th3]))
            
    return valid_configurations

# ==========================================
# ДЕМОНСТРАЦИЯ
# ==========================================
L1, L2, L3 = 1.0, 0.7, 0.4
target_x, target_y = -1.5, 0.8

# Получаем все возможные позы
configs = get_all_ik_solutions(target_x, target_y, L1, L2, L3, num_samples=50)

print(f"Найдено {len(configs)} уникальных конфигураций для точки ({target_x}, {target_y})")

# Рисуем первые 10 решений, чтобы не засорять график
plt.figure(figsize=(8, 8))
plt.plot(target_x, target_y, 'r*', markersize=15, label='Target')

for i, q in enumerate(configs[::len(configs)//10]): # Берем 10 равномерно распределенных решений
    x0, y0 = 0, 0
    x1, y1 = L1*np.cos(q[0]), L1*np.sin(q[0])
    x2, y2 = x1 + L2*np.cos(q[0]+q[1]), y1 + L2*np.sin(q[0]+q[1])
    x3, y3 = x2 + L3*np.cos(q[0]+q[1]+q[2]), y2 + L3*np.sin(q[0]+q[1]+q[2])
    
    plt.plot([x0, x1, x2, x3], [y0, y1, y2, y3], 'o-', alpha=0.5, lw=2)

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.grid(True)
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title("Разные конфигурации для достижения одной точки")
plt.legend()
plt.show()
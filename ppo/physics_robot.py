import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from numba import njit

class Robot_Dynamic_3DOF:
    def __init__(self, masses, lengthes):
        self.m = masses
        self.l = lengthes
        self.lc = lengthes / 2.0
        self.I = (1/12) * masses * lengthes**2
        self.g = 9.81
    
    def get_matrices(self, q):
        return get_matrices_fast(q, self.m, self.l, self.lc, self.I, self.g)
    
    def dynamics(self, q, dq, tau):
        return dynamics_fast(q, dq, tau, self.m, self.l, self.lc, self.I, self.g)
    
    def update_rk4(self, q, dq, tau, dt):
        return update_rk4_fast(q, dq, tau, dt, self.m, self.l, self.lc, self.I, self.g)
    

@njit(cache=True)
def get_matrices_fast(q, masses, lengthes, lc, I, g):
    m1, m2, m3 = masses
    l1, l2, l3 = lengthes
    lc1, lc2, lc3 = lc
    I1, I2, I3 = I
    g = g
    th1, th2, th3 = q

    c2 = np.cos(th2)
    c3 = np.cos(th3)
    c23 = np.cos(th2 + th3)

    # Inertia matrix (M)
    M = np.zeros((3, 3))

    # Auxiliary terms for inertia matrix
    m3l1lc3 = m3 * l1 * lc3
    m3l2lc3 = m3 * l2 * lc3
    m2l1lc2 = m2 * l1 * lc2
    m3l1l2 = m3 * l1 * l2

    # Symmetric inertia matrix elements
    M[2,2] = I3 + m3 * lc3**2
    M[1,2] = M[2,2] + m3l2lc3 * c3
    M[0,2] = M[2,2] + m3l2lc3 * c3 + m3l1lc3 * c23

    M[2,1] = M[1,2]
    M[1,1] = I2 + m2 * lc2**2 + I3 + m3 * (l2**2 + lc3**2 + 2*l2*lc3*c3)
    M[0,1] = M[1,1] + (m2l1lc2 + m3l1l2) * c2 + m3l1lc3 * c23

    M[2,0] = M[0,2]
    M[1,0] = M[0,1]
    M[0,0] = I1 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2) + \
                m3*(l1**2 + l2**2 + lc3**2 + 2*l1*l2*c2 + 2*l2*lc3*c3 + 2*l1*lc3*c23)

    # Vector of gravity (G)
    G = np.zeros(3)
    G[2] = m3 * g * lc3 * np.cos(th1 + th2 + th3)
    G[1] = (m2*lc2 + m3*l2) * g * np.cos(th1 + th2) + G[2]
    G[0] = (m1*lc1 + (m2 + m3)*l1) * g * np.cos(th1) + G[1]

    return M, G

@njit(cache=True)
def dynamics_fast(q, dq, tau, masses, lengthes, lc, I, g):
    M, G = get_matrices_fast(q, masses, lengthes, lc, I, g)
    M += np.eye(3) * 1e-4 
    damping = 0.3 * dq
    rhs = tau - G - damping
    q_dd = np.linalg.solve(M, rhs)
    return dq, q_dd 

@njit(cache=True)
def update_rk4_fast(q, dq, tau, dt, masses, lengthes, lc, I, g):
    # k1 - first step
    v1, a1 = dynamics_fast(q, dq, tau, masses, lengthes, lc, I, g)
    # k2 - trying a step in the middle
    v2, a2 = dynamics_fast(q + v1 * dt/2, dq + a1 * dt/2, tau, masses, lengthes, lc, I, g)
    
    # k3 - one more trying a step in the middle
    v3, a3 = dynamics_fast(q + v2 * dt/2, dq + a2 * dt/2, tau, masses, lengthes, lc, I, g)
    
    # k4 - trying a full step
    v4, a4 = dynamics_fast(q + v3 * dt, dq + a3 * dt, tau, masses, lengthes, lc, I, g)
    
    # Weighing the increments to get the new state
    q_new = q + (dt/6.0) * (v1 + 2*v2 + 2*v3 + v4)
    dq_new = dq + (dt/6.0) * (a1 + 2*a2 + 2*a3 + a4)
    
    # Numerical stability: clipping the velocities to prevent them from becoming too large
    dq_new = np.clip(dq_new, -15, 15)
    
    return q_new, dq_new
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from numba import njit
from .config import ModelConfig

class Robot_Dynamic_3DOF:
    def __init__(self, masses, lengthes, cfg=ModelConfig):
        self.m = masses
        self.l = lengthes
        self.cfg = cfg
        self.lc = lengthes / 2.0
        self.I = (1/12) * masses * lengthes**2
        self.g = 9.81
    
    def get_matrices(self, q):
        return get_matrices_fast(q, self.m, self.l, self.lc, self.I, self.g)
    
    def dynamics(self, q, dq, tau):
        return dynamics_fast(q, dq, tau, self.m, self.l, self.lc, self.I, self.g, self.cfg.damping)
    
    def update_rk4(self, q, dq, tau, dt):
        return update_rk4_fast(q, dq, tau, dt, self.m, self.l, self.lc, self.I, self.g, self.cfg.damping)
    

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
def coriolis_vector(q, dq, masses, lengthes, lc):
    _, m2, m3 = masses
    l1, l2, _ = lengthes
    _, lc2, lc3 = lc
    th2, th3 = q[1], q[2]
    dq0, dq1, dq2 = dq[0], dq[1], dq[2]

    # Christoffel-symbol coefficients derived from M(q)
    f1 = (m2 * l1 * lc2 + m3 * l1 * l2) * np.sin(th2)
    f2 = m3 * l1 * lc3 * np.sin(th2 + th3)
    f3 = m3 * l2 * lc3 * np.sin(th3)

    Cdq = np.zeros(3)
    Cdq[0] = (-2*(f1+f2)*dq0*dq1 - 2*(f3+f2)*dq0*dq2
              - (f1+f2)*dq1*dq1 - 2*(f3+f2)*dq1*dq2
              - (f3+f2)*dq2*dq2)
    Cdq[1] = ((f1+f2)*dq0*dq0
              - 2*f3*dq0*dq2 - 2*f3*dq1*dq2
              - f3*dq2*dq2)
    Cdq[2] = ((f3+f2)*dq0*dq0
              + 2*f3*dq0*dq1
              + f3*dq1*dq1)
    return Cdq


@njit(cache=True)
def dynamics_fast(q, dq, tau, masses, lengthes, lc, I, g, damping):
    M, G = get_matrices_fast(q, masses, lengthes, lc, I, g)
    M += np.eye(3) * 1e-4
    Cdq = coriolis_vector(q, dq, masses, lengthes, lc)
    damping = damping * dq
    rhs = tau - G - Cdq - damping
    q_dd = np.linalg.solve(M, rhs)
    return dq, q_dd

@njit(cache=True)
def update_rk4_fast(q, dq, tau, dt, masses, lengthes, lc, I, g, damping):
    # k1 - first step
    v1, a1 = dynamics_fast(q, dq, tau, masses, lengthes, lc, I, g, damping)
    # k2 - trying a step in the middle
    v2, a2 = dynamics_fast(q + v1 * dt/2, dq + a1 * dt/2, tau, masses, lengthes, lc, I, g, damping)
    
    # k3 - one more trying a step in the middle
    v3, a3 = dynamics_fast(q + v2 * dt/2, dq + a2 * dt/2, tau, masses, lengthes, lc, I, g, damping)
    
    # k4 - trying a full step
    v4, a4 = dynamics_fast(q + v3 * dt, dq + a3 * dt, tau, masses, lengthes, lc, I, g, damping)
    
    # Weighing the increments to get the new state
    q_new = q + (dt/6.0) * (v1 + 2*v2 + 2*v3 + v4)
    dq_new = dq + (dt/6.0) * (a1 + 2*a2 + 2*a3 + a4)
    
    # Numerical stability: clipping the velocities to prevent them from becoming too large
    dq_new = np.clip(dq_new, -15, 15)
    
    return q_new, dq_new
"""
Direct torque control simulation — tune tau and see how the arm moves.
No PD controller. Torques are applied directly with gravity compensation.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ppo.physics_robot import Robot_Dynamic_3DOF

# ===========================================================================
# TUNABLE PARAMETERS
# ===========================================================================
TAU          = np.array([100.0, 0.0, 0.0])   # Nm, applied torques (residual, gravity already compensated)
GRAVITY_COMP = True                          # add G(q) so tau=0 holds the arm still

dt_phys = 0.0001
n_substeps = 100

INITIAL_ANGLES = np.array([0.0, 0.0, 0.0])  # rad

MASSES       = np.array([1.0, 0.7, 0.4])
LINK_LENGTHS = np.array([100.0, 70.0, 40.0]) / 100.0  # px → metres, same as robot.py
BASE_XY      = np.array([0.0, 0.0])
DAMPING      = 0.05

# ===========================================================================
# PHYSICS
# ===========================================================================
physics = Robot_Dynamic_3DOF(masses=MASSES, lengthes=LINK_LENGTHS)


def wrap(theta):
    return (np.asarray(theta) + math.pi) % (2 * math.pi) - math.pi


def forward_kinematics(q, base=BASE_XY, links=LINK_LENGTHS):
    points = [base.copy()]
    p = base.copy()
    cumulative = 0.0
    for i, L in enumerate(links):
        cumulative += q[i]
        p = p + np.array([L * math.cos(cumulative), L * math.sin(cumulative)])
        points.append(p.copy())
    return np.array(points)


# ===========================================================================
# PRE-SIMULATE
# ===========================================================================
q  = INITIAL_ANGLES.copy().astype(np.float64)
dq = np.array([10, 20, 30])

frames_q        = [q.copy()]
frames_vel_norm = [0.0]
frames_tau_norm = [float(np.linalg.norm(TAU))]

for _ in range(n_substeps):
    tau = TAU.copy()
    if GRAVITY_COMP:
        tau = tau 

    q, dq = physics.update_rk4(q, dq, tau, dt_phys)
    q = wrap(q)

    frames_q.append(q.copy())
    frames_vel_norm.append(float(np.linalg.norm(dq)))
    frames_tau_norm.append(float(np.linalg.norm(TAU)))

frames_q = np.array(frames_q)
n_frames  = len(frames_q)
time_axis = np.arange(n_frames) * dt_phys

# ===========================================================================
# VISUALISATION
# ===========================================================================
fig = plt.figure(figsize=(13, 7))
fig.suptitle(
    f"Direct torque | tau={TAU} Nm | grav_comp={GRAVITY_COMP} | dt={dt_phys}s | damping={DAMPING}",
    fontsize=12, fontweight="bold"
)

ax_arm = fig.add_subplot(1, 2, 1)
ax_vel = fig.add_subplot(2, 2, 2)
ax_q   = fig.add_subplot(2, 2, 4)

reach  = float(np.sum(LINK_LENGTHS))
margin = 0.3
ax_arm.set_xlim(-reach - margin, reach + margin)
ax_arm.set_ylim(-reach - margin, reach + margin)
ax_arm.set_aspect("equal")
ax_arm.set_title("Robot arm")
ax_arm.grid(True, alpha=0.3)
ax_arm.axhline(0, color="k", lw=0.5)
ax_arm.axvline(0, color="k", lw=0.5)

link_colors = ["#1E88E5", "#43A047", "#FB8C00"]
arm_lines, arm_joints = [], []
for j in range(3):
    line, = ax_arm.plot([], [], lw=5 - j, color=link_colors[j], solid_capstyle="round", zorder=3)
    dot,  = ax_arm.plot([], [], "o", color=link_colors[j], markersize=8, zorder=4)
    arm_lines.append(line)
    arm_joints.append(dot)
ee_dot, = ax_arm.plot([], [], "o", color="red", markersize=10, zorder=5, label="EE")
ax_arm.legend(fontsize=9)

step_text = ax_arm.text(0.02, 0.97, "", transform=ax_arm.transAxes,
                         fontsize=10, va="top", family="monospace")

ax_vel.set_xlim(0, time_axis[-1])
ax_vel.set_ylim(0, max(frames_vel_norm) * 1.1 + 1e-6)
ax_vel.set_ylabel("||dq|| (rad/s)")
ax_vel.set_title("Velocity norm")
ax_vel.grid(True, alpha=0.3)
vel_line, = ax_vel.plot([], [], lw=1.5, color="#8E24AA")
vel_dot,  = ax_vel.plot([], [], "o", color="#8E24AA", markersize=5)

joint_colors = ["#1E88E5", "#43A047", "#FB8C00"]
q_lines, q_dots = [], []
ax_q.set_xlim(0, time_axis[-1])
ax_q.set_ylim(-math.pi - 0.2, math.pi + 0.2)
ax_q.set_ylabel("angle (rad)")
ax_q.set_xlabel("time (s)")
ax_q.set_title("Joint angles")
ax_q.axhline(0, color="k", lw=0.5, ls="--")
ax_q.grid(True, alpha=0.3)
for j in range(3):
    ln, = ax_q.plot([], [], lw=1.5, color=joint_colors[j], label=f"q{j}")
    dt, = ax_q.plot([], [], "o", color=joint_colors[j], markersize=5)
    q_lines.append(ln)
    q_dots.append(dt)
ax_q.legend(fontsize=9)

plt.tight_layout()


def update(frame):
    q_now  = frames_q[frame]
    joints = forward_kinematics(q_now)

    for j in range(3):
        seg = joints[j:j+2]
        arm_lines[j].set_data(seg[:, 0], seg[:, 1])
        arm_joints[j].set_data([joints[j, 0]], [joints[j, 1]])
    ee_dot.set_data([joints[-1, 0]], [joints[-1, 1]])

    t   = time_axis[:frame + 1]
    vel = frames_vel_norm[:frame + 1]
    vel_line.set_data(t, vel)
    vel_dot.set_data([t[-1]], [vel[-1]])

    for j in range(3):
        angles = frames_q[:frame + 1, j]
        q_lines[j].set_data(t, angles)
        q_dots[j].set_data([t[-1]], [angles[-1]])

    step_text.set_text(
        f"step {frame}/{n_frames-1}\n"
        f"|dq|={frames_vel_norm[frame]:.2f} rad/s\n"
        f"q={q_now.round(2)}"
    )
    return arm_lines + arm_joints + [ee_dot, vel_line, vel_dot] + q_lines + q_dots + [step_text]


anim = FuncAnimation(fig, update, frames=n_frames, interval=0, blit=True, repeat=True)
plt.show()

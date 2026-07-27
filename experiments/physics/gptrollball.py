# Simulate a ball rolling on a sine-shaped track with simple physics
# using matplotlib animation.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Physical constants
g = 9.81  # gravity (m/s^2)
mu = 0.05  # friction coefficient (simplified)
ball_radius = 0.1  # radius of ball in meters

# Track parameters
x = np.linspace(0, 10, 500)
y = 0.5 * np.sin(2 * np.pi * x / 5)  # sine wave shape

# Ball state
ball_x = 2.0  # initial x position
ball_v = 0.0  # initial velocity

# Time step
dt = 0.01


# Track slope calculation for physics
def slope(x_pos):
    dx = x[1] - x[0]
    i = int((x_pos - x[0]) / dx)
    if i < 0 or i >= len(x) - 1:
        return 0
    dy = y[i + 1] - y[i]
    return dy / dx


# Update function for animation
def update(frame):
    global ball_x, ball_v

    # Calculate slope and acceleration
    m = slope(ball_x)
    theta = np.arctan(m)
    a = g * np.sin(theta) - mu * g * np.cos(theta)

    # Update velocity and position
    ball_v += a * dt
    ball_x += ball_v * dt

    # Wrap around if ball rolls past edges
    if ball_x < x[0]:
        ball_x = x[-1]
    elif ball_x > x[-1]:
        ball_x = x[0]

    # Update ball position on plot
    i = int((ball_x - x[0]) / (x[1] - x[0]))
    ball.set_data(ball_x, y[i] + ball_radius)
    return (ball,)


# Plot setup
fig, ax = plt.subplots()
ax.plot(x, y)
(ball,) = ax.plot([], [], marker="o", markersize=10)

ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Ball Rolling on a Sine Track (with Physics)")

# Animation
ani = FuncAnimation(fig, update, frames=1000, interval=10, blit=True)
plt.show()

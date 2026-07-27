# Simulate falling raindrops with realistic physics (gravity + quadratic air drag)
# and display an animation using matplotlib.
# - Physical parameters are realistic (drop radius in mm, water density, air density, drag coefficient)
# - Uses semi-implicit Euler integration for stability
# - Shows splashes as short-lived upward droplets when a drop hits the ground
# - No explicit colors/styles are set (matplotlib defaults used)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Physical constants
g = 9.81  # m/s^2, gravity
rho_air = 1.225  # kg/m^3, air density at sea level
rho_water = 1000.0  # kg/m^3, water density
C_d = 0.47  # drag coefficient for a sphere (approx)

# Simulation parameters
N = 120  # number of active raindrops concurrently shown
drop_rate = 300  # drops created per second (poisson rate)
dt = 0.01  # simulation time step (s)
sim_time = 10.0  # seconds to simulate in animation (used for total frames)
frames = int(sim_time / dt)

# Visual area (meters)
width_m = 2.0  # horizontal width in meters
height_m = 3.0  # vertical height in meters (top to ground)


# Convert physical radius (m) to matplotlib scatter size units roughly
def radius_to_markerarea(r_m):
    # matplotlib 's' is area in points^2; use a heuristic scaling
    return (r_m * 1000.0) ** 2  # mm^2 -> visual area (heuristic)


# Raindrop class containing physical properties
class Raindrop:
    def __init__(self):
        # sample radius from realistic distribution: 0.3 mm to 2.5 mm (typical raindrop)
        r_mm = 0.3 + np.random.rand() ** 2 * (2.5 - 0.3)  # bias toward small drops
        self.r = r_mm / 1000.0  # convert to meters
        self.area = np.pi * self.r**2
        self.volume = 4.0 / 3.0 * np.pi * self.r**3
        self.mass = rho_water * self.volume
        # initial position (x uniform across width, y slightly above top)
        self.x = np.random.rand() * width_m
        self.y = height_m + np.random.rand() * 0.3  # spawn slightly above top
        # initial velocity (small downward perturbation)
        self.vx = 0.0
        self.vy = -0.5 - np.random.rand() * 0.5  # downward
        # compute approximate terminal velocity for reference
        self.vt = np.sqrt((2 * self.mass * g) / (rho_air * self.area * C_d))
        # life flag
        self.alive = True

    def step(self, dt, wind=0.0):
        # quadratic drag: F_d = 0.5 * rho_air * C_d * A * v * |v|
        # apply separately for x and y
        # vx update
        v_rel_x = self.vx - wind
        Fd_x = 0.5 * rho_air * C_d * self.area * v_rel_x * abs(v_rel_x)
        ax = -Fd_x / self.mass
        # vy update (including gravity)
        v_rel_y = self.vy
        Fd_y = 0.5 * rho_air * C_d * self.area * v_rel_y * abs(v_rel_y)
        ay = -g - Fd_y / self.mass

        # semi-implicit Euler: update velocities with acceleration, then positions
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        # ground collision: simple splash model if hitting ground (y <= 0)
        if self.y <= 0:
            self.alive = False
            return True  # signal that splash should be created
        return False


# Simple splash particle for small upward droplets (not physically detailed)
class SplashParticle:
    def __init__(self, x, y, energy, count=6):
        # create a small cluster of upward-moving micro-drops
        angles = (
            np.linspace(-np.pi / 3, -2 * np.pi / 3, count)
            + (np.random.rand(count) - 0.5) * 0.2
        )
        speeds = np.random.rand(count) * energy * 2.0
        self.xs = np.array([x] * count)
        self.ys = np.array([y] * count)
        self.vxs = speeds * np.cos(angles)
        self.vys = speeds * np.sin(angles)
        self.rs = np.full(count, 0.2 / 1000.0)  # 0.2 mm micro-drops
        self.life = np.full(
            count, 0.25 + np.random.rand(count) * 0.25
        )  # seconds remaining

    def step(self, dt):
        # micro-particles feel gravity and drag (but tiny mass)
        alive_mask = self.life > 0
        if not np.any(alive_mask):
            return False
        for i in range(len(self.life)):
            if self.life[i] <= 0:
                continue
            # drag approximate (linearized because tiny particles and short life)
            v = np.sqrt(self.vxs[i] ** 2 + self.vys[i] ** 2) + 1e-12
            Fd = 0.5 * rho_air * C_d * (np.pi * self.rs[i] ** 2) * v * v
            ax = (
                -Fd * self.vxs[i] / (v * (rho_water * 4 / 3 * np.pi * self.rs[i] ** 3))
                if v > 0
                else 0
            )
            ay = (
                -g
                - Fd * self.vys[i] / (v * (rho_water * 4 / 3 * np.pi * self.rs[i] ** 3))
                if v > 0
                else -g
            )
            self.vxs[i] += ax * dt
            self.vys[i] += ay * dt
            self.xs[i] += self.vxs[i] * dt
            self.ys[i] += self.vys[i] * dt
            self.life[i] -= dt
        return np.any(self.life > 0)


# Simulation containers
drops = []
splashes = []

# initialize with some drops
for _ in range(int(N * 0.6)):
    drops.append(Raindrop())


# wind profile (m/s) - gentle horizontal wind
def wind_field(y):
    # simple vertical shear: stronger wind higher up
    return 0.5 + 0.5 * (y / height_m)


# Prepare matplotlib figure
fig, ax = plt.subplots(figsize=(4, 6))
ax.set_xlim(0, width_m)
ax.set_ylim(0, height_m)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Raindrops — gravity + quadratic air drag")

# initial scatter plots for drops and splash particles
scat = ax.scatter([], [], s=[])
splash_scat = ax.scatter([], [], s=[])


def init():
    scat.set_offsets(np.empty((0, 2)))
    scat.set_sizes([])
    splash_scat.set_offsets(np.empty((0, 2)))
    splash_scat.set_sizes([])
    return scat, splash_scat


# animation update function
frame_count = 0


def update(frame):
    global frame_count
    frame_count += 1
    # probabilistic spawn of new drops (Poisson-ish)
    expected_new = drop_rate * dt
    num_new = np.random.poisson(expected_new)
    for _ in range(num_new):
        if len(drops) < N:
            drops.append(Raindrop())

    # step existing drops
    drop_positions = []
    drop_sizes = []
    to_remove = []
    for i, d in enumerate(drops):
        created_splash = d.step(dt, wind=wind_field(d.y))
        if created_splash:
            # create a splash with energy roughly proportional to vertical speed
            energy = min(3.0, abs(d.vy))
            splashes.append(SplashParticle(d.x, 0.0, energy))
            to_remove.append(i)
        else:
            # keep within horizontal bounds (wrap-around for visual effect)
            if d.x < 0:
                d.x += width_m
            elif d.x > width_m:
                d.x -= width_m
            drop_positions.append([d.x, max(d.y, 0.0)])
            drop_sizes.append(radius_to_markerarea(d.r))

    # remove dead drops by index (reverse to avoid reindexing problems)
    for idx in reversed(to_remove):
        del drops[idx]

    # step splashes and collect their positions
    splash_positions = []
    splash_sizes = []
    to_keep_splashes = []
    for sp in splashes:
        alive = sp.step(dt)
        if alive:
            # include all alive micro-particles
            for x, y, r in zip(sp.xs, sp.ys, sp.rs):
                if y >= 0 and y <= height_m and r > 0:
                    splash_positions.append(
                        [x if 0 <= x <= width_m else (x % width_m), y]
                    )
                    splash_sizes.append(radius_to_markerarea(r) * 0.7)
            to_keep_splashes.append(sp)
    splashes[:] = to_keep_splashes

    # update scatter plots
    if drop_positions:
        scat.set_offsets(np.array(drop_positions))
        scat.set_sizes(np.array(drop_sizes))
    else:
        scat.set_offsets(np.empty((0, 2)))
        scat.set_sizes([])

    if splash_positions:
        splash_scat.set_offsets(np.array(splash_positions))
        splash_scat.set_sizes(np.array(splash_sizes))
    else:
        splash_scat.set_offsets(np.empty((0, 2)))
        splash_scat.set_sizes([])

    return scat, splash_scat


ani = FuncAnimation(
    fig, update, frames=frames, init_func=init, interval=dt * 1000, blit=True
)
plt.show()

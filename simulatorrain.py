import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 物理参数
g = 9.81  # 重力加速度 (m/s²)
rho_air = 1.2  # 空气密度 (kg/m³)
C_d = 0.47  # 雨滴阻力系数（球形）
r = 0.001  # 雨滴半径 (m)，约1mm
rho_water = 1000  # 水的密度 (kg/m³)

# 计算雨滴质量与终端速度
mass = (4 / 3) * np.pi * r**3 * rho_water
A = np.pi * r**2  # 截面积
v_terminal = np.sqrt((2 * mass * g) / (rho_air * C_d * A))  # 终端速度

print(f"雨滴终端速度: {v_terminal:.2f} m/s")

# 模拟区域
height = 20  # 下落高度 (m)
width = 10  # 水平范围 (m)

# 初始化雨滴（位置和速度）
n_drops = 50
x = np.random.uniform(0, width, n_drops)
y = np.random.uniform(height, height + 5, n_drops)  # 从上方随机位置开始
v_y = np.zeros(n_drops)  # 初始垂直速度为0

# 时间步长
dt = 0.02  # 秒

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("雨滴下落实时模拟（含空气阻力）")
scat = ax.scatter(x, y, s=10, c="blue", alpha=0.7)


def update(frame):
    global x, y, v_y

    # 更新每个雨滴的速度（考虑重力和空气阻力）
    # 阻力 F_drag = 0.5 * rho_air * C_d * A * v^2，方向与速度相反
    drag_force = 0.5 * rho_air * C_d * A * v_y**2
    acceleration = g - drag_force / mass

    v_y += acceleration * dt
    y -= v_y * dt  # y轴向下减小（地面在 y=0）

    # 重置落到地面的雨滴（重新从顶部生成）
    reset_mask = y <= 0
    y[reset_mask] = np.random.uniform(height, height + 2)
    v_y[reset_mask] = 0
    x[reset_mask] = np.random.uniform(0, width)

    # 更新散点图数据
    scat.set_offsets(np.c_[x, y])
    return (scat,)


# 创建动画
ani = FuncAnimation(fig, update, frames=500, interval=30, blit=True, repeat=True)

plt.tight_layout()
plt.show()

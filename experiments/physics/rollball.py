import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========== 物理参数 ==========
g = 9.81  # 重力加速度 (m/s²)
A = 2.0  # 正弦振幅 (m)
k = 1.0  # 波数，控制波长：λ = 2π/k
mass = 1.0  # 小球质量（实际不影响加速度，因 m 被约掉）
dt = 0.01  # 时间步长 (s)


# ========== 跑道函数及其导数 ==========
def y_track(x):
    """跑道高度 y = A * sin(kx)"""
    return A * np.sin(k * x)


def dy_dx(x):
    """跑道一阶导数 dy/dx = A*k*cos(kx)"""
    return A * k * np.cos(k * x)


def d2y_dx2(x):
    """跑道二阶导数 d²y/dx² = -A*k²*sin(kx)"""
    return -A * k**2 * np.sin(k * x)


# ========== 初始状态 ==========
x = 0.0  # 初始 x 位置
v = 0.5  # 初始沿轨道速度（标量，>0 表示向右）
total_energy = None  # 用于验证能量守恒（可选）


# ========== 辅助函数：计算曲率半径与切向加速度 ==========
def curvature_radius(x):
    """计算轨道在 x 处的曲率半径 R"""
    y1 = dy_dx(x)
    y2 = d2y_dx2(x)
    if abs(y2) < 1e-8:
        return np.inf  # 近似直线
    R = (1 + y1**2) ** 1.5 / abs(y2)
    return R


def tangent_angle(x):
    """轨道切线与水平夹角 θ（弧度）"""
    return np.arctan(dy_dx(x))


def tangential_acceleration(x):
    """重力在切线方向的分量：a_t = g * sin(θ) 向下坡方向"""
    theta = tangent_angle(x)
    # 若小球向右运动，下坡时 θ<0 → sin(θ)<0 → 加速；上坡时 θ>0 → 减速
    # 但加速度方向应始终沿重力投影：a_t = -g * sin(theta) （以 x 增加为正方向）
    return -g * np.sin(theta)


# ========== 数值积分（使用速度-Verlet 简化版）==========
# 我们用 x 作为广义坐标，v 是沿轨道的速率（标量），方向由运动方向隐含

# 存储轨迹用于绘图
trail_x = []
trail_y = []
max_trail = 200

# ========== 绘图设置 ==========
fig, ax = plt.subplots(figsize=(10, 5))
x_track = np.linspace(-2, 10, 500)
y_track_vals = y_track(x_track)
ax.plot(x_track, y_track_vals, "k-", lw=2, label="跑道 $y = A \\sin(kx)$")
(ball_plot,) = ax.plot([], [], "ro", markersize=10, label="小球")
(trail_plot,) = ax.plot([], [], "r-", alpha=0.5, lw=1)

ax.set_xlim(-1, 9)
ax.set_ylim(-A * 1.2, A * 1.2)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("小球在正弦跑道上滚动（含物理引擎）")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)


# ========== 动画更新函数 ==========
def update(frame):
    global x, v, trail_x, trail_y

    # 计算当前切向加速度
    a_t = tangential_acceleration(x)

    # 更新速度和位置（沿轨道弧长 s，但用 x 近似——适用于小斜率）
    # 更精确做法：用弧长参数化，但此处用 dx ≈ ds / sqrt(1 + (dy/dx)^2)
    slope_factor = np.sqrt(1 + dy_dx(x) ** 2)  # ds/dx
    ds = v * dt
    dx = ds / slope_factor
    x += dx

    # 更新速度（a_t 是沿轨道的加速度）
    v += a_t * dt

    # 防止小球跑出可视区域（可选循环或反弹）
    if x > 9:
        x = -1
        v = max(0.1, abs(v))  # 保留速度大小

    # 记录轨迹
    y_ball = y_track(x)
    trail_x.append(x)
    trail_y.append(y_ball)
    if len(trail_x) > max_trail:
        trail_x.pop(0)
        trail_y.pop(0)

    # 更新绘图
    ball_plot.set_data([x], [y_ball])
    trail_plot.set_data(trail_x, trail_y)

    # 可选：打印能量（动能 + 势能）
    # KE = 0.5 * mass * v**2
    # PE = mass * g * y_ball
    # print(f"总能量: {KE + PE:.3f} J")

    return ball_plot, trail_plot


# ========== 启动动画 ==========
ani = FuncAnimation(
    fig,
    update,
    frames=2000,
    interval=20,  # ~50 FPS
    blit=True,
    repeat=True,
)

plt.tight_layout()
plt.show()

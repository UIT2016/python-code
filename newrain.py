import pygame
import sys
import math

# 初始化 Pygame
pygame.init()

# 屏幕设置
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("小球滚动动画 - 物理引擎")

# 颜色
BACKGROUND = (30, 30, 40)
ROAD_COLOR = (60, 60, 70)
BALL_COLOR = (220, 100, 100)
SLOPE_COLOR = (50, 80, 50)

# 物理常量（单位：像素、秒）
G = 980  # 重力加速度（像素/秒²），约等于 9.8 m/s² 的 100 倍缩放
FRICTION_COEFF = 0.3  # 滚动摩擦系数（无量纲）
AIR_RESISTANCE = 0.998  # 空气阻力（每帧乘以此值，可设为1.0关闭）

# 路面参数
SLOPE_ANGLE = 15  # 斜坡角度（度）
SLOPE_LENGTH = 300  # 斜坡长度（像素）
FLAT_START_X = 300  # 平路起始 x 坐标


class RollingBall:
    def __init__(self, x, y, radius=20):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = 0  # 水平速度
        self.vy = 0  # 垂直速度（实际由路径约束）
        self.on_slope = True  # 初始在斜坡上

    def update(self, dt):
        if self.on_slope:
            # 斜坡上的物理计算
            theta = math.radians(SLOPE_ANGLE)
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)

            # 重力沿斜面分量（驱动）
            a_gravity = G * sin_t

            # 摩擦力（阻碍）
            a_friction = FRICTION_COEFF * G * cos_t

            # 合加速度（沿斜面方向）
            a_net = a_gravity - a_friction

            # 更新沿斜面的速度
            v_along = math.sqrt(self.vx**2 + self.vy**2)
            if v_along > 0:
                # 保持方向一致
                dir_x = self.vx / v_along
                dir_y = self.vy / v_along
            else:
                dir_x = math.cos(theta)
                dir_y = math.sin(theta)
                v_along = 0

            # 应用加速度
            v_along += a_net * dt

            # 应用空气阻力
            v_along *= AIR_RESISTANCE

            # 防止反向加速（摩擦不能让球倒滚）
            if a_net < 0 and v_along < 0:
                v_along = 0

            self.vx = v_along * dir_x
            self.vy = v_along * dir_y

            # 更新位置
            self.x += self.vx * dt
            self.y += self.vy * dt

            # 检查是否到达斜坡底部
            if self.x >= FLAT_START_X:
                self.on_slope = False
                # 对齐到平路起点
                self.x = FLAT_START_X
                self.y = HEIGHT - 100 - self.radius  # 平路 y 坐标
                # 保留水平速度，垂直速度归零
                speed = abs(self.vx)  # 因为此时 vx > 0
                self.vx = speed
                self.vy = 0

        else:
            # 平路上：只有摩擦减速
            if self.vx > 0:
                # 摩擦加速度（负值）
                a_friction_flat = -FRICTION_COEFF * G
                self.vx += a_friction_flat * dt

                # 防止反向运动
                if self.vx < 0:
                    self.vx = 0

                # 空气阻力
                self.vx *= AIR_RESISTANCE

                # 更新位置
                self.x += self.vx * dt
                self.y = HEIGHT - 100 - self.radius  # 保持在地面

    def draw(self, surface):
        pygame.draw.circle(surface, BALL_COLOR, (int(self.x), int(self.y)), self.radius)
        # 添加高光
        highlight_pos = (
            int(self.x - self.radius * 0.3),
            int(self.y - self.radius * 0.3),
        )
        pygame.draw.circle(
            surface, (255, 220, 220), highlight_pos, int(self.radius * 0.3)
        )


def draw_road():
    # 绘制斜坡
    slope_end_x = FLAT_START_X
    slope_end_y = HEIGHT - 100
    slope_start_y = slope_end_y - SLOPE_LENGTH * math.sin(math.radians(SLOPE_ANGLE))
    slope_start_x = FLAT_START_X - SLOPE_LENGTH * math.cos(math.radians(SLOPE_ANGLE))

    pygame.draw.polygon(
        screen,
        SLOPE_COLOR,
        [
            (slope_start_x, slope_start_y),
            (slope_end_x, slope_end_y),
            (slope_end_x - 20, slope_end_y + 20),
            (slope_start_x - 20, slope_start_y + 20),
        ],
    )

    # 绘制平路
    pygame.draw.rect(screen, ROAD_COLOR, (FLAT_START_X - 20, HEIGHT - 80, WIDTH, 100))


# 创建小球（放在斜坡顶端）
ball_x = FLAT_START_X - SLOPE_LENGTH * math.cos(math.radians(SLOPE_ANGLE))
ball_y = HEIGHT - 100 - SLOPE_LENGTH * math.sin(math.radians(SLOPE_ANGLE)) - 20
ball = RollingBall(ball_x, ball_y)

clock = pygame.time.Clock()
FPS = 60
running = True

while running:
    dt = clock.tick(FPS) / 1000.0  # 秒

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # 按 R 重置
                ball = RollingBall(ball_x, ball_y)
            elif event.key == pygame.K_ESCAPE:
                running = False

    # 更新物理
    ball.update(dt)

    # 绘制
    screen.fill(BACKGROUND)
    draw_road()
    ball.draw(screen)

    # 显示信息
    font = pygame.font.SysFont(None, 24)
    info = f"速度: {ball.vx:.1f} px/s | 位置: ({ball.x:.0f}, {ball.y:.0f}) | {'斜坡' if ball.on_slope else '平路'}"
    text = font.render(info, True, (200, 220, 255))
    screen.blit(text, (10, 10))

    hint = font.render("按 R 重置小球 | ESC 退出", True, (180, 180, 200))
    screen.blit(hint, (10, 40))

    pygame.display.flip()

pygame.quit()
sys.exit()

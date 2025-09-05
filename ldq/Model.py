# model.py
# 核心模型文件，包含了项目所有的物理规律和判定逻辑。

import numpy as np

# --- 1. 定义全局物理常量 ---
GRAVITY = np.array([0, 0, -9.8])
CLOUD_SINK_SPEED = np.array([0, 0, -3.0])
SMOKE_RADIUS = 10.0
SMOKE_DURATION = 20.0

# --- 2. 定义场景实体位置 (可以被所有求解器共享) ---
MISSILE_START_POS = np.array([20000, 0, 2000])
MISSILE_SPEED = 300.0
UAV_START_POS = np.array([17800, 0, 1800])
FAKE_TARGET_POS = np.array([0, 0, 0])
REAL_TARGET_CENTER = np.array([0, 200, 5])

# --- 3. 时空运动模型 ---

def get_missile_pos(t):
    """计算任意时刻 t 的导弹位置"""
    direction = FAKE_TARGET_POS - MISSILE_START_POS
    unit_direction = direction / np.linalg.norm(direction)
    velocity_vector = MISSILE_SPEED * unit_direction
    return MISSILE_START_POS + velocity_vector * t

def get_uav_pos(t, uav_start_pos, uav_flight_velocity):
    """计算任意时刻 t 的无人机位置 (通用版)"""
    return uav_start_pos + uav_flight_velocity * t

def get_detonation_pos(uav_start_pos, uav_flight_velocity, drop_time, detonation_delay):
    """计算烟幕弹爆炸瞬间的位置 (通用版)"""
    # 阶段一：无人机飞行到投放点
    drop_pos = get_uav_pos(drop_time, uav_start_pos, uav_flight_velocity)
    
    # 阶段二：烟幕弹做抛体运动
    initial_grenade_velocity = uav_flight_velocity
    flight_time = detonation_delay
    
    detonation_pos = drop_pos + initial_grenade_velocity * flight_time + 0.5 * GRAVITY * flight_time**2
    return detonation_pos

def get_cloud_center_pos(t, detonation_pos, drop_time, detonation_delay):
    """计算爆炸后任意时刻 t 的烟幕云心位置 (通用版)"""
    detonation_time = drop_time + detonation_delay
    time_after_detonation = t - detonation_time
    return detonation_pos + CLOUD_SINK_SPEED * time_after_detonation

# --- 4. 遮蔽判定模型 ---

def check_veiling(missile_pos, cloud_pos):
    """
    判定原则: 导弹与真目标的视线线段，是否与烟幕云球体相交。
    """
    p1 = missile_pos
    p2 = REAL_TARGET_CENTER
    sphere_center = cloud_pos
    sphere_radius = SMOKE_RADIUS

    v = p2 - p1
    a = np.dot(v, v)
    if a == 0: return False

    b = 2 * np.dot(v, p1 - sphere_center)
    c = np.dot(p1 - sphere_center, p1 - sphere_center) - sphere_radius**2
    
    delta = b**2 - 4*a*c
    if delta < 0:
        return False

    t1 = (-b - np.sqrt(delta)) / (2*a)
    t2 = (-b + np.sqrt(delta)) / (2*a)
    
    if (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1):
        return True
        
    return False
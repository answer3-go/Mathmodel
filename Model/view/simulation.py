# simulation.py (V2.0 最终版)
import numpy as np

# --- 1. 定义全局物理常量 ---
G_ACCEL = np.array([0, 0, -9.8])  # 重力加速度 (m/s^2)
CLOUD_VELOCITY = np.array([0, 0, -3.0]) # 烟幕云下沉速度 (m/s)
SMOKE_LIFESPAN = 20.0 # 烟幕有效持续时间 (秒)

# --- 2. 独立的运动轨迹计算函数 ---
def calculate_entity_path(p_start, velocity, time_vector):
    """计算任意物体的匀速直线运动轨迹"""
    return p_start + velocity * time_vector[:, np.newaxis]

def calculate_grenade_path(p_drop, v_drop, drop_time, detonate_time, dt=0.1):
    """计算烟幕弹的抛体运动轨迹"""
    flight_duration = detonate_time - drop_time
    if flight_duration <= 0:
        return np.array([p_drop]), np.array([drop_time])
    time_grenade_relative = np.arange(0, flight_duration + dt, dt)
    path = p_drop + v_drop * time_grenade_relative[:, np.newaxis] + 0.5 * G_ACCEL * (time_grenade_relative**2)[:, np.newaxis]
    time_grenade_absolute = time_grenade_relative + drop_time
    return path, time_grenade_absolute

def calculate_cloud_path(p_detonate, detonate_time, end_time, dt=0.1):
    """计算烟幕云云心的匀速下沉轨迹"""
    drift_duration = end_time - detonate_time
    if drift_duration <= 0:
        return np.array([p_detonate]), np.array([detonate_time])
    time_cloud_relative = np.arange(0, drift_duration + dt, dt)
    path = p_detonate + CLOUD_VELOCITY * time_cloud_relative[:, np.newaxis]
    time_cloud_absolute = time_cloud_relative + detonate_time
    return path, time_cloud_absolute
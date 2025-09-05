# solver_q1.py
# 问题一的求解器文件。
# 职责：配置策略参数，调用核心模型进行计算，并格式化输出结果。

import sys
import os
import numpy as np

# 添加父目录到路径以便导入Model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Model as model 

def solve():
    """
    应用模型求解问题一的核心函数。
    """
    # --- 1. 配置问题一的特定策略 ---
    uav_flight_velocity = np.array([-120.0, 0, 0])
    drop_time = 1.5
    detonation_delay = 3.6

    # === 增强功能：详细打印所有用到的物理量参数配置 ===
    print("\n[1. 物理参数配置审查]")
    print("--------------------------------------------------")
    print("  [+] 场景实体初始状态:")
    print(f"      - 导弹M1初始位置   : {model.MISSILE_START_POS} m")
    print(f"      - 导弹M1飞行速度   : {model.MISSILE_SPEED} m/s")
    print(f"      - 无人机FY1初始位置: {model.UAV_START_POS} m")
    print(f"      - 假目标位置       : {model.FAKE_TARGET_POS} m")
    print(f"      - 真目标关键点     : {model.REAL_TARGET_CENTER} m")
    
    print("\n  [+] 物理常量:")
    print(f"      - 重力加速度       : {model.GRAVITY} m/s²")
    print(f"      - 烟幕云下沉速度   : {model.CLOUD_SINK_SPEED} m/s")
    print(f"      - 烟幕云有效半径   : {model.SMOKE_RADIUS} m")
    print(f"      - 烟幕云持续时间   : {model.SMOKE_DURATION} s")
    
    print("\n  [+] 问题一特定策略参数:")
    print(f"      - 无人机FY1飞行速度: {uav_flight_velocity} m/s")
    print(f"      - 烟幕弹投放时间   : {drop_time} s")
    print(f"      - 引信延迟时间     : {detonation_delay} s")
    print("--------------------------------------------------")
    
    # --- 2. 调用模型进行计算 ---
    
    # 步骤一: 计算出固定的起爆点位置
    detonation_pos = model.get_detonation_pos(
        model.UAV_START_POS, 
        uav_flight_velocity, 
        drop_time, 
        detonation_delay
    )
    
    # 步骤二: 设置仿真参数
    dt = 0.01  # 时间步长
    t_end = 70.0 # 仿真结束时间
    time_steps = np.arange(0, t_end, dt)
    
    total_veiling_time = 0
    detonation_time = drop_time + detonation_delay
    expire_time = detonation_time + model.SMOKE_DURATION

    # === 增强功能：打印计算得出的关键中间变量 ===
    print("\n[2. 模型中间计算结果]")
    print("--------------------------------------------------")
    print(f"  - 计算得出的烟幕弹起爆位置: {np.round(detonation_pos, 4)} m")
    print(f"  - 计算得出的起爆时刻      : {detonation_time:.2f} s")
    print(f"  - 计算得出的烟幕失效时刻  : {expire_time:.2f} s")
    print("\n  - 仿真参数:")
    print(f"      - 时间步长 dt      : {dt} s")
    print(f"      - 仿真总时长       : {t_end} s")
    print(f"      - 总仿真步数       : {len(time_steps)} 步")
    print("--------------------------------------------------")
    
    print("\n[3. 正在进行时间积分计算...]")
    # 步骤三: 遍历时间，进行遮蔽判定和时长累加
    for t in time_steps:
        if detonation_time <= t < expire_time:
            missile_pos = model.get_missile_pos(t)
            cloud_pos = model.get_cloud_center_pos(t, detonation_pos, drop_time, detonation_delay)
            is_veiled = model.check_veiling(missile_pos, cloud_pos)
            
            if is_veiled:
                total_veiling_time += dt
    
    print("    计算完成！")
    return total_veiling_time, detonation_time

# --- 主程序入口 ---
if __name__ == "__main__":
    print("="*50)
    print("             数学建模问题A - 问题一求解器")
    print("="*50)
    
    # 调用求解函数
    veiling_time, det_time = solve()
    
    # --- 3. 美化最终答案输出 ---
    print("\n" + "="*20 + " 最终答案 " + "="*20)
    print(f"\n  在问题一给定的策略下，")
    print(f"  烟幕弹于 t = {det_time:.2f} 秒时起爆，")
    print(f"  对导弹 M1 的【有效遮蔽总时长】为: {veiling_time:.4f} 秒")
    print("\n" + "="*50)
    
    print("\n[模型说明]")
    print("  - 本结果基于以下核心模型计算得出：")
    print("    1. 时空运动模型 (匀速直线、抛体、匀速下沉)")
    print("    2. 遮蔽判定模型 (视线线段与烟幕云球体相交)")
    print("    3. 积分求解模型 (时间离散化数值积分)")
    print("\n求解完成！")
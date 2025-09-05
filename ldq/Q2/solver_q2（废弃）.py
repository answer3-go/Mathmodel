# solver_q2.py (V2.1 修正版)
import sys
import os
import numpy as np

# 添加父目录到路径以便导入Model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Model as model 
from scipy.optimize import differential_evolution
import time

def evaluate_strategy(x):
    """
    目标函数 (或称适应度函数)。
    接收一组策略参数 x，返回该策略的评价值（负的总遮蔽时长）。
    """
    try:
        v, alpha_deg, t_drop, detonation_delay = x
        alpha_rad = np.deg2rad(alpha_deg)
        uav_flight_velocity = np.array([v * np.cos(alpha_rad), v * np.sin(alpha_rad), 0])
        
        detonation_pos = model.get_detonation_pos(
            model.UAV_START_POS, uav_flight_velocity, t_drop, detonation_delay
        )
        
        # *** 关键修正: 统一评估精度 ***
        # 使用与问题一求解时相同的精确时间步长，确保不会错判
        dt = 0.01
        t_end = 70.0
        time_steps = np.arange(0, t_end, dt)
        
        total_veiling_time = 0
        detonation_time = t_drop + detonation_delay
        expire_time = detonation_time + model.SMOKE_DURATION

        for t in time_steps:
            if detonation_time <= t < expire_time:
                missile_pos = model.get_missile_pos(t)
                cloud_pos = model.get_cloud_center_pos(t, detonation_pos, t_drop, detonation_delay)
                is_veiled = model.check_veiling(missile_pos, cloud_pos)
                if is_veiled:
                    total_veiling_time += dt
                    
    except Exception as e:
        # 发生任何计算错误，都返回一个巨大的惩罚值
        return 1000.0 
    
    return -total_veiling_time

# --- 主程序入口 ---
if __name__ == "__main__":
    print("="*50)
    print("             数学建模问题A - 问题二求解器")
    print("="*50)

    # ... (测试代码部分可以注释掉，以节省时间) ...
    
    print("\n[开始使用差分进化算法进行全局寻优...]")
    print("  计算精度已设为 dt=0.01，过程会比较耗时，请耐心等待。")
    print("  优化器将利用您所有的CPU核心进行并行计算以加速。")

    bounds = [
        (70, 140),
        (0, 360),
        (0.1, 30),
        (0.1, 15)
    ]

    start_time = time.time()

    result = differential_evolution(
        func=evaluate_strategy,
        bounds=bounds,
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        updating='deferred',
        workers=1
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n寻优完成！")
    print(f"  - 总耗时: {elapsed_time:.2f} 秒")
    print(f"  - 优化是否成功: {result.success}")
    print(f"  - 总迭代次数: {result.nit}")
    print(f"  - 最优评价值 (负遮蔽时长): {result.fun:.4f}")

    print("\n" + "="*20 + " 问题二最优策略 " + "="*20)

    best_params = result.x
    best_v, best_alpha_deg, best_t_drop, best_detonation_delay = best_params
    best_veiling_time = -result.fun

    print(f"\n  [+] 最优决策变量:")
    print(f"      - 无人机飞行速度 (v)      : {best_v:.4f} m/s")
    print(f"      - 无人机飞行方向 (alpha)    : {best_alpha_deg:.4f} 度")
    print(f"      - 烟幕弹投放时间 (t_drop)   : {best_t_drop:.4f} s")
    print(f"      - 引信延迟 (det_delay)      : {best_detonation_delay:.4f} s")

    best_alpha_rad = np.deg2rad(best_alpha_deg)
    best_uav_velocity = np.array([best_v * np.cos(best_alpha_rad), best_v * np.sin(best_alpha_rad), 0])
    
    best_drop_pos = model.get_uav_pos(best_t_drop, model.UAV_START_POS, best_uav_velocity)
    best_detonation_pos = model.get_detonation_pos(model.UAV_START_POS, best_uav_velocity, best_t_drop, best_detonation_delay)

    print(f"\n  [+] 对应的物理状态:")
    print(f"      - 烟幕弹投放点坐标 : [{best_drop_pos[0]:.2f}, {best_drop_pos[1]:.2f}, {best_drop_pos[2]:.2f}] m")
    print(f"      - 烟幕弹起爆点坐标 : [{best_detonation_pos[0]:.2f}, {best_detonation_pos[1]:.2f}, {best_detonation_pos[2]:.2f}] m")
    
    print("\n  [+] 最终优化结果:")
    print(f"      - 在此最优策略下，对导弹M1的最大有效遮蔽时长为: {best_veiling_time:.4f} 秒")

    print("\n" + "="*56)
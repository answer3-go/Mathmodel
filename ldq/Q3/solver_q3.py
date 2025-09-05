# solver_q3.py
# 问题三的求解器文件：使用“套娃再优化”策略，将已知的6.14秒最优解作为种子进行精细化搜索。

import sys
import os
import numpy as np

# 添加父目录到路径以便导入Model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Model as model 
import time

# --- 模块三: 策略评价函数 (保持不变) ---
def evaluate_strategy_q3(x):
    """
    目标函数 (适应度函数) for Question 3.
    接收一组6维策略参数 x，返回该策略的总遮蔽时长。
    """
    try:
        v, alpha_deg, t_drop1, dt_drop2, dt_drop3, det_delay = x
        alpha_rad = np.deg2rad(alpha_deg)
        uav_flight_velocity = np.array([v * np.cos(alpha_rad), v * np.sin(alpha_rad), 0])

        drop_times = [t_drop1, t_drop1 + dt_drop2, t_drop1 + dt_drop2 + dt_drop3]
        detonation_times = [t + det_delay for t in drop_times]
        detonation_positions = [
            model.get_detonation_pos(model.UAV_START_POS, uav_flight_velocity, dt, det_delay)
            for dt in drop_times
        ]

        dt_sim = 0.01
        t_end = 70.0
        time_steps = np.arange(0, t_end, dt_sim)
        total_veiling_time = 0

        for t in time_steps:
            missile_pos = model.get_missile_pos(t)
            is_veiled_this_step = False
            for i in range(3):
                if detonation_times[i] <= t < detonation_times[i] + model.SMOKE_DURATION:
                    cloud_pos = model.get_cloud_center_pos(t, detonation_positions[i], drop_times[i], det_delay)
                    if model.check_veiling(missile_pos, cloud_pos):
                        is_veiled_this_step = True
                        break
            if is_veiled_this_step:
                total_veiling_time += dt_sim
    except Exception as e:
        return 1e-6 
    return total_veiling_time

# --- 模块四: 寻优求解引擎 (套娃再优化版) ---
def pso_solver_q3_finetune():
    """
    使用增强型PSO算法，并以已知的最优解为种子，进行精细化搜索。
    """
    # 1. PSO 算法参数微调：鼓励精细化搜索
    n_particles = 80      # 增加粒子数，提供更多样性
    n_dimensions = 6      
    max_iter = 300        # 增加迭代次数，进行更长时间的挖掘
    w = 0.4               # 降低惯性权重，减弱粒子“盲目”飞行的趋势
    c1 = 1.8              # 略微增加个体学习因子
    c2 = 1.8              # 略微增加社会学习因子

    bounds = np.array([
        [70, 140], [0, 360], [0.1, 20], [1.0, 10], [1.0, 10], [0.1, 15]
    ])
    
    # --- 关键改动: 注入6.14秒的最优解作为新种子 ---
    
    # 1. 定义第一次随机运行得到的最优解
    q3_best_known_solution = np.array([
        80.8599,      # v
        10.9027,      # alpha
        0.1000,       # t_drop1
        1.0000,       # dt_drop2
        10.0000,      # dt_drop3
        0.1060        # det_delay
    ])

    # 2. 初始化粒子群
    particles_pos = np.random.rand(n_particles, n_dimensions) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    
    # 3. 将第一个粒子强制替换为我们已知的最优种子
    print("\n[初始化阶段] 正在向初始种群注入已知的最优种子...")
    print(f"  - 种子策略 (来自上次最优解): {np.round(q3_best_known_solution, 4)}")
    particles_pos[0] = q3_best_known_solution
    
    # --- 后续流程保持不变 ---
    
    particles_vel = np.zeros((n_particles, n_dimensions))
    
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([evaluate_strategy_q3(p) for p in pbest_pos])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest_pos = pbest_pos[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]

    print(f"PSO初始化完成，初始种群最优得分: {gbest_fitness:.4f} s (由种子或更优的随机粒子贡献)")
    print("开始进行精细化迭代寻优...")
    
    for k in range(max_iter):
        for i in range(n_particles):
            r1 = np.random.rand(n_dimensions)
            r2 = np.random.rand(n_dimensions)
            
            cognitive_vel = c1 * r1 * (pbest_pos[i] - particles_pos[i])
            social_vel = c2 * r2 * (gbest_pos - particles_pos[i])
            particles_vel[i] = w * particles_vel[i] + cognitive_vel + social_vel
            
            particles_pos[i] = particles_pos[i] + particles_vel[i]
            
            particles_pos[i] = np.maximum(particles_pos[i], bounds[:, 0])
            particles_pos[i] = np.minimum(particles_pos[i], bounds[:, 1])
            
            current_fitness = evaluate_strategy_q3(particles_pos[i])
            
            if current_fitness > pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest_pos[i] = particles_pos[i]
                
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_pos = particles_pos[i]
        
        if (k + 1) % 10 == 0:
            print(f"迭代 {k+1}/{max_iter}, 当前最优遮蔽时长: {gbest_fitness:.4f} s")
        
    return gbest_pos, gbest_fitness

# --- 主程序入口 ---
if __name__ == "__main__":
    print("="*50)
    print("      数学建模问题A - 问题三求解器 (套娃再优化版)")
    print("="*50)

    start_time = time.time()
    
    best_params, best_veiling_time = pso_solver_q3_finetune()
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n寻优完成！")
    print(f"  - 总耗时: {elapsed_time:.2f} 秒")

    print("\n" + "="*20 + " 问题三最终最优策略 " + "="*20)

    v, alpha_deg, t_drop1, dt_drop2, dt_drop3, det_delay = best_params

    print(f"\n  [+] 最优决策变量 (6维):")
    print(f"      - 无人机飞行速度 (v)       : {v:.4f} m/s")
    print(f"      - 无人机飞行方向 (alpha)     : {alpha_deg:.4f} 度")
    print(f"      - 首次投放时间 (t_drop1)   : {t_drop1:.4f} s")
    print(f"      - 第二次投放间隔 (dt_drop2): {dt_drop2:.4f} s (>= 1.0s)")
    print(f"      - 第三次投放间隔 (dt_drop3): {dt_drop3:.4f} s (>= 1.0s)")
    print(f"      - 统一引信延迟 (det_delay)   : {det_delay:.4f} s")
    
    print("\n  [+] 最终优化结果:")
    print(f"      - 在此最优策略下，对导弹M1的最大有效遮蔽时长为: {best_veiling_time:.4f} 秒")
    
    alpha_rad = np.deg2rad(alpha_deg)
    uav_velocity = np.array([v * np.cos(alpha_rad), v * np.sin(alpha_rad), 0])
    
    drop_times = [t_drop1, t_drop1 + dt_drop2, t_drop1 + dt_drop2 + dt_drop3]
    print("\n  [+] 详细投放时序与坐标:")
    for i in range(3):
        drop_pos = model.get_uav_pos(drop_times[i], model.UAV_START_POS, uav_velocity)
        detonation_pos = model.get_detonation_pos(model.UAV_START_POS, uav_velocity, drop_times[i], det_delay)
        print(f"    --- 第 {i+1} 枚烟幕弹 ---")
        print(f"      - 投放时刻: {drop_times[i]:.2f} s")
        print(f"      - 投放点坐标: [{drop_pos[0]:.2f}, {drop_pos[1]:.2f}, {drop_pos[2]:.2f}] m")
        print(f"      - 起爆时刻: {drop_times[i] + det_delay:.2f} s")
        print(f"      - 起爆点坐标: [{detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f}] m")

    print("\n" + "="*56)
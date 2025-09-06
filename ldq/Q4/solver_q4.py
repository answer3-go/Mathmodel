# solver_q4_pso.py
# 问题四的求解器文件：使用真正的“量体裁衣”战术推演种子生成器。

import sys
import os
import numpy as np

# 添加父目录到路径以便导入Model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Model as model 
import time

# --- 模块三: 策略评价函数 (已修正) ---
def evaluate_strategy_q4(x):
    """
    目标函数 (适应度函数) for Question 4.
    接收一组10维策略参数 x，返回该策略的总遮蔽时长。
    """
    try:
        # 1. 解析10维决策变量
        v1, alpha1_deg, t_drop1 = x[0], x[1], x[2]
        v2, alpha2_deg, t_drop2 = x[3], x[4], x[5]
        v3, alpha3_deg, t_drop3 = x[6], x[7], x[8]
        det_delay = x[9]

        # 为每架无人机正确分配策略和初始位置
        uav_definitions = {
            'FY1': {'start_pos': model.UAV_START_POS,         'v': v1, 'alpha_deg': alpha1_deg, 't_drop': t_drop1},
            'FY2': {'start_pos': np.array([12000, 1400, 1400]), 'v': v2, 'alpha_deg': alpha2_deg, 't_drop': t_drop2},
            'FY3': {'start_pos': np.array([6000, -3000, 700]),  'v': v3, 'alpha_deg': alpha3_deg, 't_drop': t_drop3}
        }
        
        # 2. 为每架无人机计算其烟幕弹的时序和起爆点
        detonation_times = []
        detonation_positions = []

        for name, strat in uav_definitions.items():
            alpha_rad = np.deg2rad(strat['alpha_deg'])
            uav_velocity = np.array([strat['v'] * np.cos(alpha_rad), strat['v'] * np.sin(alpha_rad), 0])
            
            detonation_times.append(strat['t_drop'] + det_delay)
            detonation_positions.append(
                model.get_detonation_pos(strat['start_pos'], uav_velocity, strat['t_drop'], det_delay)
            )

        # 3. 设置仿真并计算总遮蔽时长
        dt_sim = 0.01
        t_end = 70.0
        time_steps = np.arange(0, t_end, dt_sim)
        total_veiling_time = 0

        for t in time_steps:
            missile_pos = model.get_missile_pos(t)
            is_veiled_this_step = False
            
            for i in range(len(uav_definitions)):
                if detonation_times[i] <= t < detonation_times[i] + model.SMOKE_DURATION:
                    original_drop_time = list(uav_definitions.values())[i]['t_drop']
                    cloud_pos = model.get_cloud_center_pos(t, detonation_positions[i], original_drop_time, det_delay)
                    
                    if model.check_veiling(missile_pos, cloud_pos):
                        is_veiled_this_step = True
                        break
            
            if is_veiled_this_step:
                total_veiling_time += dt_sim
                    
    except Exception as e:
        return 1e-6 
    
    return total_veiling_time

# --- V2.3 新增: 真正的“量体裁衣”战术推演种子生成器 ---
def generate_q4_seed_strategy_tailored():
    """
    通过为每架无人机独立评估最优方案，并进行组合，来生成高质量种子。
    """
    print("\n[初始化阶段] 启动“量体裁衣”战术推演，为问题四生成高质量种子...")
    
    # 1. 继承问题三的最优“作战节奏”和“飞行模式”作为基础
    q3_best_params = [80.8599, 10.9027, 0.1000, 1.0000, 10.0000, 0.1060]
    v_base, alpha_base, t1_base, dt2_base, dt3_base, delay_base = q3_best_params

    uav_defs = {
        'FY1': {'start_pos': model.UAV_START_POS},
        'FY2': {'start_pos': np.array([12000, 1400, 1400])},
        'FY3': {'start_pos': np.array([6000, -3000, 700])}
    }
    
    # 2. “量体裁衣”：计算出“标杆”任务点
    base_velocity = np.array([v_base * np.cos(np.deg2rad(alpha_base)), v_base * np.sin(np.deg2rad(alpha_base)), 0])
    base_drop_times = [t1_base, t1_base + dt2_base, t1_base + dt2_base + dt3_base]
    base_detonation_points = [model.get_detonation_pos(model.UAV_START_POS, base_velocity, dt, delay_base) for dt in base_drop_times]

    print("  - 标杆任务 (基于Q3解): 在三个关键点上形成烟幕。")

    seed_strategies = {}
    # 为每架无人机独立计算飞向第一个标杆点的最优路径
    for name, uav in uav_defs.items():
        target_point = base_detonation_points[0] # 以第一个关键点为主要汇合目标
        start_pos = uav['start_pos']
        
        direction_vec = target_point - start_pos
        
        # 确定飞行方向 (只考虑水平面)
        alpha_seed = np.rad2deg(np.arctan2(direction_vec[1], direction_vec[0])) % 360
        # 确定飞行速度 (全部使用最大速度以求最快到达)
        v_seed = 140.0
        
        # 估算到达目标点所需的飞行时间
        dist_to_target = np.linalg.norm(target_point - start_pos)
        fly_time_estimate = dist_to_target / v_seed
        
        # 投放时间 = 飞行时间 - 引信延迟 (确保至少为0.1s)
        t_drop_seed = max(0.1, fly_time_estimate - delay_base)
        
        seed_strategies[name] = [v_seed, alpha_seed, t_drop_seed]
        print(f"  - 为 {name} 量身定制的初始飞行方案: v={v_seed:.2f}, alpha={alpha_seed:.2f}, t_drop={t_drop_seed:.2f}")

    # 3. 组装成10维种子向量
    #    飞行策略是“量体裁衣”的，但引信依然继承Q3的智慧
    final_seed = np.array([
        *seed_strategies['FY1'],
        *seed_strategies['FY2'],
        *seed_strategies['FY3'],
        delay_base
    ])

    print(f"\n  - 战术推演完成，最终生成的10维种子为: {np.round(final_seed, 4)}")
    return final_seed

# --- 模块四: 寻优求解引擎 (PSO实现) ---
def pso_solver_q4():
    """
    使用增强型PSO算法解决问题四。
    """
    n_particles = 80
    n_dimensions = 10
    max_iter = 300
    w, c1, c2 = 0.5, 1.5, 1.5

    bounds = np.array([
        [70, 140], [0, 360], [0.1, 40],
        [70, 140], [0, 360], [0.1, 40],
        [70, 140], [0, 360], [0.1, 40],
        [0.1, 15]
    ])
    
    # --- 关键改动: 调用全新的、“量体裁衣”的种子生成器 ---
    q4_seed = generate_q4_seed_strategy_tailored()
    
    particles_pos = np.random.rand(n_particles, n_dimensions) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    particles_pos[0] = q4_seed
    
    particles_vel = np.zeros((n_particles, n_dimensions))
    
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([evaluate_strategy_q4(p) for p in pbest_pos])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest_pos = pbest_pos[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]

    print(f"\nPSO初始化完成，初始种群最优得分: {gbest_fitness:.4f} s (由战术种子贡献)")
    print("开始迭代寻优...")
    
    for k in range(max_iter):
        for i in range(n_particles):
            r1,r2 = np.random.rand(2, n_dimensions)
            cognitive_vel = c1 * r1 * (pbest_pos[i] - particles_pos[i])
            social_vel = c2 * r2 * (gbest_pos - particles_pos[i])
            particles_vel[i] = w * particles_vel[i] + cognitive_vel + social_vel
            particles_pos[i] = particles_pos[i] + particles_vel[i]
            particles_pos[i] = np.maximum(particles_pos[i], bounds[:, 0])
            particles_pos[i] = np.minimum(particles_pos[i], bounds[:, 1])
            current_fitness = evaluate_strategy_q4(particles_pos[i])
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
    print("         数学建模问题A - 问题四求解器 (战术推演版)")
    print("="*50)

    start_time = time.time()
    
    best_params, best_veiling_time = pso_solver_q4()
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n寻优完成！")
    print(f"  - 总耗时: {elapsed_time:.2f} 秒")

    print("\n" + "="*20 + " 问题四最优策略 " + "="*20)

    v1, a1, t1, v2, a2, t2, v3, a3, t3, delay = best_params
    params_list = [
        {'name': 'FY1', 'v': v1, 'alpha': a1, 't_drop': t1},
        {'name': 'FY2', 'v': v2, 'alpha': a2, 't_drop': t2},
        {'name': 'FY3', 'v': v3, 'alpha': a3, 't_drop': t3}
    ]
    uav_start_positions = {
        'FY1': model.UAV_START_POS,
        'FY2': np.array([12000, 1400, 1400]),
        'FY3': np.array([6000, -3000, 700])
    }

    print(f"\n  [+] 统一引信延迟 (det_delay): {delay:.4f} s")
    print("\n  [+] 各无人机最优策略及物理状态:")
    
    for p in params_list:
        alpha_rad = np.deg2rad(p['alpha'])
        uav_velocity = np.array([p['v'] * np.cos(alpha_rad), p['v'] * np.sin(alpha_rad), 0])
        start_pos = uav_start_positions[p['name']]
        
        drop_pos = model.get_uav_pos(p['t_drop'], start_pos, uav_velocity)
        detonation_pos = model.get_detonation_pos(start_pos, uav_velocity, p['t_drop'], delay)
        
        print(f"    --- 无人机 {p['name']} ---")
        print(f"      - 飞行速度: {p['v']:.2f} m/s, 飞行方向: {p['alpha']:.2f} 度")
        print(f"      - 投放时刻: {p['t_drop']:.2f} s")
        print(f"      - 投放点坐标: [{drop_pos[0]:.2f}, {drop_pos[1]:.2f}, {drop_pos[2]:.2f}] m")
        print(f"      - 起爆时刻: {p['t_drop'] + delay:.2f} s")
        print(f"      - 起爆点坐标: [{detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f}] m")

    print("\n  [+] 最终优化结果:")
    print(f"      - 在此三机协同策略下，对导弹M1的最大有效遮蔽时长为: {best_veiling_time:.4f} 秒")
    print("\n" + "="*58)
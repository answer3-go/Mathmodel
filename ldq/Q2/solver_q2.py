# solver_q2_pso.py
# 问题二的求解器文件：使用专业级PSO算法寻找最优策略。

import sys
import os
import numpy as np

# 添加父目录到路径以便导入Model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Model as model 
import time

# --- 模块三: 策略评价函数 (增强版) ---
def evaluate_strategy(x):
    """
    目标函数 (适应度函数)。
    接收一组策略参数 x，返回该策略的评价值（总遮蔽时长）。
    """
    try:
        v, alpha_deg, t_drop, detonation_delay = x
        alpha_rad = np.deg2rad(alpha_deg)
        uav_flight_velocity = np.array([v * np.cos(alpha_rad), v * np.sin(alpha_rad), 0])
        
        detonation_pos = model.get_detonation_pos(
            model.UAV_START_POS, uav_flight_velocity, t_drop, detonation_delay
        )
        
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
        # *** 改进建议 3: 异常信息记录而非直接归零 ***
        # 打印出异常信息，方便调试
        # print(f"评估参数 {x} 时出现计算异常: {e}") 
        # 返回一个非常小的值，而不是0，避免错误值主导优化
        return 1e-6 
    
    return total_veiling_time

# --- 模块四: 寻优求解引擎 (专业级PSO实现) ---
def pso_solver():
    """
    使用经过专业优化的粒子群优化算法解决问题二。
    """
    # 1. PSO 算法参数设置
    # *** 改进建议 1: 增加粒子数和迭代次数以加强搜索 ***
    n_particles = 50      # 增加粒子数量
    n_dimensions = 4      
    max_iter = 200        # 增加最大迭代次数
    w = 0.5               
    c1 = 1.5              
    c2 = 1.5              

    # 2. 定义决策变量的边界
    bounds = np.array([
        [70, 140], [0, 360], [0.1, 30], [0.1, 15]
    ])
    
    # 3. 初始化粒子群
    particles_pos = np.random.rand(n_particles, n_dimensions) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    
    # *** 改进建议 4: 保留第一题解作种群初始粒子 ***
    # 将第一个粒子强制设置为问题一的已知可行解
    q1_solution = np.array([120.0, 180.0, 1.5, 3.6])
    particles_pos[0] = q1_solution
    
    particles_vel = np.zeros((n_particles, n_dimensions)) # 速度通常初始化为0
    
    pbest_pos = np.copy(particles_pos)
    # 使用列表推导和并行计算的思想（虽然这里是串行，但结构更清晰）
    pbest_fitness = np.array([evaluate_strategy(p) for p in pbest_pos])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest_pos = pbest_pos[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]

    print("PSO初始化完成，最优种子得分为: {:.4f} s".format(gbest_fitness))
    print("开始迭代寻优...")
    
    # 4. 迭代寻优
    for k in range(max_iter):
        for i in range(n_particles):
            # *** 改进建议 2: 将 r1, r2 改为向量 ***
            r1 = np.random.rand(n_dimensions)
            r2 = np.random.rand(n_dimensions)
            
            cognitive_vel = c1 * r1 * (pbest_pos[i] - particles_pos[i])
            social_vel = c2 * r2 * (gbest_pos - particles_pos[i])
            particles_vel[i] = w * particles_vel[i] + cognitive_vel + social_vel
            
            particles_pos[i] = particles_pos[i] + particles_vel[i]
            
            particles_pos[i] = np.maximum(particles_pos[i], bounds[:, 0])
            particles_pos[i] = np.minimum(particles_pos[i], bounds[:, 1])
            
            current_fitness = evaluate_strategy(particles_pos[i])
            
            if current_fitness > pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest_pos[i] = particles_pos[i]
                
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_pos = particles_pos[i]
        
        # 避免过于频繁的打印，可以每10代打印一次
        if (k + 1) % 10 == 0:
            print(f"迭代 {k+1}/{max_iter}, 当前最优遮蔽时长: {gbest_fitness:.4f} s")
        
    return gbest_pos, gbest_fitness

# --- 主程序入口 ---
if __name__ == "__main__":
    print("="*50)
    print("         数学建模问题A - 问题二求解器 (PSO专业版)")
    print("="*50)

    start_time = time.time()
    
    best_params, best_veiling_time = pso_solver()
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n寻优完成！")
    print(f"  - 总耗时: {elapsed_time:.2f} 秒")

    print("\n" + "="*20 + " 问题二最优策略 " + "="*20)

    best_v, best_alpha_deg, best_t_drop, best_detonation_delay = best_params

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
    # *** 改进建议 4: 使用更高精度打印最终结果 ***
    print(f"      - 在此最优策略下，对导弹M1的最大有效遮蔽时长为: {best_veiling_time:.8f} 秒")

    print("\n" + "="*56)
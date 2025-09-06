# solver_q5.py
# 问题五的求解器文件：在一个文件中实现完整的分层优化策略，不依赖任何外部自定义模块。

import numpy as np
import time

# ============================================================================
# --- 基础模型库 (Base Models Library) ---
# 本区域包含了我们为解决此系列问题而建立的核心模型。
# 在求解问题五时，我们将直接复用这些已经过验证的模型函数。
# ============================================================================

# --- 模块一: 时空运动模型 ---
# 【复用自问题一至问题四的 `model.py` 文件】
# 目的: 提供计算所有物体在任意时刻位置的数学函数。

# 1.1 全局物理常量
GRAVITY = np.array([0, 0, -9.8])
CLOUD_SINK_SPEED = np.array([0, 0, -3.0])
SMOKE_RADIUS = 10.0
SMOKE_DURATION = 20.0
MISSILE_SPEED = 300.0

# 1.2 场景实体初始状态
FAKE_TARGET_POS = np.array([0, 0, 0])
REAL_TARGET_CENTER = np.array([0, 200, 5])
UAV_START_POSITIONS = {
    'FY1': np.array([17800, 0, 1800]),
    'FY2': np.array([12000, 1400, 1400]),
    'FY3': np.array([6000, -3000, 700]),
    'FY4': np.array([11000, 2000, 1800]),
    'FY5': np.array([13000, -2000, 1300]),
}
MISSILE_START_POSITIONS = {
    'M1': np.array([20000, 0, 2000]),
    'M2': np.array([19000, 600, 2100]),
    'M3': np.array([18000, -600, 1900])
}

# 1.3 核心运动学函数
def get_missile_pos_generic(t, missile_name):
    """计算任意导弹在任意时刻 t 的位置"""
    start_pos = MISSILE_START_POSITIONS[missile_name]
    direction = FAKE_TARGET_POS - start_pos
    unit_direction = direction / np.linalg.norm(direction)
    velocity_vector = MISSILE_SPEED * unit_direction
    return start_pos + velocity_vector * t

def get_uav_pos(t, uav_start_pos, uav_flight_velocity):
    """计算任意时刻 t 的无人机位置"""
    return uav_start_pos + uav_flight_velocity * t

def get_detonation_pos(uav_start_pos, uav_flight_velocity, drop_time, detonation_delay):
    """计算烟幕弹爆炸瞬间的位置"""
    drop_pos = get_uav_pos(drop_time, uav_start_pos, uav_flight_velocity)
    initial_grenade_velocity = uav_flight_velocity
    flight_time = detonation_delay
    detonation_pos = drop_pos + initial_grenade_velocity * flight_time + 0.5 * GRAVITY * flight_time**2
    return detonation_pos

def get_cloud_center_pos(t, detonation_pos, drop_time, detonation_delay):
    """计算爆炸后任意时刻 t 的烟幕云心位置"""
    detonation_time = drop_time + detonation_delay
    time_after_detonation = t - detonation_time
    return detonation_pos + CLOUD_SINK_SPEED * time_after_detonation

# --- 模块二: 遮蔽判定模型 ---
# 【复用自问题一至问题四的 `model.py` 文件】
# 目的: 提供一个在任意瞬间判断是否形成有效遮蔽的几何规则。
def check_veiling(missile_pos, cloud_pos):
    """判定原则: 导弹与真目标的视线线段，是否与烟幕云球体相交。"""
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
    if delta < 0: return False
    t1 = (-b - np.sqrt(delta)) / (2*a)
    t2 = (-b + np.sqrt(delta)) / (2*a)
    if (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1): return True
    return False

# ============================================================================
# --- 第 II 阶段：底层决策 - 小编队协同优化模块 ---
# ============================================================================

def evaluate_squad_strategy(x, uav_names, target_missile_name, smokes_per_uav):
    """
    【复用并扩展自问题三/四的策略评价函数】
    评估一个小编队的协同策略。
    """
    try:
        total_veiling_time = 0
        num_uavs = len(uav_names)
        params_per_uav = 1 + 1 + smokes_per_uav
        det_delay = x[-1]
        
        all_detonation_times = []
        all_detonation_positions = []
        all_drop_times = []

        for i in range(num_uavs):
            uav_name = uav_names[i]
            base_idx = i * params_per_uav
            v, alpha_deg = x[base_idx], x[base_idx + 1]
            drop_times_this_uav = x[base_idx + 2 : base_idx + 2 + smokes_per_uav]
            
            alpha_rad = np.deg2rad(alpha_deg)
            uav_velocity = np.array([v * np.cos(alpha_rad), v * np.sin(alpha_rad), 0])
            start_pos = UAV_START_POSITIONS[uav_name]
            
            for dt in drop_times_this_uav:
                all_drop_times.append(dt)
                all_detonation_times.append(dt + det_delay)
                all_detonation_positions.append(
                    get_detonation_pos(start_pos, uav_velocity, dt, det_delay)
                )

        dt_sim = 0.01
        t_end = 70.0
        time_steps = np.arange(0, t_end, dt_sim)
        
        target_missile_path = np.array([get_missile_pos_generic(t, target_missile_name) for t in time_steps])

        for i, t in enumerate(time_steps):
            missile_pos = target_missile_path[i]
            is_veiled_this_step = False
            for j in range(len(all_detonation_times)):
                if all_detonation_times[j] <= t < all_detonation_times[j] + SMOKE_DURATION:
                    cloud_pos = get_cloud_center_pos(t, all_detonation_positions[j], all_drop_times[j], det_delay)
                    if check_veiling(missile_pos, cloud_pos):
                        is_veiled_this_step = True
                        break
            if is_veiled_this_step:
                total_veiling_time += dt_sim
                    
    except Exception as e:
        return 1e-6 
    return total_veiling_time

def solve_subproblem(uav_names, target_missile_name, smokes_per_uav):
    """
    【V2.1 优化版】
    为一个独立的小编队协同任务，寻找最优策略，并增加进度反馈。
    """
    print(f"\n--- 开始求解子问题: {uav_names} vs {target_missile_name} ({smokes_per_uav * len(uav_names)}枚弹) ---")
    
    num_uavs = len(uav_names)
    params_per_uav = 1 + 1 + smokes_per_uav
    n_dimensions = num_uavs * params_per_uav + 1
    
    bounds = []
    for _ in range(num_uavs):
        bounds.extend([(70, 140), (0, 360)])
        for _ in range(smokes_per_uav):
            bounds.append((0.1, 40.0))
    bounds.append((0.1, 15.0))
    bounds = np.array(bounds)

    # *** 关键优化: 调整PSO参数以平衡效果和时间 ***
    n_particles = 10 * n_dimensions  # 适当减少粒子数
    max_iter = 25 * n_dimensions     # 适当减少迭代次数
    if n_dimensions < 10: # 对简单任务可以增加迭代深度
        max_iter = 200
        
    w, c1, c2 = 0.5, 1.5, 1.5

    particles_pos = np.random.rand(n_particles, n_dimensions) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    particles_vel = np.zeros((n_particles, n_dimensions))
    
    print(f"  - 子问题维度: {n_dimensions}, 粒子数: {n_particles}, 最大迭代: {max_iter}")
    print("  - 正在进行初始化评估...")
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([evaluate_squad_strategy(p, uav_names, target_missile_name, smokes_per_uav) for p in pbest_pos])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest_pos = pbest_pos[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]

    print(f"  - 子问题初始化完成，初始最优得分: {gbest_fitness:.4f} s")
    print("  - 开始迭代寻优 (每10代汇报一次进度)...")

    # *** 关键优化: 在迭代循环中增加进度打印 ***
    for k in range(max_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(2, n_dimensions)
            cognitive_vel = c1 * r1 * (pbest_pos[i] - particles_pos[i])
            social_vel = c2 * r2 * (gbest_pos - particles_pos[i])
            particles_vel[i] = w * particles_vel[i] + cognitive_vel + social_vel
            particles_pos[i] += particles_vel[i]
            particles_pos[i] = np.maximum(particles_pos[i], bounds[:, 0])
            particles_pos[i] = np.minimum(particles_pos[i], bounds[:, 1])
            current_fitness = evaluate_squad_strategy(particles_pos[i], uav_names, target_missile_name, smokes_per_uav)
            if current_fitness > pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest_pos[i] = particles_pos[i]
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_pos = particles_pos[i]
        
        # 每10代打印一次当前最优结果，作为进度反馈
        if (k + 1) % 10 == 0:
            print(f"    [进度] 迭代 {k+1}/{max_iter}, 当前最优: {gbest_fitness:.4f} s")

    print(f"--- 子问题求解完成: {uav_names} vs {target_missile_name}, 最优时长: {gbest_fitness:.4f} s ---")
    return gbest_pos, gbest_fitness

# ============================================================================
# --- 第 I 阶段：顶层决策 - 任务规划模块 ---
# ============================================================================
def threat_assessment(missiles):
    """
    对来袭导弹进行威胁评估。
    """
    print("\n[顶层决策 - 步骤1] --- 威胁评估 ---")
    threat_scores = {}
    for name, m_info in missiles.items():
        dist = np.linalg.norm(m_info['start'] - FAKE_TARGET_POS)
        t_impact = dist / MISSILE_SPEED
        p_start = m_info['start']
        v_dir = (FAKE_TARGET_POS - p_start) / np.linalg.norm(FAKE_TARGET_POS - p_start)
        t_closest = -np.dot(v_dir, p_start - REAL_TARGET_CENTER)
        p_closest = p_start + MISSILE_SPEED * t_closest * v_dir
        d_min = np.linalg.norm(p_closest - REAL_TARGET_CENTER)
        threat_score = 0.5 * (1 / t_impact) + 0.5 * (1 / d_min)
        threat_scores[name] = threat_score
        print(f"  - 导弹 {name}: 预计到达时间 ~{t_impact:.2f}s, 最小掠过距离 ~{d_min:.2f}m, 威胁值: {threat_score:.6f}")
    sorted_missiles = sorted(threat_scores.items(), key=lambda item: item[1], reverse=True)
    print(f"  - 威胁排序结果: {[m[0] for m in sorted_missiles]}")
    return sorted_missiles

def task_and_resource_allocation(sorted_missiles, uavs):
    """
    根据威胁排序，使用启发式规则进行任务和资源分配。
    """
    print("\n[顶层决策 - 步骤2] --- 任务与资源分配 ---")
    allocation = {}
    available_uavs = list(uavs.keys())
    
    # 启发式规则：为威胁最高的导弹分配2架，次高的2架，最低的1架
    uav_counts = [2, 2, 1]
    # 资源规则：每架无人机最多可使用3枚弹药
    smokes_per_uav = 3

    for i in range(len(sorted_missiles)):
        target_missile_name = sorted_missiles[i][0]
        num_uavs_to_assign = uav_counts[i]
        
        target_missile_start_pos = MISSILE_START_POSITIONS[target_missile_name]
        costs = {uav: np.linalg.norm(uavs[uav]['start'] - target_missile_start_pos) for uav in available_uavs}
        assigned_uavs = sorted(costs, key=costs.get)[:num_uavs_to_assign]
        
        allocation[target_missile_name] = {'uavs': assigned_uavs, 'smokes_per_uav': smokes_per_uav}
        for uav in assigned_uavs: available_uavs.remove(uav)

    print("  - 分配方案如下:")
    for missile, plan in allocation.items():
        print(f"    - 拦截任务 {missile}: 由无人机 {plan['uavs']} 执行, 每架最多使用 {plan['smokes_per_uav']} 枚弹。")
    return allocation

# ============================================================================
# --- 主程序入口 ---
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("         数学建模问题A - 问题五求解器 (单文件完整版)")
    print("="*60)

    start_time = time.time()
    
    # --- 关键修正: 定义一个完整的、包含所有信息的无人机定义字典 ---
    MISSILE_DEFS = {
        'M1': {'start': MISSILE_START_POSITIONS['M1'], 'target': FAKE_TARGET_POS},
        'M2': {'start': MISSILE_START_POSITIONS['M2'], 'target': FAKE_TARGET_POS},
        'M3': {'start': MISSILE_START_POSITIONS['M3'], 'target': FAKE_TARGET_POS},
    }
    UAV_DEFS = {
        'FY1': {'start': UAV_START_POSITIONS['FY1']},
        'FY2': {'start': UAV_START_POSITIONS['FY2']},
        'FY3': {'start': UAV_START_POSITIONS['FY3']},
        'FY4': {'start': UAV_START_POSITIONS['FY4']},
        'FY5': {'start': UAV_START_POSITIONS['FY5']},
    }
    
    # 执行顶层决策
    # --- 关键修正: 确保传入正确的数据结构 ---
    sorted_missiles_with_scores = threat_assessment(MISSILE_DEFS)
    allocation_plan = task_and_resource_allocation(sorted_missiles_with_scores, UAV_DEFS)

    # 循环求解所有子问题
    print("\n[底层决策 - 步骤3] --- 求解各小编队协同策略 ---")
    final_results = {}
    
    for missile_name, plan in allocation_plan.items():
        uav_names_for_task = plan['uavs']
        smokes_for_task = plan['smokes_per_uav']
        
        best_params, best_time = solve_subproblem(
            uav_names_for_task,
            missile_name,
            smokes_for_task,
        )
        
        final_results[missile_name] = {
            'team': uav_names_for_task,
            'params': best_params,
            'time': best_time
        }

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n全部子问题求解完成！总耗时: {elapsed_time:.2f} 秒")

    # --- 结果汇总与输出 (增强版) ---
    print("\n" + "="*22 + " 问题五最终最优策略 " + "="*22)
    
    # 指标一：总时长简单求和
    total_time_sum = sum(result['time'] for result in final_results.values())
    
    # 指标二：基于威胁加权的总时长
    threat_values = {name: score for name, score in sorted_missiles_with_scores}
    total_threat_score = sum(threat_values.values())
    
    weighted_total_time = 0
    for missile_name, result in final_results.items():
        weight = threat_values[missile_name] / total_threat_score
        weighted_total_time += weight * result['time']

    print(f"\n  [+] 总体优化结果 (双指标对比):")
    print(f"      - 指标一 (总时长简单求和): {total_time_sum:.4f} 秒")
    print(f"      - 指标二 (基于威胁加权的总有效时长): {weighted_total_time:.4f}")
    
    print("\n  [+] 各拦截任务详细最优策略:")
    for missile_name, result in final_results.items():
        print(f"\n    --- 拦截任务: {missile_name} (威胁值: {threat_values[missile_name]:.6f}, 最优时长: {result['time']:.4f}s) ---")
        
        plan = allocation_plan[missile_name]
        num_uavs = len(result['team'])
        smokes_per_uav = plan['smokes_per_uav']
        params_per_uav = 1 + 1 + smokes_per_uav
        
        params = result['params']
        delay = params[-1]
        print(f"      - 统一引信延迟: {delay:.4f} s")
        
        for i in range(num_uavs):
            uav_name = result['team'][i]
            base_idx = i * params_per_uav
            v = params[base_idx]
            alpha = params[base_idx + 1]
            drop_times = params[base_idx + 2 : base_idx + 2 + smokes_per_uav]
            start_pos = UAV_START_POSITIONS[uav_name]

            print(f"        - 无人机 {uav_name}:")
            print(f"          - 飞行策略: v={v:.2f} m/s, alpha={alpha:.2f} 度")
            for j, dt in enumerate(drop_times):
                print(f"          - 第 {j+1} 枚弹投放时刻: {dt:.2f} s")
    
    print("\n" + "="*60)
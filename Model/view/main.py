# main.py (V2.0 版本 - 修正最终版)
import numpy as np
from simulation import *
from visualization import plot_scenario_static_v2, animate_scenario

def run_full_simulation(missile_defs, uav_strategy, end_time=70.0, dt=0.1):
    """
    根据给定的导弹和无人机策略，运行一次完整的仿真。
    """
    time_vector = np.arange(0, end_time + dt, dt)
    simulation_data = {
        'time_vector': time_vector,
        'missile_paths': {},
        'uav_strategies': {},
        'target_info': {'center_x': 0, 'center_y': 200, 'radius': 7, 'height': 10}
    }

    # 计算所有导弹的轨迹
    for name, m_def in missile_defs.items():
        dir_vec = (m_def['target'] - m_def['start']) / np.linalg.norm(m_def['target'] - m_def['start'])
        vel_vec = m_def['speed'] * dir_vec
        path = calculate_entity_path(m_def['start'], vel_vec, time_vector)
        simulation_data['missile_paths'][name] = {'path': path, 'time': time_vector}

    # 计算所有无人机及其投放物的轨迹
    for name, plan in uav_strategy.items():
        uav_path = calculate_entity_path(plan['initial_pos'], plan['velocity'], time_vector)
        uav_data = {'uav_path': uav_path, 'uav_time': time_vector, 'smokes': []}
        
        for i in range(len(plan['drop_times'])):
            t_drop = plan['drop_times'][i]
            t_detonate = t_drop + plan['detonation_delays'][i]
            
            p_drop = calculate_entity_path(plan['initial_pos'], plan['velocity'], np.array([t_drop]))[0]
            v_drop = plan['velocity']

            grenade_path, grenade_time = calculate_grenade_path(p_drop, v_drop, t_drop, t_detonate, dt)
            p_detonate = grenade_path[-1]

            cloud_path, cloud_time = calculate_cloud_path(p_detonate, t_detonate, end_time, dt)
            
            uav_data['smokes'].append({
                'grenade_path': grenade_path, 'grenade_time': grenade_time,
                'cloud_path': cloud_path, 'cloud_time': cloud_time,
                't_detonate': t_detonate
            })
        simulation_data['uav_strategies'][name] = uav_data
    
    return simulation_data

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 1. 定义战场想定中的所有实体 ---
    P_FAKE_TARGET = np.array([0, 0, 0]) # 假目标位置

    MISSILE_DEFINITIONS = {
        'M1': {'start': np.array([20000, 0, 2000]), 'speed': 300.0, 'target': P_FAKE_TARGET},
        'M2': {'start': np.array([19000, 600, 2100]), 'speed': 300.0, 'target': P_FAKE_TARGET},
        'M3': {'start': np.array([18000, -600, 1900]), 'speed': 300.0, 'target': P_FAKE_TARGET},
    }

    # --- 2. 定义本次要运行的具体策略 ---
    # 这是问题1的策略
    uav_strategy_q1 = {
        'FY1': {
            'initial_pos': np.array([17800, 0, 1800]),
            'velocity': np.array([-120.0, 0, 0]),
            'drop_times': [1.5],
            'detonation_delays': [3.6]
        }
    }
    
    # --- 3. 运行指定的想定 ---
    # 选择问题1的想定: 只使用M1导弹和问题1的策略
    q1_missiles = {'M1': MISSILE_DEFINITIONS['M1']}
    simulation_results = run_full_simulation(q1_missiles, uav_strategy_q1)

    # --- 4. 选择可视化方式 ---
    
    # **方式一：生成并保存一张静态图**
    plot_scenario_static_v2(simulation_results, output_filename="result1_visualization.png")

    # **方式二：运行并保存交互式动画**
    # *** 修正：将参数名从 save_path 改为 output_filename ***
    animate_scenario(simulation_results, output_filename="result1_animation.mp4")
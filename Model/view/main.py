# main.py (V2.1 版本 - 最终版)
import numpy as np
from simulation import *
from visualization import plot_scenario_static_v2, animate_scenario

def run_full_simulation(missile_defs, uav_strategy, end_time=70.0, dt=0.1):
    # (此函数无需改动，代码省略，请使用你已有的版本)
    time_vector = np.arange(0, end_time + dt, dt)
    simulation_data = {
        'time_vector': time_vector,
        'missile_paths': {},
        'uav_strategies': {},
        'target_info': {'center_x': 0, 'center_y': 0, 'radius': 7, 'height': 10}
    }
    for name, m_def in missile_defs.items():
        dir_vec = (m_def['target'] - m_def['start']) / np.linalg.norm(m_def['target'] - m_def['start'])
        vel_vec = m_def['speed'] * dir_vec
        path = calculate_entity_path(m_def['start'], vel_vec, time_vector)
        simulation_data['missile_paths'][name] = {'path': path, 'time': time_vector}
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
    P_FAKE_TARGET = np.array([0, 0, 0])
    MISSILE_DEFINITIONS = {
        'M1': {'start': np.array([20000, 0, 2000]), 'speed': 300.0, 'target': P_FAKE_TARGET},
        'M2': {'start': np.array([19000, 600, 2100]), 'speed': 300.0, 'target': P_FAKE_TARGET},
        'M3': {'start': np.array([18000, -600, 1900]), 'speed': 300.0, 'target': P_FAKE_TARGET},
    }

    uav_strategy_q1 = {
        'FY1': {
            'initial_pos': np.array([17800, 0, 1800]),
            'velocity': np.array([-120.0, 0, 0]),
            'drop_times': [1.5],
            'detonation_delays': [3.6]
        }
    }
    
    q1_missiles = {'M1': MISSILE_DEFINITIONS['M1']}
    simulation_results = run_full_simulation(q1_missiles, uav_strategy_q1)
    # **方式二：生成动画 (使用新的聚焦视角！)**
    # *** V2.1 新功能: 添加 camera_mode='focus' 参数来启用智能摄像机 ***
    animate_scenario(simulation_results, camera_mode='focus', output_filename="result1_animation_focus.mp4")
    # **方式一：生成静态图 (全局视角)**
    plot_scenario_static_v2(simulation_results, output_filename="result1_visualization_static.png")

    
# visualization.py (V2.1 版本 - 革命性更新版)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import warnings
from simulation import SMOKE_LIFESPAN

# --- 中文字体和负号的正确显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 辅助函数 ---
def plot_cylinder(ax, center_x, center_y, radius, height_z, color='royalblue', alpha=0.3):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color=color)

def plot_sphere(ax, center, radius, color='orange', alpha=0.2):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

# --- V2.1 新增: 动态设置摄像机视角 ---
def set_camera_view(ax, uav_pos, missile_pos, mode='focus', zoom_factor=1000):
    """根据模式设置3D视图的中心和范围"""
    if mode == 'full':
        ax.set_xlim(-1000, 21000)
        ax.set_ylim(-1000, 1000)
        ax.set_zlim(0, 2500)
    elif mode == 'focus':
        center_point = (uav_pos + missile_pos) / 2.0
        ax.set_xlim(center_point[0] - zoom_factor, center_point[0] + zoom_factor)
        ax.set_ylim(center_point[1] - zoom_factor, center_point[1] + zoom_factor)
        ax.set_zlim(max(0, center_point[2] - zoom_factor), center_point[2] + zoom_factor)

# --- 静态图绘制函数 (V2.1 信息完整版) ---
def plot_scenario_static_v2(simulation_data, output_filename=None):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    label_dict = {}

    target_info = simulation_data['target_info']
    # 绘制真目标 (蓝色圆柱)
    plot_cylinder(ax, target_info['center_x'], target_info['center_y'], 
                  target_info['radius'], target_info['height'])
    if '真目标' not in label_dict:
        dummy_true = ax.scatter([], [], [], color='royalblue', marker='s', label='真目标')
        label_dict['真目标'] = dummy_true

    # 绘制假目标 (灰色圆柱)
    plot_cylinder(ax, 0, 0, target_info['radius'], target_info['height'], color='gray', alpha=0.3)
    if '假目标' not in label_dict:
        dummy_fake = ax.scatter([], [], [], color='gray', marker='s', label='假目标')
        label_dict['假目标'] = dummy_fake

    missile_colors = {'M1': 'red', 'M2': 'darkred', 'M3': 'firebrick'}
    for name, data in simulation_data['missile_paths'].items():
        path = data['path']
        color = missile_colors.get(name, 'black')
        line, = ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linestyle='-', label=f'轨迹 {name}')
        label_dict[f'轨迹 {name}'] = line
        scatter = ax.scatter(path[0, 0], path[0, 1], path[0, 2], color=color, marker='x', s=100, label=f'起始点 {name}')
        label_dict[f'起始点 {name}'] = scatter

    uav_colors = {'FY1': 'blue', 'FY2': 'cyan', 'FY3': 'purple', 'FY4': 'deepskyblue', 'FY5': 'navy'}
    for name, data in simulation_data['uav_strategies'].items():
        path = data['uav_path']
        color = uav_colors.get(name, 'gray')
        if '无人机轨迹' not in label_dict:
            line, = ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linestyle='--', label='无人机轨迹')
            label_dict['无人机轨迹'] = line
        else:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linestyle='--')
        
        if '无人机起始点' not in label_dict:
            scatter = ax.scatter(path[0, 0], path[0, 1], path[0, 2], color='blue', marker='^', s=100, label='无人机起始点')
            label_dict['无人机起始点'] = scatter
        else:
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], color='blue', marker='^', s=100)

        for i, smoke in enumerate(data['smokes']):
            p_drop = smoke['grenade_path'][0]
            p_detonate = smoke['cloud_path'][0]
            if '投放点' not in label_dict:
                scatter = ax.scatter(p_drop[0], p_drop[1], p_drop[2], color='green', marker='o', label='投放点')
                label_dict['投放点'] = scatter
            else:
                 ax.scatter(p_drop[0], p_drop[1], p_drop[2], color='green', marker='o')

            if '起爆点' not in label_dict:
                scatter = ax.scatter(p_detonate[0], p_detonate[1], p_detonate[2], color='orange', marker='*', s=150, label='起爆点')
                label_dict['起爆点'] = scatter
            else:
                 ax.scatter(p_detonate[0], p_detonate[1], p_detonate[2], color='orange', marker='*', s=150)
            
            plot_sphere(ax, p_detonate, 10)
            
            # 补上烟幕弹抛物线轨迹
            grenade_path = smoke['grenade_path']
            if '烟幕弹轨迹' not in label_dict:
                line_g, = ax.plot(grenade_path[:,0], grenade_path[:,1], grenade_path[:,2], color='green', linestyle=':', alpha=0.7, label='烟幕弹轨迹')
                label_dict['烟幕弹轨迹'] = line_g
            else:
                ax.plot(grenade_path[:,0], grenade_path[:,1], grenade_path[:,2], color='green', linestyle=':', alpha=0.7)

    ax.set_xlim(-1000, 21000); ax.set_ylim(-1000, 1000); ax.set_zlim(0, 2500)
    ax.set_xlabel('X 轴 (m)'); ax.set_ylabel('Y 轴 (m)'); ax.set_zlabel('Z 轴 (m)')
    ax.set_title('战场想定三维静态可视化')
    ax.legend(label_dict.values(), label_dict.keys())

    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        warnings.warn("您的Matplotlib版本较低，无法使用set_box_aspect。")

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"静态图已保存至: {output_filename}")
    plt.show()

# --- 动画生成函数 (V2.1 智能摄像机版) ---
def animate_scenario(simulation_data, camera_mode='full', output_filename=None):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    time_vector = simulation_data['time_vector']
    
    legend_text = ("图例:\n"
                   "红色 X: 导弹\n" "蓝色 Δ: 无人机\n" "橙色球: 烟幕云\n"
                   "蓝色圆柱: 真目标\n" "灰色圆柱: 假目标\n" "绿色虚线: 烟幕弹轨迹")
    fig.text(0.02, 0.98, legend_text, fontsize=12, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    def update(frame):
        ax.clear()
        current_time = time_vector[frame]
        
        m1_data = next(iter(simulation_data['missile_paths'].values()))
        m1_idx = np.searchsorted(m1_data['time'], current_time)
        m1_pos = m1_data['path'][m1_idx - 1] if m1_idx > 0 else m1_data['path'][0]

        fy1_data = next(iter(simulation_data['uav_strategies'].values()))
        fy1_idx = np.searchsorted(fy1_data['uav_time'], current_time)
        fy1_pos = fy1_data['uav_path'][fy1_idx - 1] if fy1_idx > 0 else fy1_data['uav_path'][0]

        target_info = simulation_data['target_info']
        plot_cylinder(ax, target_info['center_x'], target_info['center_y'], 
                      target_info['radius'], target_info['height'])
        plot_cylinder(ax, 0, 0, target_info['radius'], target_info['height'], color='gray', alpha=0.3)
        
        missile_colors = {'M1': 'red', 'M2': 'darkred', 'M3': 'firebrick'}
        for name, data in simulation_data['missile_paths'].items():
            path, time = data['path'], data['time']
            idx = np.searchsorted(time, current_time)
            if idx > 0:
                color = missile_colors.get(name, 'black')
                ax.plot(path[:idx, 0], path[:idx, 1], path[:idx, 2], color=color)
                ax.scatter(path[idx-1, 0], path[idx-1, 1], path[idx-1, 2], color=color, marker='x', s=100)

        uav_colors = {'FY1': 'blue', 'FY2': 'cyan', 'FY3': 'purple', 'FY4': 'deepskyblue', 'FY5': 'navy'}
        for name, data in simulation_data['uav_strategies'].items():
            path, time = data['uav_path'], data['uav_time']
            idx = np.searchsorted(time, current_time)
            if idx > 0:
                color = uav_colors.get(name, 'gray')
                ax.plot(path[:idx, 0], path[:idx, 1], path[:idx, 2], color=color, linestyle='--')
                ax.scatter(path[idx-1, 0], path[idx-1, 1], path[idx-1, 2], color=color, marker='^', s=100)
            
            for smoke in data['smokes']:
                if smoke['t_detonate'] <= current_time < smoke['t_detonate'] + SMOKE_LIFESPAN:
                    cloud_time, cloud_path = smoke['cloud_time'], smoke['cloud_path']
                    cloud_idx = np.searchsorted(cloud_time, current_time)
                    if cloud_idx > 0:
                        center = cloud_path[cloud_idx - 1]
                        plot_sphere(ax, center, 10)
                grenade_time, grenade_path = smoke['grenade_time'], smoke['grenade_path']
                idx_g = np.searchsorted(grenade_time, current_time)
                if idx_g > 0:
                    ax.plot(grenade_path[:idx_g,0], grenade_path[:idx_g,1], grenade_path[:idx_g,2], color='green', linestyle=':', alpha=0.7)

        ax.set_title(f'战场想定三维动态可视化 (时间: {current_time:.1f}s)')
        ax.set_xlabel('X 轴 (m)'); ax.set_ylabel('Y 轴 (m)'); ax.set_zlabel('Z 轴 (m)')
        set_camera_view(ax, fy1_pos, m1_pos, mode=camera_mode)
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError: pass
    
    ani = FuncAnimation(fig, update, frames=len(time_vector), blit=False, interval=50)

    if output_filename:
        try:
            ani.save(output_filename, writer='ffmpeg', fps=20, dpi=150)
            print(f"动画已成功保存至: {output_filename}")
        except Exception as e:
            print(f"保存动画失败: {e}")
            print("错误！请确认您已正确安装 'ffmpeg' 并且已将其添加到了系统的环境变量(PATH)中。")
    plt.show()
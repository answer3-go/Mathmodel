# visualization.py (V2.0 版本 - 优化版)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import warnings

# --- 中文字体和负号的正确显示 ---
# 这段代码确保图中的中文和负号都能正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 绘制基本形状的辅助函数 ---

def plot_cylinder(ax, center_x, center_y, radius, height_z, color='royalblue', alpha=0.3):
    """在指定的3D坐标轴上绘制一个圆柱体"""
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color=color)

def plot_sphere(ax, center, radius, color='orange', alpha=0.2):
    """在指定的3D坐标轴上绘制一个球体 (用于表示烟幕云)"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    # 返回绘制的对象，方便在动画中移除
    return ax.plot_surface(x, y, z, color=color, alpha=alpha)


# --- 2. V2.0 版本的静态图绘制函数 ---
def plot_scenario_static_v2(simulation_data, output_filename=None):
    """根据仿真数据，生成一张高质量的静态3D轨迹图"""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    label_dict = {}

    target_info = simulation_data['target_info']
    plot_cylinder(ax, target_info['center_x'], target_info['center_y'], 
                  target_info['radius'], target_info['height'])

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

    ax.set_xlabel('X 轴 (m)'), ax.set_ylabel('Y 轴 (m)'), ax.set_zlabel('Z 轴 (m)')
    ax.set_title('战场想定三维静态可视化')
    ax.legend(label_dict.values(), label_dict.keys())
    
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        warnings.warn("您的Matplotlib版本较低，无法使用set_box_aspect。三维图形比例可能失真，建议升级至 3.3 或更高版本。")

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"静态图已保存至: {output_filename}")

    plt.show()

# --- 3. V2.0 版本的动画生成函数 ---
def animate_scenario(simulation_data, output_filename=None):
    """根据仿真数据，生成一段动态的3D演进动画"""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    time_vector = simulation_data['time_vector']

    def init():
        ax.clear()
        plot_cylinder(ax, simulation_data['target_info']['center_x'], simulation_data['target_info']['center_y'],
                      simulation_data['target_info']['radius'], simulation_data['target_info']['height'])
        ax.set_xlabel('X 轴 (m)'), ax.set_ylabel('Y 轴 (m)'), ax.set_zlabel('Z 轴 (m)')
        ax.set_title('战场想定三维动态可视化')
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError: pass
        return fig,

    dynamic_elements = []

    def update(frame):
        for element in dynamic_elements:
            element.remove()
        dynamic_elements.clear()

        current_time = time_vector[frame]
        ax.set_title(f'战场想定三维动态可视化 (时间: {current_time:.1f}s)')

        missile_colors = {'M1': 'red', 'M2': 'darkred', 'M3': 'firebrick'}
        for name, data in simulation_data['missile_paths'].items():
            path, time = data['path'], data['time']
            idx = np.searchsorted(time, current_time)
            if idx > 0:
                color = missile_colors.get(name, 'black')
                line, = ax.plot(path[:idx, 0], path[:idx, 1], path[:idx, 2], color=color)
                scatter = ax.scatter(path[idx-1, 0], path[idx-1, 1], path[idx-1, 2], color=color, marker='x', s=100)
                dynamic_elements.extend([line, scatter])

        uav_colors = {'FY1': 'blue', 'FY2': 'cyan', 'FY3': 'purple', 'FY4': 'deepskyblue', 'FY5': 'navy'}
        for name, data in simulation_data['uav_strategies'].items():
            path, time = data['uav_path'], data['uav_time']
            idx = np.searchsorted(time, current_time)
            if idx > 0:
                color = uav_colors.get(name, 'gray')
                line, = ax.plot(path[:idx, 0], path[:idx, 1], path[:idx, 2], color=color, linestyle='--')
                scatter = ax.scatter(path[idx-1, 0], path[idx-1, 1], path[idx-1, 2], color=color, marker='^', s=100)
                dynamic_elements.extend([line, scatter])
            
            for smoke in data['smokes']:
                if smoke['t_detonate'] <= current_time < smoke['t_detonate'] + SMOKE_LIFESPAN:
                    cloud_time, cloud_path = smoke['cloud_time'], smoke['cloud_path']
                    cloud_idx = np.searchsorted(cloud_time, current_time)
                    if cloud_idx > 0:
                        center = cloud_path[cloud_idx - 1]
                        sphere_surface = plot_sphere(ax, center, 10)
                        dynamic_elements.append(sphere_surface)
        
        return fig,

    ani = FuncAnimation(fig, update, frames=len(time_vector), init_func=init, blit=False, interval=50)

    if output_filename:
        try:
            ani.save(output_filename, writer='ffmpeg', fps=20, dpi=150)
            print(f"动画已保存至: {output_filename}。(注意: 这需要您的电脑安装了 'ffmpeg' 工具)")
        except Exception as e:
            print(f"保存动画失败: {e}")
            print("请确认您已安装 'ffmpeg' 并将其添加到了系统环境变量中。")

    plt.show()
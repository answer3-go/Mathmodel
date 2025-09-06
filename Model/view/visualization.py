# visualization.py (V3.3 可视化增强版)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import warnings
from simulation import SMOKE_LIFESPAN

# --- 可视化缩放配置 (V4.0 夸张化显示版) ---
VISUAL_SCALE = {
    'target_scale': 30,        # 目标圆柱体缩放倍数
    'smoke_cloud_radius': 200,  # 烟云球体半径 (大幅增加，原25)
    'missile_marker_size': 800, # 导弹标记大小 (大幅增加，原200)
    'uav_marker_size': 600,    # 无人机标记大小 (大幅增加，原150)
    'trajectory_linewidth': 8, # 轨迹线宽度 (大幅增加，原3)
    'explosion_base_radius': 300, # 爆炸球体基础半径 (大幅增加，原80)
    'detonation_marker_size': 1000, # 起爆点标记大小 (大幅增加，原250)
    'drop_marker_size': 500,   # 投放点标记大小 (新增)
    'grenade_trajectory_linewidth': 6, # 烟幕弹轨迹线宽 (新增)
    'uav_trajectory_linewidth': 6,     # 无人机轨迹线宽 (新增)
    'missile_trajectory_linewidth': 8  # 导弹轨迹线宽 (新增)
}

# --- 中文字体和负号的正确显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 辅助函数 (保持不变) ---
def plot_cylinder(ax, center_x, center_y, radius, height_z, color='royalblue', alpha=0.3):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color=color)

def plot_sphere(ax, center, radius, color='orange', alpha=0.2, edgecolor=None, linewidth=1):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    
    # 如果指定了边框颜色，添加边框效果
    if edgecolor:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, rstride=4, cstride=4, 
                       edgecolor=edgecolor, linewidth=linewidth)
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, rstride=4, cstride=4)

# --- 静态图绘制函数 (保持V2.5的完整功能) ---
def plot_scenario_static_v2(simulation_data, output_filename=None):
    # (代码省略, 请使用V2.5的最终版本)
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    label_dict = {}
    axis_length = 2000
    ax.plot([0, axis_length], [0, 0], [0, 0], color='black', linewidth=1.5, alpha=0.6)
    ax.plot([0, 0], [0, axis_length], [0, 0], color='black', linewidth=1.5, alpha=0.6)
    ax.plot([0, 0], [0, 0], [0, axis_length], color='black', linewidth=1.5, alpha=0.6)
    ax.text(axis_length, 0, 0, 'X', color='black', fontsize=12)
    ax.text(0, axis_length, 0, 'Y', color='black', fontsize=12)
    ax.text(0, 0, axis_length, 'Z', color='black', fontsize=12)
    # 使用新的缩放配置
    target_info = simulation_data['target_info']
    plot_cylinder(ax, target_info['center_x'], target_info['center_y'], 
                  target_info['radius'] * VISUAL_SCALE['target_scale'], 
                  target_info['height'] * VISUAL_SCALE['target_scale'])
    plot_cylinder(ax, 0, 0, 
                  target_info['radius'] * VISUAL_SCALE['target_scale'], 
                  target_info['height'] * VISUAL_SCALE['target_scale'], 
                  color='gray', alpha=0.3)
    dummy_true = ax.scatter([], [], [], color='royalblue', marker='s', s=100, label='真目标 (示意放大)')
    label_dict['真目标 (示意放大)'] = dummy_true
    dummy_fake = ax.scatter([], [], [], color='gray', marker='s', s=100, label='假目标 (示意放大)')
    label_dict['假目标 (示意放大)'] = dummy_fake
    missile_colors = {'M1': 'red', 'M2': 'darkred', 'M3': 'firebrick'}
    for name, data in simulation_data['missile_paths'].items():
        path = data['path']
        color = missile_colors.get(name, 'black')
        line, = ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, 
                       linestyle='-', linewidth=VISUAL_SCALE['missile_trajectory_linewidth'], 
                       label=f'轨迹 {name}')
        label_dict[f'轨迹 {name}'] = line
        scatter = ax.scatter(path[0, 0], path[0, 1], path[0, 2], color=color, 
                           marker='x', s=VISUAL_SCALE['missile_marker_size'], 
                           label=f'起始点 {name}')
        label_dict[f'起始点 {name}'] = scatter
    uav_colors = {'FY1': 'blue', 'FY2': 'cyan', 'FY3': 'purple', 'FY4': 'deepskyblue', 'FY5': 'navy'}
    for name, data in simulation_data['uav_strategies'].items():
        path = data['uav_path']
        color = uav_colors.get(name, 'gray')
        if '无人机轨迹' not in label_dict:
            line, = ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, 
                           linestyle='--', linewidth=VISUAL_SCALE['uav_trajectory_linewidth'], 
                           label='无人机轨迹')
            label_dict['无人机轨迹'] = line
        else:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, 
                   linestyle='--', linewidth=VISUAL_SCALE['uav_trajectory_linewidth'])
        if '无人机起始点' not in label_dict:
            scatter = ax.scatter(path[0, 0], path[0, 1], path[0, 2], color='blue', 
                               marker='^', s=VISUAL_SCALE['uav_marker_size'], 
                               label='无人机起始点')
            label_dict['无人机起始点'] = scatter
        else:
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], color='blue', 
                      marker='^', s=VISUAL_SCALE['uav_marker_size'])
        for i, smoke in enumerate(data['smokes']):
            p_drop = smoke['grenade_path'][0]
            p_detonate = smoke['cloud_path'][0]
            if '投放点' not in label_dict:
                scatter = ax.scatter(p_drop[0], p_drop[1], p_drop[2], color='green', marker='o', 
                                   s=VISUAL_SCALE['drop_marker_size'], label='投放点')
                label_dict['投放点'] = scatter
            else:
                 ax.scatter(p_drop[0], p_drop[1], p_drop[2], color='green', marker='o', 
                           s=VISUAL_SCALE['drop_marker_size'])
            if '起爆点' not in label_dict:
                scatter = ax.scatter(p_detonate[0], p_detonate[1], p_detonate[2], color='orange', 
                                   marker='*', s=VISUAL_SCALE['detonation_marker_size'], 
                                   label='起爆点')
                label_dict['起爆点'] = scatter
            else:
                 ax.scatter(p_detonate[0], p_detonate[1], p_detonate[2], color='orange', 
                           marker='*', s=VISUAL_SCALE['detonation_marker_size'])
            grenade_path = smoke['grenade_path']
            if '烟幕弹轨迹' not in label_dict:
                line_g, = ax.plot(grenade_path[:,0], grenade_path[:,1], grenade_path[:,2], 
                                 color='green', linestyle=':', alpha=0.8, 
                                 linewidth=VISUAL_SCALE['grenade_trajectory_linewidth'], 
                                 label='烟幕弹轨迹')
                label_dict['烟幕弹轨迹'] = line_g
            else:
                ax.plot(grenade_path[:,0], grenade_path[:,1], grenade_path[:,2], 
                       color='green', linestyle=':', alpha=0.8, 
                       linewidth=VISUAL_SCALE['grenade_trajectory_linewidth'])
            
            # 在静态图中显示烟雾云效果（显示起爆后的烟雾云）
            if 't_detonate' in smoke and 'cloud_path' in smoke:
                # 显示烟雾云（在起爆点位置显示一个大的烟雾球）
                detonation_pos = smoke['cloud_path'][0]
                plot_sphere(ax, detonation_pos, VISUAL_SCALE['smoke_cloud_radius'], 
                           color='orange', alpha=0.4, edgecolor='darkorange', linewidth=2)
                
                # 添加烟雾云图例（只添加一次）
                if '烟幕云' not in label_dict:
                    dummy_smoke = ax.scatter([], [], [], color='orange', marker='o', s=100, 
                                           label='烟幕云 (示意放大)')
                    label_dict['烟幕云 (示意放大)'] = dummy_smoke
    
    # 优化视角设置，重点关注目标区域和烟雾效果
    ax.set_xlim(-2000, 5000); ax.set_ylim(-1000, 1000); ax.set_zlim(0, 1000)
    ax.set_xlabel('X 轴 (m)'), ax.set_ylabel('Y 轴 (m)'), ax.set_zlabel('Z 轴 (m)')
    ax.set_title('战场想定三维静态可视化 (夸张化显示版)')
    
    # 设置最佳观察角度：俯视角度，便于观察烟雾遮挡效果
    ax.view_init(elev=30, azim=45)  # 仰角30度，方位角45度
    
    ax.legend(label_dict.values(), label_dict.keys())
    try: ax.set_box_aspect([1, 1, 1])
    except AttributeError: warnings.warn("您的Matplotlib版本较低，无法使用set_box_aspect。")
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"静态图已保存至: {output_filename}")
    plt.show()

# --- 动画生成函数 (V3.2 稳定可调最终版) ---
def animate_scenario(simulation_data, output_filename=None, quality='high'):
    if output_filename is None: print("进入快速预览模式，动画将不会被保存。")
    
    # 扩展质量选项
    if quality == 'ultra_draft':
        fps, dpi = (5, 50)  # 极低帧率和分辨率
    elif quality == 'draft':
        fps, dpi = (10, 100)
    else:
        fps, dpi = (20, 150)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    time_vector = simulation_data['time_vector']
    
    legend_text = ("图例 (夸张化显示):\n"
                   "红色 X: 导弹 (放大显示)\n" "蓝色 Δ: 无人机 (放大显示)\n" "橙色球: 烟幕云 (放大显示)\n"
                   "蓝色圆柱: 真目标\n" "灰色圆柱: 假目标\n" "绿色虚线: 烟幕弹轨迹 (加粗显示)")
    fig.text(0.02, 0.98, legend_text, fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    def update(frame):
        ax.clear()
        current_time = time_vector[frame]
        
        target_info = simulation_data['target_info']
        plot_cylinder(ax, target_info['center_x'], target_info['center_y'], 
                      target_info['radius'] * VISUAL_SCALE['target_scale'], 
                      target_info['height'] * VISUAL_SCALE['target_scale'])
        plot_cylinder(ax, 0, 0, 
                      target_info['radius'] * VISUAL_SCALE['target_scale'], 
                      target_info['height'] * VISUAL_SCALE['target_scale'], 
                      color='gray', alpha=0.3)
        
        all_positions = []
        
        missile_pos = None
        missile_colors = {'M1': 'red', 'M2': 'darkred', 'M3': 'firebrick'}
        for name, data in simulation_data['missile_paths'].items():
            path, time = data['path'], data['time']
            idx = np.searchsorted(time, current_time)
            if idx > 0:
                missile_pos = path[idx - 1]
                all_positions.append(missile_pos)
                color = missile_colors.get(name, 'black')
                # 增加轨迹线宽
                ax.plot(path[:idx, 0], path[:idx, 1], path[:idx, 2], color=color, 
                       linewidth=VISUAL_SCALE['missile_trajectory_linewidth'])
                # 增大导弹标记
                ax.scatter(missile_pos[0], missile_pos[1], missile_pos[2], color=color, 
                          marker='x', s=VISUAL_SCALE['missile_marker_size'])
                # 导弹撞击地面爆炸效果增强
                if missile_pos[2] <= 1:
                    plot_sphere(ax, missile_pos, VISUAL_SCALE['explosion_base_radius'], 
                               color='red', alpha=0.6, edgecolor='darkred', linewidth=2)

        uav_colors = {'FY1': 'blue', 'FY2': 'cyan', 'FY3': 'purple', 'FY4': 'deepskyblue', 'FY5': 'navy'}
        for name, data in simulation_data['uav_strategies'].items():
            # *** 关键错误修正: 使用 'uav_time' 而不是 'time' ***
            path, time = data['uav_path'], data['uav_time'] 
            idx = np.searchsorted(time, current_time)
            if idx > 0:
                uav_pos = path[idx - 1]
                all_positions.append(uav_pos)
                color = uav_colors.get(name, 'gray')
                # 增加无人机轨迹线宽
                ax.plot(path[:idx, 0], path[:idx, 1], path[:idx, 2], color=color, 
                       linestyle='--', linewidth=VISUAL_SCALE['uav_trajectory_linewidth'])
                # 增大无人机标记
                ax.scatter(uav_pos[0], uav_pos[1], uav_pos[2], color=color, 
                          marker='^', s=VISUAL_SCALE['uav_marker_size'])
            
            for smoke in data['smokes']:
                if smoke['t_detonate'] <= current_time < smoke['t_detonate'] + SMOKE_LIFESPAN:
                    cloud_time, cloud_path = smoke['cloud_time'], smoke['cloud_path']
                    cloud_idx = np.searchsorted(cloud_time, current_time)
                    if cloud_idx > 0:
                        cloud_pos = cloud_path[cloud_idx-1]
                        all_positions.append(cloud_pos)
                        # 增大烟云半径，添加边框
                        plot_sphere(ax, cloud_pos, VISUAL_SCALE['smoke_cloud_radius'], 
                                    color='orange', alpha=0.3, edgecolor='darkorange', linewidth=1)
                
                explosion_duration = 0.5
                # 增强爆炸效果
                if smoke['t_detonate'] <= current_time < smoke['t_detonate'] + explosion_duration:
                    time_after_detonation = current_time - smoke['t_detonate']
                    detonation_pos = smoke['cloud_path'][0]
                    radius = VISUAL_SCALE['explosion_base_radius'] * (time_after_detonation / explosion_duration)
                    alpha = 0.8 * (1 - time_after_detonation / explosion_duration)
                    plot_sphere(ax, detonation_pos, radius, color='yellow', alpha=alpha, 
                               edgecolor='orange', linewidth=2)
                
                grenade_time, grenade_path = smoke['grenade_time'], smoke['grenade_path']
                idx_g = np.searchsorted(grenade_time, current_time)
                # 增强烟幕弹轨迹线宽和透明度
                if idx_g > 0: 
                     ax.plot(grenade_path[:idx_g,0], grenade_path[:idx_g,1], grenade_path[:idx_g,2], 
                            color='green', linestyle=':', alpha=0.8, 
                            linewidth=VISUAL_SCALE['grenade_trajectory_linewidth'])

        # 固定相机视角，确保能看到所有重要元素
        # 设置固定的坐标轴范围，重点关注目标区域和烟雾效果
        ax.set_xlim(-2000, 5000)   # 扩大X轴范围，包含烟雾弹轨迹
        ax.set_ylim(-1000, 1000)   # 扩大Y轴范围，包含真假目标
        ax.set_zlim(0, 1000)       # 降低Z轴上限，重点关注低空区域
        
        # 设置最佳观察角度：俯视角度，便于观察烟雾遮挡效果
        ax.view_init(elev=30, azim=45)  # 仰角30度，方位角45度

        ax.set_title(f'战场想定三维动态可视化 (夸张化显示版) (时间: {current_time:.1f}s)')
        ax.set_xlabel('X 轴 (m)'), ax.set_ylabel('Y 轴 (m)'), ax.set_zlabel('Z 轴 (m)')
        try: ax.set_box_aspect([1, 1, 1])
        except AttributeError: pass
    
    ani = FuncAnimation(fig, update, frames=len(time_vector), blit=False, interval=50)
    
    if output_filename:
        print(f"开始保存动画至 {output_filename} (质量: {quality})... 这可能需要几分钟时间。")
        ani.save(output_filename, writer='ffmpeg', fps=fps, dpi=dpi)
        print(f"动画已成功保存。")
    else:
        plt.show()
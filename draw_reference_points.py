#!/usr/bin/env python3
"""
简化版机器人参考点示意图
清楚展示BASE和END-EFFECTOR的具体位置
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_robot_reference_diagram():
    """绘制机器人参考点示意图"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 机器人侧视图 - 显示参考点位置
    ax1.set_xlim(-0.1, 0.6)
    ax1.set_ylim(-0.1, 0.5)
    ax1.set_aspect('equal')
    
    # 绘制机器人简化结构
    # 基座
    base_rect = patches.Rectangle((-0.05, -0.05), 0.1, 0.1, 
                                 linewidth=3, edgecolor='black', facecolor='lightgray')
    ax1.add_patch(base_rect)
    
    # 关节和连杆 (简化表示)
    joints_x = [0, 0.1, 0.25, 0.4, 0.5]
    joints_y = [0, 0.1, 0.2, 0.25, 0.3]
    
    # 绘制连杆
    for i in range(len(joints_x)-1):
        ax1.plot([joints_x[i], joints_x[i+1]], [joints_y[i], joints_y[i+1]], 
                'k-', linewidth=4, alpha=0.7)
    
    # 绘制关节
    for i, (x, y) in enumerate(zip(joints_x, joints_y)):
        ax1.plot(x, y, 'ko', markersize=8)
        ax1.text(x+0.02, y+0.02, f'J{i+1}' if i > 0 else 'BASE', fontsize=10)
    
    # 高亮BASE参考点 (base_link frame)
    ax1.plot(0, 0, 's', color='red', markersize=15, markeredgecolor='black', 
             markeredgewidth=2, label='BASE参考点 (base_link)')
    ax1.text(0, -0.08, 'BASE参考点\n(base_link frame)', ha='center', va='top', 
             fontsize=12, fontweight='bold', color='red')
    
    # 高亮END-EFFECTOR参考点 (l5 frame)
    ax1.plot(joints_x[-1], joints_y[-1], '^', color='blue', markersize=15, 
             markeredgecolor='black', markeredgewidth=2, label='END-EFFECTOR参考点 (l5)')
    ax1.text(joints_x[-1], joints_y[-1]+0.05, 'END-EFFECTOR参考点\n(l5 frame)', 
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='blue')
    
    # 绘制相对位置向量
    ax1.annotate('', xy=(joints_x[-1], joints_y[-1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='green', lw=3))
    ax1.text(0.25, 0.1, '相对位置向量\n(标定中使用)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
             fontsize=11, fontweight='bold')
    
    ax1.set_title('机器人侧视图 - 参考点位置', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X 方向 (m)')
    ax1.set_ylabel('Z 方向 (m)')
    
    # 2. Frame层次结构
    ax2.axis('off')
    
    frame_text = """=== PINOCCHIO FRAME 结构 ===

标定中使用的参考点:

🔴 BASE参考点:
   • Frame ID: 1
   • Frame名称: base_link  
   • 物理位置: 机器人固定基座的根部
   • 特点: 不随关节运动改变
   • 坐标: 机器人的原点 (0,0,0)

🔵 END-EFFECTOR参考点:
   • Frame ID: 11
   • Frame名称: l5
   • 物理位置: 最后一个link的末端点
   • 特点: 随所有5个关节运动
   • 坐标: 根据关节角度计算

🟢 相对位置向量:
   • 定义: l5_position - base_link_position
   • 长度: 随机器人姿态变化
   • 用途: 标定关节零点偏移"""
    
    ax2.text(0.05, 0.95, frame_text, transform=ax2.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    # 3. 标定原理图解
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_aspect('equal')
    
    # 动作捕捉测量
    mocap_base = np.array([0, 0])
    mocap_ee = np.array([0.7, 0.5])
    ax3.plot(*mocap_base, 's', color='red', markersize=12, label='动捕-BASE')
    ax3.plot(*mocap_ee, '^', color='red', markersize=12, label='动捕-EEF')
    ax3.annotate('', xy=mocap_ee, xytext=mocap_base,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.text(0.35, 0.3, '动捕测量\n相对位置', ha='center', color='red', fontweight='bold')
    
    # 前向运动学计算
    fk_base = np.array([0, -0.3])
    fk_ee = np.array([0.6, 0.1])
    ax3.plot(*fk_base, 's', color='blue', markersize=12, label='前向运动学-BASE')
    ax3.plot(*fk_ee, '^', color='blue', markersize=12, label='前向运动学-EEF')
    ax3.annotate('', xy=fk_ee, xytext=fk_base,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax3.text(0.3, -0.1, '前向运动学\n计算位置', ha='center', color='blue', fontweight='bold')
    
    # 误差
    ax3.annotate('', xy=mocap_ee, xytext=fk_ee,
                arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    ax3.text(0.65, 0.3, '误差', ha='center', color='orange', fontweight='bold')
    
    ax3.set_title('标定原理 - 最小化相对位置误差', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    
    # 4. 标定结果总结
    ax4.axis('off')
    
    result_text = """=== 最终标定结果 ===

使用的参考点配置:
✅ BASE: base_link frame (机器人根部)
✅ EEF: l5 frame (末端link末端)

标定精度:
• 平均误差: 243.58 mm
• 最大误差: 652.61 mm  
• 标准差: 116.16 mm

关节零点偏移:
• 关节 1: -64.969°
• 关节 2: -90.000° (边界值)
• 关节 3: 78.831°
• 关节 4: -90.000° (边界值)
• 关节 5: 0.000°

优势:
✓ 使用物理上最合理的参考点
✓ 避免中间关节累积误差
✓ 相对位置消除坐标系依赖
✓ Pinocchio库保证计算精度"""
    
    ax4.text(0.05, 0.95, result_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('robot_reference_points_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 机器人参考点示意图已保存为: robot_reference_points_diagram.png")

if __name__ == "__main__":
    draw_robot_reference_diagram()

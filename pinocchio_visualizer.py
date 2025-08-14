#!/usr/bin/env python3
"""
使用Pinocchio原生可视化显示机器人和frame
"""
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import time

def main():
    print("=== PINOCCHIO NATIVE VISUALIZATION ===")
    
    # 加载URDF模型
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm/urdf/ic_arm.urdf"
    pkg_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm/urdf"
    # 加载模型
    # model, _, _  = pin.buildModelsFromUrdf(urdf_path)
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
    print(f"✅ Loaded model: {model.name}")
    print(f"   Joints: {model.nq}")
    print(f"   Frames: {model.nframes}")
    
    # 创建可视化器

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    # viz = MeshcatVisualizer(model)
    
    # 初始化可视化器
    viz.initViewer(open=True)
    viz.loadViewerModel()
    
    print("\n🌐 Meshcat viewer opened in browser!")
    print("   URL: http://127.0.0.1:7000/static/")
    
    # 创建数据结构
    data = model.createData()
    
    # 显示所有frame信息
    print(f"\n=== AVAILABLE FRAMES ===")
    base_frame_id = None
    ee_frame_id = None
    
    for i in range(model.nframes):
        frame = model.frames[i]
        print(f"  {i:2d}: {frame.name}")
        if frame.name == 'base_link':
            base_frame_id = i
        elif frame.name == 'l5':
            ee_frame_id = i
    
    if base_frame_id is None:
        base_frame_id = 1  # 默认
    if ee_frame_id is None:
        ee_frame_id = 11   # 默认
        
    print(f"\n🎯 Using frames:")
    print(f"   BASE: Frame {base_frame_id} ({model.frames[base_frame_id].name})")
    print(f"   EEF:  Frame {ee_frame_id} ({model.frames[ee_frame_id].name})")
    
    # 设置初始关节角度
    q = np.zeros(model.nq)
    
    # 计算前向运动学
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    # 获取frame位置
    base_pos = data.oMf[base_frame_id].translation
    ee_pos = data.oMf[ee_frame_id].translation
    relative_pos = ee_pos - base_pos
    
    print(f"\n📍 Frame positions:")
    print(f"   BASE position: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
    print(f"   EEF position:  [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"   Relative:      [{relative_pos[0]:.3f}, {relative_pos[1]:.3f}, {relative_pos[2]:.3f}]")
    
    # 显示机器人
    viz.display(q)
    
    # 添加frame标记（如果支持）
    try:
        # 尝试添加球体标记
        import meshcat.geometry as g
        
        # BASE标记 (红色球体)
        viz.viewer["base_marker"].set_object(
            g.Sphere(0.02), 
            material=g.MeshLambertMaterial(color=0xff0000, opacity=0.8)
        )
        viz.viewer["base_marker"].set_transform(
            pin.SE3(np.eye(3), base_pos).homogeneous
        )
        
        # END-EFFECTOR标记 (蓝色球体)
        viz.viewer["ee_marker"].set_object(
            g.Sphere(0.02),
            material=g.MeshLambertMaterial(color=0x0000ff, opacity=0.8)
        )
        viz.viewer["ee_marker"].set_transform(
            pin.SE3(np.eye(3), ee_pos).homogeneous
        )
        
        # 连接线 (绿色)
        line_points = np.array([base_pos, ee_pos]).T
        viz.viewer["connection_line"].set_object(
            g.Line(g.PointsGeometry(line_points)),
            material=g.LineBasicMaterial(color=0x00ff00, linewidth=3)
        )
        
        print(f"\n🎨 Added visual markers:")
        print(f"   🔴 Red sphere at BASE position")
        print(f"   🔵 Blue sphere at EEF position") 
        print(f"   🟢 Green line showing relative vector")
        
    except Exception as e:
        print(f"\n⚠️  Could not add markers: {e}")
        print("   But you can still see the robot model!")
    
    # 动画演示不同姿态
    print(f"\n🎬 Starting animation demo...")
    print("   Watch how the frame positions change with joint motion!")
    print("   Press Ctrl+C to stop")
    
    try:
        t = 0
        while True:
            # 生成动态关节角度
            q[0] = 0.5 * np.sin(0.5 * t)      # j1
            q[1] = 0.3 * np.cos(0.3 * t)      # j2  
            q[2] = 0.4 * np.sin(0.7 * t)      # j3
            q[3] = 0.2 * np.cos(0.9 * t)      # j4
            q[4] = 0.6 * np.sin(1.1 * t)      # j5
            
            # 更新运动学
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            
            # 更新显示
            viz.display(q)
            
            # 更新标记位置
            try:
                new_base_pos = data.oMf[base_frame_id].translation
                new_ee_pos = data.oMf[ee_frame_id].translation
                
                viz.viewer["base_marker"].set_transform(
                    pin.SE3(np.eye(3), new_base_pos).homogeneous
                )
                viz.viewer["ee_marker"].set_transform(
                    pin.SE3(np.eye(3), new_ee_pos).homogeneous
                )
                
                # 更新连接线
                line_points = np.array([new_base_pos, new_ee_pos]).T
                viz.viewer["connection_line"].set_object(
                    g.Line(g.PointsGeometry(line_points)),
                    material=g.LineBasicMaterial(color=0x00ff00, linewidth=3)
                )
            except:
                pass
            
            time.sleep(0.05)
            t += 0.05
            
    except KeyboardInterrupt:
        print(f"\n✅ Animation stopped")
        print("   Viewer will remain open - check your browser!")
        print("   Close browser tab to exit completely")
        
        # 保持最终状态
        input("Press Enter to exit...")
        
if __name__ == "__main__":
    main()

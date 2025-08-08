#!/usr/bin/env python3
"""
Visualize base_link and l5 frames on the robot using MuJoCo
"""
import mujoco
import mujoco.viewer
import numpy as np
import os

URDF_PATH = "ic_arm_urdf/urdf/ic1.1.2.urdf"

# Helper to find body id by name
def find_body_id(model, name):
    for i in range(model.nbody):
        if model.body(i).name == name:
            return i
    return None

def main():
    # Load URDF using mujoco
    urdf_abspath = os.path.join(os.path.dirname(__file__), URDF_PATH)
    print(f"Loading URDF: {urdf_abspath}")
    m = mujoco.MjModel.from_xml_path(urdf_abspath)
    d = mujoco.MjData(m)
    print("Bodies in MuJoCo model:")
    for i in range(m.nbody):
        print(f"{i}: {m.body(i).name}")
    # Find body ids for base (world) and l5
    base_id = 0  # In MuJoCo, world is always body 0 (robot base)
    l5_id = find_body_id(m, 'l5')
    if l5_id is None:
        print("Could not find l5 in model bodies!")
        return
    
    print(f"Using base_id: {base_id} (world), l5_id: {l5_id} (l5)")

    # Forward kinematics to update positions
    mujoco.mj_forward(m, d)

    # Get world positions
    base_pos = d.xpos[base_id].copy()
    l5_pos = d.xpos[l5_id].copy()
    print(f"base_link frame world position: {base_pos}")
    print(f"l5 frame world position: {l5_pos}")

    # Simple approach: modify the model directly to add visible markers
    print(f"\n=== ADDING VISUAL MARKERS TO MODEL ===")
    
    # Create a new model with added sphere geometries
    xml_string = f'''
    <mujoco>
        <include file="{urdf_abspath}"/>
        
        <worldbody>
            <!-- BASE marker -->
            <body name="base_marker" pos="{base_pos[0]} {base_pos[1]} {base_pos[2]}">
                <geom name="base_sphere" type="sphere" size="0.02" rgba="1 0 0 1"/>
            </body>
            
            <!-- END-EFFECTOR marker -->
            <body name="ee_marker" pos="{l5_pos[0]} {l5_pos[1]} {l5_pos[2]}">
                <geom name="ee_sphere" type="sphere" size="0.02" rgba="0 0 1 1"/>
            </body>
        </worldbody>
    </mujoco>
    '''
    
    try:
        # Load model with markers
        m_with_markers = mujoco.MjModel.from_xml_string(xml_string)
        d_with_markers = mujoco.MjData(m_with_markers)
        
        print(f"Model with markers loaded successfully!")
        print(f"RED sphere at BASE position: {base_pos}")
        print(f"BLUE sphere at END-EFFECTOR position: {l5_pos}")
        print(f"Relative vector: {l5_pos - base_pos}")
        print(f"Press ESC to exit viewer")
        
        # Launch viewer with marked model
        with mujoco.viewer.launch_passive(m_with_markers, d_with_markers) as viewer:
            while viewer.is_running():
                mujoco.mj_step(m_with_markers, d_with_markers)
                viewer.sync()
                
    except Exception as e:
        print(f"Failed to create model with markers: {e}")
        print(f"Falling back to original model without markers...")
        
        # Fallback: just show the original model
        with mujoco.viewer.launch_passive(m, d) as viewer:
            print(f"\n=== FRAME POSITIONS (no visual markers) ===")
            print(f"BASE (world): {base_pos}")
            print(f"END-EFFECTOR (l5): {l5_pos}")
            print(f"Relative vector: {l5_pos - base_pos}")
            
            while viewer.is_running():
                viewer.sync()

if __name__ == "__main__":
    main()

# IC ARM Control

A unified motor control interface for IC ARM robotic system supporting multiple motor types including Damiao, High Torque, and Servo motors.

## Overview

This project provides a complete workflow for robotic arm control, from basic motor operations to advanced trajectory execution and analysis.

## Installation

```bash
# Install in development mode
pip install -e .
```

## Complete Workflow

### 1. Basic ARM Control
The main control interface is provided by `IC_ARM.py`:
```bash
python -c "from ic_arm_control.control.IC_ARM import ICARM; arm = ICARM()"
```

### 2. Position Monitoring
Monitor real-time arm positions:
```bash
python -m ic_arm_control.tools.position_monitor
```

### 3. Set Zero Positions
Calibrate motor zero positions:
```bash
python -m ic_arm_control.tools.set_zero_position
```

### 4. URDF Generation and Export
Build and export URDF files to the `urdfs/` folder:
```bash
# URDF files are generated and stored in ./urdfs/ directory
```

### 5. MuJoCo Simulation Synchronization
Sync URDF directions with MuJoCo simulation:
```bash
python -m ic_arm_control.tools.mujoco_simulation
```

### 6. URDF Limit Configuration
Set joint limits in the URDF:
```bash
python -m ic_arm_control.tools.urdf_limit_updater
```

### 7. Trajectory Generation
Generate test trajectories:
```bash
python -m ic_arm_control.tools.trajectory_generator
```

### 8. Trajectory Execution
Execute generated trajectories:
```bash
python -m ic_arm_control.tools.trajectory_executor
```

### 9. Minimum GC Analysis
Analyze minimum geometric constraints:
```bash
python -m ic_arm_control.control.utils.minimum_gc
```

### 10. GC Flag Support
The IC ARM now supports geometric constraint (GC) flag for advanced control modes.

## Project Structure

```
ic_arm_control/
├── control/           # Core control modules
│   ├── IC_ARM.py     # Main ARM control interface
│   ├── unified_motor_control.py  # Unified motor interface
│   ├── damiao.py     # Damiao motor control
│   ├── ht_motor.py   # High Torque motor control
│   ├── servo_motor.py # Servo motor control
│   └── utils/        # Utility functions
└── tools/            # Tools and utilities
    ├── position_monitor.py
    ├── set_zero_position.py
    ├── mujoco_simulation.py
    ├── urdf_limit_updater.py
    ├── trajectory_generator.py
    ├── trajectory_executor.py
    └── trajectory_analysis.py
```

## Features

- **Unified Motor Control**: Support for multiple motor types through a single interface
- **Real-time Monitoring**: Position, velocity, and torque monitoring
- **Trajectory Planning**: Advanced trajectory generation and execution
- **MuJoCo Integration**: Seamless simulation support
- **URDF Support**: Complete robot description and configuration
- **Safety Features**: Emergency stop and error handling mechanisms 
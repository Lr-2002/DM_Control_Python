# UMC (Unified Motor Control) 架构关系图

## 整体架构

```mermaid
graph TB
    subgraph "上层应用"
        IC_ARM["IC_ARM<br/>机械臂控制器"]
    end
    
    subgraph "统一电机控制层 (UMC)"
        MotorManager["MotorManager<br/>统一电机管理器"]
        UnifiedMotor["UnifiedMotor<br/>统一电机接口"]
        CANDispatcher["CANFrameDispatcher<br/>CAN帧分发器"]
    end
    
    subgraph "协议层 (Protocols)"
        MotorProtocol["MotorProtocol<br/>抽象协议基类"]
        DamiaoProtocol["DamiaoProtocol<br/>达妙电机协议"]
        HTProtocol["HTProtocol<br/>HT电机协议"]
        ServoProtocol["ServoProtocol<br/>舵机协议"]
    end
    
    subgraph "电机管理层 (Managers)"
        DmMotorManager["DmMotorManager<br/>达妙电机管理器"]
        HTMotorManager["HTMotorManager<br/>HT电机管理器"]
        ServoMotorManager["ServoMotorManager<br/>舵机管理器"]
    end
    
    subgraph "电机实体层 (Motors)"
        DM_Motor["Motor<br/>达妙电机"]
        HTMotor["HTMotor<br/>HT电机"]
        ServoMotor["ServoMotor<br/>舵机"]
    end
    
    subgraph "硬件接口层"
        USBWrapper["USBHardwareWrapper<br/>USB硬件包装器"]
        USBClass["usb_class<br/>底层USB接口"]
    end
    
    subgraph "数据结构"
        MotorInfo["MotorInfo<br/>电机配置信息"]
        MotorFeedback["MotorFeedback<br/>电机反馈数据"]
        DmActData["DmActData<br/>达妙电机数据"]
    end
    
    %% 连接关系
    IC_ARM --> MotorManager
    MotorManager --> UnifiedMotor
    MotorManager --> CANDispatcher
    MotorManager --> DamiaoProtocol
    MotorManager --> HTProtocol
    MotorManager --> ServoProtocol
    
    DamiaoProtocol --> MotorProtocol
    HTProtocol --> MotorProtocol
    ServoProtocol --> MotorProtocol
    
    DamiaoProtocol --> DmMotorManager
    HTProtocol --> HTMotorManager
    ServoProtocol --> ServoMotorManager
    
    DmMotorManager --> DM_Motor
    HTMotorManager --> HTMotor
    ServoMotorManager --> ServoMotor
    
    UnifiedMotor --> MotorProtocol
    CANDispatcher --> USBWrapper
    
    DmMotorManager --> USBWrapper
    HTMotorManager --> USBWrapper
    ServoMotorManager --> USBWrapper
    
    USBWrapper --> USBClass
    
    MotorManager --> MotorInfo
    UnifiedMotor --> MotorFeedback
    DmMotorManager --> DmActData
```

## 详细组件关系

### 1. 核心控制流程

```mermaid
sequenceDiagram
    participant App as IC_ARM
    participant MM as MotorManager
    participant UM as UnifiedMotor
    participant Proto as Protocol
    participant Mgr as Manager
    participant Motor as Motor
    participant USB as USB Hardware
    
    App->>MM: 初始化电机系统
    MM->>Proto: 创建协议实例
    Proto->>Mgr: 关联管理器
    Mgr->>Motor: 创建电机实例
    MM->>UM: 创建统一接口
    
    App->>UM: 发送控制命令
    UM->>Proto: 调用协议接口
    Proto->>Mgr: 转发到管理器
    Mgr->>Motor: 控制具体电机
    Motor->>USB: 发送CAN帧
    
    USB-->>Motor: 接收反馈
    Motor-->>Mgr: 更新状态
    Mgr-->>Proto: 返回反馈
    Proto-->>UM: 统一格式
    UM-->>App: 返回状态
```

### 2. 协议层架构

```mermaid
classDiagram
    class MotorProtocol {
        <<abstract>>
        +usb_hw
        +motors: dict
        +add_motor(motor_info)
        +enable_motor(motor_id)
        +disable_motor(motor_id)
        +set_command(motor_id, pos, vel, kp, kd, tau)
        +send_commands()
        +read_feedback(motor_id)
        +set_zero_position(motor_id)
        +get_limits(motor_id)
    }
    
    class DamiaoProtocol {
        +motor_control: DmMotorManager
        +pending_commands: dict
        +add_motor(motor_info)
        +enable_motor(motor_id)
        +set_command(motor_id, ...)
        +read_feedback(motor_id)
    }
    
    class HTProtocol {
        +ht_manager: HTMotorManager
        +pending_commands: dict
        +add_motor(motor_info)
        +enable_motor(motor_id)
        +set_command(motor_id, ...)
        +send_commands()
    }
    
    class ServoProtocol {
        +servo_manager: ServoMotorManager
        +motor_mapping: dict
        +add_motor(motor_info)
        +enable_motor(motor_id)
        +set_command(motor_id, ...)
    }
    
    MotorProtocol <|-- DamiaoProtocol
    MotorProtocol <|-- HTProtocol
    MotorProtocol <|-- ServoProtocol
```

### 3. 管理器层架构

```mermaid
classDiagram
    class DmMotorManager {
        +usb_hw
        +motors: list
        +addMotor(motor)
        +control_mit(motor, kp, kd, pos, vel, tau)
        +switchControlMode(motor, mode)
        +refresh_motor_status(motor)
        +canframeCallback(frame)
    }
    
    class HTMotorManager {
        +usb_hw
        +motors: dict
        +add_motor(can_id)
        +mit_control(pos_list, vel_list, ...)
        +brake()
        +refresh_motor_status()
    }
    
    class ServoMotorManager {
        +usb_hw
        +servos: dict
        +add_servo(motor_id, can_id, rx_id)
        +enable_all()
        +disable_all()
        +set_positions(positions, velocities)
        +read_all_status()
    }
```

### 4. 数据流向

```mermaid
flowchart LR
    subgraph "输入数据"
        CMD[控制命令]
        POS[位置]
        VEL[速度]
        TAU[力矩]
    end
    
    subgraph "处理层"
        UMC[统一电机控制]
        PROTO[协议转换]
        MGR[管理器处理]
    end
    
    subgraph "输出数据"
        CAN[CAN帧]
        FB[反馈数据]
        STATUS[状态信息]
    end
    
    CMD --> UMC
    POS --> UMC
    VEL --> UMC
    TAU --> UMC
    
    UMC --> PROTO
    PROTO --> MGR
    MGR --> CAN
    
    CAN --> FB
    FB --> STATUS
```

## 关键特性

### 1. 统一接口
- **UnifiedMotor**: 为所有类型电机提供统一的控制接口
- **MotorManager**: 管理多种电机协议，提供统一的管理功能
- **标准化命令**: 所有电机都使用相同的MIT控制命令格式

### 2. 协议抽象
- **MotorProtocol**: 抽象基类定义了所有电机协议必须实现的接口
- **具体协议**: 每种电机类型都有专门的协议实现
- **CAN帧分发**: 统一处理CAN帧的接收和分发

### 3. 模块化设计
- **分层架构**: 应用层 → 统一控制层 → 协议层 → 管理器层 → 电机层
- **松耦合**: 各层之间通过接口交互，便于扩展和维护
- **可扩展**: 新增电机类型只需实现对应的协议和管理器

### 4. 硬件抽象
- **USB包装器**: 提供统一的USB硬件接口
- **兼容性处理**: 自动处理不同平台的USB库兼容性问题
- **错误处理**: 完善的错误处理和恢复机制

#!/usr/bin/env python3
"""
ICARM Position Monitor
持续监控电机位置并使用 rerun 实时显示
"""

from IC_ARM import ICARM

def main():
    """Main function to run position monitoring"""
    print("ICARM Position Monitor")
    print("=" * 50)
    
    try:
        # Initialize ICARM
        arm = ICARM()
        
        # Start continuous position monitoring
        # 10 Hz update rate, no duration limit (infinite)
        arm.monitor_positions_continuous(update_rate=10.0, duration=None)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            arm.close()
        except:
            pass

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple ICARM Position Monitor
只显示控制台输出的实时位置监控
"""

from IC_ARM import ICARM
import time

def main():
    """Main function to run simple position monitoring"""
    print("Simple ICARM Position Monitor")
    print("=" * 50)
    
    try:
        # Initialize ICARM
        arm = ICARM()
        
        print("Starting simple position monitoring...")
        print("Press Ctrl+C to stop")
        
        # Enable all motors first
        arm.enable_all_motors()
        
        start_time = time.time()
        update_rate = 10.0  # 10 Hz
        update_interval = 1.0 / update_rate
        
        try:
            while True:
                current_time = time.time()
                
                # Get positions
                positions = arm.get_positions_only()
                
                # Print to console (compact format)
                pos_str = " | ".join([f"{name}: {positions[name]['deg']:.2f}°" 
                                     for name in positions.keys() if positions[name]['deg'] is not None])
                
                # Add timestamp
                elapsed = current_time - start_time
                print(f"\r[{elapsed:6.1f}s] {pos_str}", end="", flush=True)
                
                # Wait for next update
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping position monitoring...")
        finally:
            # Disable motors for safety
            arm.disable_all_motors()
            print("Position monitoring stopped.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            arm.close()
        except:
            pass

if __name__ == "__main__":
    main()

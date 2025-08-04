#!/usr/bin/env python3
"""
Test rerun functionality with simulated data
"""

import rerun as rr
import time
import math

def test_rerun():
    """Test rerun with simulated motor data"""
    print("Testing rerun with simulated data...")
    
    # Initialize rerun
    rr.init("Test_Motor_Data", spawn=True)
    
    print("Generating simulated motor data for 10 seconds...")
    print("Check if rerun viewer opens and shows the plots")
    
    start_time = time.time()
    
    for i in range(100):  # 10 seconds at 10Hz
        current_time = time.time() - start_time
        
        # Set time
        rr.set_time_seconds("timestamp", current_time)
        
        # Simulate motor positions (sine waves with different frequencies)
        m1_angle = 30 * math.sin(0.5 * current_time)  # Slow oscillation
        m2_angle = 20 * math.sin(1.0 * current_time)  # Medium oscillation
        m3_angle = 15 * math.sin(2.0 * current_time)  # Fast oscillation
        
        # Log to rerun
        rr.log("motors/m1/position_deg", rr.Scalars(m1_angle))
        rr.log("motors/m2/position_deg", rr.Scalars(m2_angle))
        rr.log("motors/m3/position_deg", rr.Scalars(m3_angle))
        
        # Also log to overview
        rr.log("overview/motor_m1_deg", rr.Scalars(m1_angle))
        rr.log("overview/motor_m2_deg", rr.Scalars(m2_angle))
        rr.log("overview/motor_m3_deg", rr.Scalars(m3_angle))
        
        print(f"\r[{current_time:5.1f}s] m1: {m1_angle:6.1f}° | m2: {m2_angle:6.1f}° | m3: {m3_angle:6.1f}°", end="", flush=True)
        
        time.sleep(0.1)  # 10Hz
    
    print("\nTest completed. Check rerun viewer for plots.")

if __name__ == "__main__":
    test_rerun()

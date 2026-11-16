#!/usr/bin/env python3
"""
Debug Time Intervals and Velocity Calculations
"""

import pandas as pd
import math

# System parameters
PULLEY_DIAMETER = 31.0  # mm
PULLEY_RADIUS = PULLEY_DIAMETER / 2  # 15.5 mm

def degrees_to_mm(degrees):
    """Convert servo position from degrees to millimeters"""
    radians = degrees * math.pi / 180
    mm = radians * PULLEY_RADIUS
    return mm

def main():
    # Load data
    flight_path = "/Users/baharakqaderi/field-data-pipeline/flight_analysis_july17/2025_07_17_11_30/flight_data.csv"
    df = pd.read_csv(flight_path)
    
    # Convert time to seconds from start
    if df['_time'].dtype == 'object':
        df['_time'] = pd.to_datetime(df['_time'], format='mixed')
    df['time_seconds'] = (df['_time'] - df['_time'].min()).dt.total_seconds()
    
    # Convert positions from degrees to mm
    df['left_position_mm'] = degrees_to_mm(df['FLIGHT_SEGMENT_left_servo_position'])
    df['right_position_mm'] = degrees_to_mm(df['FLIGHT_SEGMENT_right_servo_position'])
    
    print("🔍 DEBUGGING TIME INTERVALS AND VELOCITIES")
    print("=" * 50)
    
    # Check first 20 data points
    print("\nFirst 20 data points analysis:")
    print("Index | Time(s) | dt(s) | dt(ms) | Left_pos(mm) | dx_left(mm) | Left_vel(mm/s)")
    print("-" * 80)
    
    for i in range(1, min(21, len(df))):
        t1 = df.iloc[i-1]['time_seconds']
        t2 = df.iloc[i]['time_seconds']
        dt = t2 - t1
        
        x1_left = df.iloc[i-1]['left_position_mm']
        x2_left = df.iloc[i]['left_position_mm']
        dx_left = x2_left - x1_left
        
        if dt > 0:
            velocity_left = dx_left / dt
        else:
            velocity_left = 0
        
        print(f"{i:5d} | {t2:7.3f} | {dt:7.4f} | {dt*1000:6.1f} | {x2_left:11.2f} | {dx_left:10.2f} | {velocity_left:12.1f}")
    
    # Statistical analysis
    dt_values = []
    velocity_values = []
    position_changes = []
    
    for i in range(1, len(df)):
        t1 = df.iloc[i-1]['time_seconds']
        t2 = df.iloc[i]['time_seconds']
        dt = t2 - t1
        
        x1_left = df.iloc[i-1]['left_position_mm']
        x2_left = df.iloc[i]['left_position_mm']
        dx_left = x2_left - x1_left
        
        if dt > 0:
            velocity_left = dx_left / dt
            dt_values.append(dt)
            velocity_values.append(abs(velocity_left))
            position_changes.append(abs(dx_left))
    
    dt_series = pd.Series(dt_values)
    vel_series = pd.Series(velocity_values)
    pos_series = pd.Series(position_changes)
    
    print(f"\n📊 STATISTICAL ANALYSIS:")
    print("=" * 30)
    print(f"Time Intervals (dt):")
    print(f"  Mean: {dt_series.mean()*1000:.2f} ms")
    print(f"  Min:  {dt_series.min()*1000:.2f} ms")
    print(f"  Max:  {dt_series.max()*1000:.2f} ms")
    print(f"  Std:  {dt_series.std()*1000:.2f} ms")
    
    print(f"\nPosition Changes (dx):")
    print(f"  Mean: {pos_series.mean():.3f} mm")
    print(f"  Min:  {pos_series.min():.3f} mm")
    print(f"  Max:  {pos_series.max():.3f} mm")
    print(f"  Std:  {pos_series.std():.3f} mm")
    
    print(f"\nVelocities (|v|):")
    print(f"  Mean: {vel_series.mean():.1f} mm/s")
    print(f"  Min:  {vel_series.min():.1f} mm/s")
    print(f"  Max:  {vel_series.max():.1f} mm/s")
    print(f"  Std:  {vel_series.std():.1f} mm/s")
    
    # Find problematic cases
    high_vel_threshold = 10000  # mm/s
    high_vel_cases = [(i, v) for i, v in enumerate(velocity_values) if v > high_vel_threshold]
    
    print(f"\n⚠️  HIGH VELOCITY CASES (>{high_vel_threshold} mm/s):")
    print("=" * 40)
    print(f"Found {len(high_vel_cases)} cases")
    
    if high_vel_cases:
        print("\nFirst 10 high velocity cases:")
        print("Index | dt(ms) | dx(mm) | Velocity(mm/s)")
        print("-" * 35)
        
        for i, (idx, vel) in enumerate(high_vel_cases[:10]):
            dt = dt_values[idx]
            dx = position_changes[idx]
            print(f"{idx:5d} | {dt*1000:6.1f} | {dx:6.2f} | {vel:12.1f}")
    
    # Check for very small time intervals
    small_dt_threshold = 0.001  # 1ms
    small_dt_cases = [dt for dt in dt_values if dt < small_dt_threshold]
    
    print(f"\n⚠️  VERY SMALL TIME INTERVALS (<{small_dt_threshold*1000}ms):")
    print("=" * 45)
    print(f"Found {len(small_dt_cases)} cases")
    
    if small_dt_cases:
        small_dt_series = pd.Series(small_dt_cases)
        print(f"  Smallest: {small_dt_series.min()*1000:.3f} ms")
        print(f"  Mean:     {small_dt_series.mean()*1000:.3f} ms")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import pandas as pd
import os
import numpy as np
from datetime import datetime

# Flight time ranges (corrected)
flights = [
    ("10:29:25", "10:32:59"),
    ("10:58:01", "11:02:16"), 
    ("11:47:46", "12:18:40"),
    ("13:46:41", "13:51:33"),
    ("14:00:25", "14:03:27"),
    ("14:37:58", "14:46:57")
]

# Load the data
df = pd.read_csv('/Users/baharakqaderi/field-data-pipeline/outputs/processed_merged_flight_data_2025-07-16.csv')

# Convert _time to datetime
df['_time'] = pd.to_datetime(df['_time'], format='mixed')

# Base output directory
base_dir = '/Users/baharakqaderi/field-data-pipeline/flight_analysis'
os.makedirs(base_dir, exist_ok=True)

def calculate_metrics(flight_data):
    """Calculate all required metrics for a flight segment"""
    
    # Total force = sum of all force columns
    flight_data['total_force'] = (
        flight_data['Backline_Left_kg'] + 
        flight_data['Backline_Right_kg'] + 
        flight_data['5th_line_kg'] + 
        flight_data['Frontline_kg']
    )
    
    # Back force calculations
    flight_data['back_force_left_right_sum'] = flight_data['Backline_Left_kg'] + flight_data['Backline_Right_kg']
    flight_data['back_force_left_right_diff'] = abs(flight_data['Backline_Left_kg'] - flight_data['Backline_Right_kg'])
    
    # Torque calculations
    flight_data['torque_abs_left'] = abs(flight_data['FLIGHT_SEGMENT_l_torque'].fillna(0))
    flight_data['torque_abs_right'] = abs(flight_data['FLIGHT_SEGMENT_r_torque'].fillna(0))
    flight_data['torque_diff'] = abs(flight_data['FLIGHT_SEGMENT_l_torque'].fillna(0) - flight_data['FLIGHT_SEGMENT_r_torque'].fillna(0))
    
    metrics = {
        'Max total force': flight_data['total_force'].max(),
        'Average back force (mean(mean left,mean right))': (flight_data['Backline_Left_kg'].mean() + flight_data['Backline_Right_kg'].mean()) / 2,
        'Max back force (max(max left,max right))': max(flight_data['Backline_Left_kg'].max(), flight_data['Backline_Right_kg'].max()),
        'Average total back force (mean(left + right))': flight_data['back_force_left_right_sum'].mean(),
        'Max total back force (max(left + right))': flight_data['back_force_left_right_sum'].max(),
        'Average difference back force (mean(abs(left - right)))': flight_data['back_force_left_right_diff'].mean(),
        'Max difference back force (max(abs(left - right)))': flight_data['back_force_left_right_diff'].max(),
        'Average torque (left and right) mean(abs(torque_left),abs(torque_right))': (flight_data['torque_abs_left'].mean() + flight_data['torque_abs_right'].mean()) / 2,
        'Average torque diff mean(abs(torque_left - torque_right))': flight_data['torque_diff'].mean()
    }
    
    return metrics

# Process each flight
for i, (start_time, end_time) in enumerate(flights):
    # Create folder name with date and start hour/minute
    start_dt = datetime.strptime(f"2025-07-16 {start_time}", "%Y-%m-%d %H:%M:%S")
    folder_name = f"2025_07_16_{start_dt.hour:02d}_{start_dt.minute:02d}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    print(f"Processing flight {i+1}: {folder_name}")
    
    # Handle potential time range issues (where end < start, likely next day or typo)
    start_datetime = pd.to_datetime(f"2025-07-16 {start_time}").tz_localize('UTC')
    end_datetime = pd.to_datetime(f"2025-07-16 {end_time}").tz_localize('UTC')
    
    if end_datetime < start_datetime:
        # Assume it's a typo or next day - let's try adding a day to end time
        end_datetime = pd.to_datetime(f"2025-07-17 {end_time}").tz_localize('UTC')
        print(f"  Note: Adjusted end time to next day for flight {i+1}")
    
    # Extract flight data
    flight_data = df[(df['_time'] >= start_datetime) & (df['_time'] <= end_datetime)].copy()
    
    if flight_data.empty:
        print(f"  Warning: No data found for flight {i+1} in time range {start_time}-{end_time}")
        continue
    
    print(f"  Found {len(flight_data)} data points")
    
    # Calculate metrics
    metrics = calculate_metrics(flight_data)
    
    # Save metrics to file
    metrics_file = os.path.join(folder_path, 'flight_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Flight Analysis for {folder_name}\n")
        f.write(f"Time Range: {start_time} - {end_time}\n")
        f.write(f"Data Points: {len(flight_data)}\n")
        f.write("\nMetrics:\n")
        f.write("-" * 50 + "\n")
        
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Save raw flight data
    flight_data_file = os.path.join(folder_path, 'flight_data.csv')
    flight_data.to_csv(flight_data_file, index=False)
    
    print(f"  Saved metrics to {metrics_file}")
    print(f"  Saved data to {flight_data_file}")

print("\nAll flights processed successfully!")
#!/usr/bin/env python3
import pandas as pd
import os
import numpy as np
from datetime import datetime

# Flight time ranges for Catanzaro flights (July 28-29, 2025)
flights = [
    ("2025-07-28", "12:06:04", "12:06:36", "Flight_2"),  # July 28
    ("2025-07-29", "08:41:05", "08:47:18", "Flight_5"),  # July 29
    ("2025-07-29", "09:11:09", "09:18:14", "Flight_6"),  # July 29
    ("2025-07-29", "09:45:01", "09:47:30", "Flight_7")   # July 29
]

# Base output directory
base_dir = '/Users/baharakqaderi/field-data-pipeline/flight_analysis_catanzaro'
os.makedirs(base_dir, exist_ok=True)

def load_data_sources(date):
    """Load all data sources for a given date"""
    data_sources = {}
    
    # Flight Segment data
    fs_file = f'/Users/baharakqaderi/field-data-pipeline/data/INFLUX/FLIGHT_SEGMENT_data_{date}.csv'
    if os.path.exists(fs_file):
        data_sources['flight_segment'] = pd.read_csv(fs_file)
        data_sources['flight_segment']['_time'] = pd.to_datetime(data_sources['flight_segment']['_time'], format='mixed')
        print(f"  Loaded FLIGHT_SEGMENT data: {len(data_sources['flight_segment'])} records")
    
    # Ground Segment data
    gs_file = f'/Users/baharakqaderi/field-data-pipeline/data/INFLUX/GROUND_SEGMENT_data_{date}.csv'
    if os.path.exists(gs_file):
        data_sources['ground_segment'] = pd.read_csv(gs_file)
        data_sources['ground_segment']['_time'] = pd.to_datetime(data_sources['ground_segment']['_time'], format='mixed')
        print(f"  Loaded GROUND_SEGMENT data: {len(data_sources['ground_segment'])} records")
    
    # OPC data
    opc_file = f'/Users/baharakqaderi/field-data-pipeline/data/INFLUX/OPC_data_{date}.csv'
    if os.path.exists(opc_file):
        data_sources['opc'] = pd.read_csv(opc_file)
        data_sources['opc']['_time'] = pd.to_datetime(data_sources['opc']['_time'], format='mixed')
        print(f"  Loaded OPC data: {len(data_sources['opc'])} records")
    
    # METEO data
    meteo_file = f'/Users/baharakqaderi/field-data-pipeline/data/INFLUX/METEO_data_{date}.csv'
    if os.path.exists(meteo_file):
        data_sources['meteo'] = pd.read_csv(meteo_file)
        data_sources['meteo']['_time'] = pd.to_datetime(data_sources['meteo']['_time'], format='mixed')
        print(f"  Loaded METEO data: {len(data_sources['meteo'])} records")
    
    return data_sources

def calculate_catanzaro_metrics(flight_segment_data, opc_data, meteo_data):
    """Calculate metrics specific to Catanzaro flights"""
    
    metrics = {}
    
    # Flight Segment metrics (FS)
    if not flight_segment_data.empty:
        # Force metrics from FS loadcells_force
        fs_force = flight_segment_data['FLIGHT_SEGMENT_loadcells_force'].fillna(0)
        metrics['Average Total Force from FS'] = fs_force.mean()
        metrics['Max Total Force from FS'] = fs_force.max()
        
        # Torque metrics from FS
        torque_left = flight_segment_data['FLIGHT_SEGMENT_l_torque'].fillna(0)
        torque_right = flight_segment_data['FLIGHT_SEGMENT_r_torque'].fillna(0)
        
        metrics['Average torque (left and right) mean(abs(torque_left),abs(torque_right))'] = (abs(torque_left).mean() + abs(torque_right).mean()) / 2
        metrics['Average torque diff mean(abs(torque_left - torque_right))'] = abs(torque_left - torque_right).mean()
    
    # OPC metrics (Ground Station)
    if not opc_data.empty:
        # Force metrics from GS (OPC_DsLoadCells.MeasureFloat_SUM)
        if 'OPC_DsLoadCells.MeasureFloat_SUM' in opc_data.columns:
            gs_force = opc_data['OPC_DsLoadCells.MeasureFloat_SUM'].fillna(0)
            metrics['Average Total Force from GS'] = gs_force.mean()
            metrics['Max Total Force from GS'] = gs_force.max()
        
        # Generator torque metrics (OPC_DsInverters.Torque_ActualValue[2])
        if 'OPC_DsInverters.Torque_ActualValue[2]' in opc_data.columns:
            gen_torque = opc_data['OPC_DsInverters.Torque_ActualValue[2]'].fillna(0)
            metrics['Average torque generator'] = gen_torque.mean()
            metrics['Max torque generator'] = gen_torque.max()
    
    # METEO metrics (Wind Speed)
    if not meteo_data.empty:
        if 'METEO_meteo.speed' in meteo_data.columns:
            wind_speed = meteo_data['METEO_meteo.speed'].fillna(0)
            metrics['Average wind speed'] = wind_speed.mean()
            metrics['Max wind speed'] = wind_speed.max()
            metrics['Min wind speed'] = wind_speed.min()
        
        if 'METEO_meteo.direction' in meteo_data.columns:
            wind_direction = meteo_data['METEO_meteo.direction'].fillna(0)
            # Calculate circular mean for wind direction
            directions_rad = np.radians(wind_direction)
            sin_sum = np.sin(directions_rad).sum()
            cos_sum = np.cos(directions_rad).sum()
            mean_direction = np.degrees(np.arctan2(sin_sum, cos_sum))
            if mean_direction < 0:
                mean_direction += 360
            metrics['Average wind direction'] = mean_direction
    
    return metrics

# Process each flight
for i, (date, start_time, end_time, flight_name) in enumerate(flights):
    print(f"\nProcessing {flight_name} ({date}): {start_time} - {end_time}")
    
    # Create folder name
    start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S")
    folder_name = f"{date.replace('-', '_')}_{start_dt.hour:02d}_{start_dt.minute:02d}_{flight_name}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Load data sources for this date
    data_sources = load_data_sources(date)
    
    # Create timezone-aware datetime objects
    start_datetime = pd.to_datetime(f"{date} {start_time}").tz_localize('UTC')
    end_datetime = pd.to_datetime(f"{date} {end_time}").tz_localize('UTC')
    
    # Extract flight segment data
    flight_segment_data = pd.DataFrame()
    if 'flight_segment' in data_sources:
        flight_segment_data = data_sources['flight_segment'][
            (data_sources['flight_segment']['_time'] >= start_datetime) & 
            (data_sources['flight_segment']['_time'] <= end_datetime)
        ].copy()
        print(f"  Found {len(flight_segment_data)} FLIGHT_SEGMENT data points")
    
    # Extract ground segment data
    ground_segment_data = pd.DataFrame()
    if 'ground_segment' in data_sources:
        ground_segment_data = data_sources['ground_segment'][
            (data_sources['ground_segment']['_time'] >= start_datetime) & 
            (data_sources['ground_segment']['_time'] <= end_datetime)
        ].copy()
        print(f"  Found {len(ground_segment_data)} GROUND_SEGMENT data points")
    
    # Extract OPC data
    opc_data = pd.DataFrame()
    if 'opc' in data_sources:
        opc_data = data_sources['opc'][
            (data_sources['opc']['_time'] >= start_datetime) & 
            (data_sources['opc']['_time'] <= end_datetime)
        ].copy()
        print(f"  Found {len(opc_data)} OPC data points")
    
    # Extract METEO data
    meteo_data = pd.DataFrame()
    if 'meteo' in data_sources:
        meteo_data = data_sources['meteo'][
            (data_sources['meteo']['_time'] >= start_datetime) & 
            (data_sources['meteo']['_time'] <= end_datetime)
        ].copy()
        print(f"  Found {len(meteo_data)} METEO data points")
    
    # Calculate metrics
    metrics = calculate_catanzaro_metrics(flight_segment_data, opc_data, meteo_data)
    
    # Add brake command analysis
    if not ground_segment_data.empty:
        brake_stats = {
            'Brake command - total records': len(ground_segment_data),
            'Brake command - engaged (1) count': (ground_segment_data['GROUND_SEGMENT_brake_command'] == 1).sum(),
            'Brake command - disengaged (0) count': (ground_segment_data['GROUND_SEGMENT_brake_command'] == 0).sum(),
            'Brake command - engaged percentage': (ground_segment_data['GROUND_SEGMENT_brake_command'] == 1).mean() * 100
        }
        metrics.update(brake_stats)
    
    # Check if any data was found
    if flight_segment_data.empty and ground_segment_data.empty and opc_data.empty and meteo_data.empty:
        print(f"  Warning: No data found for {flight_name} in time range {start_time}-{end_time}")
        continue
    
    # Save metrics to file
    metrics_file = os.path.join(folder_path, 'flight_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Flight Analysis for {flight_name}\n")
        f.write(f"Date: {date}\n")
        f.write(f"Time Range: {start_time} - {end_time}\n")
        f.write(f"Data Points - FS: {len(flight_segment_data)}, GS: {len(ground_segment_data)}, OPC: {len(opc_data)}, METEO: {len(meteo_data)}\n")
        f.write("\nMetrics:\n")
        f.write("-" * 50 + "\n")
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
    
    # Save raw data files
    if not flight_segment_data.empty:
        fs_file = os.path.join(folder_path, 'flight_segment_data.csv')
        flight_segment_data.to_csv(fs_file, index=False)
        print(f"  Saved FLIGHT_SEGMENT data to {fs_file}")
    
    if not ground_segment_data.empty:
        gs_file = os.path.join(folder_path, 'ground_segment_data.csv')
        ground_segment_data.to_csv(gs_file, index=False)
        print(f"  Saved GROUND_SEGMENT data to {gs_file}")
    
    if not opc_data.empty:
        opc_file = os.path.join(folder_path, 'opc_data.csv')
        opc_data.to_csv(opc_file, index=False)
        print(f"  Saved OPC data to {opc_file}")
    
    if not meteo_data.empty:
        meteo_file = os.path.join(folder_path, 'meteo_data.csv')
        meteo_data.to_csv(meteo_file, index=False)
        print(f"  Saved METEO data to {meteo_file}")
    
    print(f"  Saved metrics to {metrics_file}")

print(f"\nAll {len(flights)} Catanzaro flights processed successfully!")
print(f"Results saved in: {base_dir}")
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

def calculate_flight_duration(start_time, end_time):
    """Calculate flight duration in minutes and seconds"""
    start_dt = datetime.strptime(start_time, "%H:%M:%S")
    end_dt = datetime.strptime(end_time, "%H:%M:%S")
    duration = end_dt - start_dt
    
    total_seconds = duration.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    return f"{minutes}m {seconds}s", total_seconds

def calculate_catanzaro_metrics(flight_segment_data, opc_data, meteo_data, ground_segment_data, start_time, end_time):
    """Calculate metrics specific to Catanzaro flights including enhanced power metrics"""
    
    metrics = {}
    pi = np.pi
    
    # Calculate flight duration and brake percentage
    flight_duration_str, flight_duration_seconds = calculate_flight_duration(start_time, end_time)
    metrics['Flight Duration'] = flight_duration_str
    metrics['Flight Duration (seconds)'] = flight_duration_seconds
    
    # POD Brake status
    if not ground_segment_data.empty and 'GROUND_SEGMENT_brake_command' in ground_segment_data.columns:
        brake_engaged_percentage = (ground_segment_data['GROUND_SEGMENT_brake_command'] == 1).mean() * 100
        metrics['POD Brake engaged percentage'] = brake_engaged_percentage
    
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
    
    # Enhanced OPC metrics (Ground Station) with new power calculations
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
        
        # Enhanced Power Metrics Calculation
        # Field mapping from your requirements:
        required_fields = {
            'RPMd': 'OPC_DsEncoder.outTamburo_SpeedRPM',
            'Td': 'OPC_DsInverters.Torque_ActualValue[2]', 
            'Vi': 'OPC_DsInverters.Velocity_ActualValue[2]',
            'Pinv': 'OPC_DsInverters.Power[2]',
            'Pbatt': 'OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_PWR_SCALED_CALC',
            'Ftot': 'OPC_DsLoadCells.MeasureFloat_SUM'
        }
        
        # Extract data for power calculations
        data_dict = {}
        for var_name, field_name in required_fields.items():
            if field_name in opc_data.columns:
                data_dict[var_name] = opc_data[field_name].fillna(0)
                # Note: Individual field statistics removed per user request
            else:
                print(f"    Warning: Field {field_name} not found in OPC data")
                data_dict[var_name] = pd.Series([0] * len(opc_data))
        
        # Calculate enhanced power metrics if we have data
        if len(data_dict['RPMd']) > 0:
            # PoMecD = RPMd/60 * pi * 0.491 * Ftot * 9.81
            PoMecD = (- data_dict['RPMd'] / 60 * pi * 0.491 * data_dict['Ftot'] * 9.81) / 1000  
            
            # PoMecGen = Vi / 60 * 2 * pi * Td  
            PoMecGen = (- data_dict['Vi'] / 60 * 2 * pi * data_dict['Td']) * 6 / 1000 
            
            # PoBatt = OPC_CONV_MEAS_FB_LS_PWR_SCALED_CALC (already extracted)
            PoBatt = data_dict['Pbatt']
            
            # Add calculated metrics to OPC data
            opc_data['PoMecD'] = PoMecD
            opc_data['PoMecGen'] = PoMecGen  
            opc_data['PoBatt'] = PoBatt
            
            # Calculate statistics for each power metric
            for name, series in [('PoMecD', PoMecD), ('PoMecGen', PoMecGen), ('PoBatt', PoBatt)]:
                clean_series = series[series != 0]
                if len(clean_series) > 0:
                    metrics[f'{name}_Mean'] = clean_series.mean()
                    metrics[f'{name}_Max'] = clean_series.max()
                    metrics[f'{name}_Min'] = clean_series.min()
                else:
                    metrics[f'{name}_Mean'] = 0
                    metrics[f'{name}_Max'] = 0
                    metrics[f'{name}_Min'] = 0
    
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
    
    return metrics, opc_data

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
    
    # Calculate enhanced metrics including new power calculations
    metrics, enhanced_opc_data = calculate_catanzaro_metrics(flight_segment_data, opc_data, meteo_data, ground_segment_data, start_time, end_time)
    
    # Add traditional brake command analysis for backward compatibility
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
        
        # Save enhanced OPC data with calculated power metrics columns if they exist
        if 'PoMecD' in enhanced_opc_data.columns:
            enhanced_opc_file = os.path.join(folder_path, 'opc_data_enhanced.csv')
            enhanced_opc_data.to_csv(enhanced_opc_file, index=False)
            print(f"  Saved enhanced OPC data with calculated metrics to {enhanced_opc_file}")
    
    if not meteo_data.empty:
        meteo_file = os.path.join(folder_path, 'meteo_data.csv')
        meteo_data.to_csv(meteo_file, index=False)
        print(f"  Saved METEO data to {meteo_file}")
    
    print(f"  Saved metrics to {metrics_file}")

print(f"\nAll {len(flights)} Catanzaro flights processed successfully!")
print(f"Results saved in: {base_dir}")
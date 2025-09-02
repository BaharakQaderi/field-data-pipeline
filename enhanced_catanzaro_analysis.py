#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Flight time ranges for Catanzaro flights (July 28-29, 2025)
flights = [
    ("2025-07-28", "12:06:04", "12:06:36", "Flight_2"),  # July 28
    ("2025-07-29", "08:41:05", "08:47:18", "Flight_5"),  # July 29
    ("2025-07-29", "09:11:09", "09:18:14", "Flight_6"),  # July 29
    ("2025-07-29", "09:45:01", "09:47:30", "Flight_7")   # July 29
]

# Base directories
base_dir = '/Users/baharakqaderi/field-data-pipeline/flight_analysis_catanzaro'

def calculate_flight_duration(start_time, end_time):
    """Calculate flight duration in minutes and seconds"""
    start_dt = datetime.strptime(start_time, "%H:%M:%S")
    end_dt = datetime.strptime(end_time, "%H:%M:%S")
    duration = end_dt - start_dt
    
    total_seconds = duration.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    return f"{minutes}m {seconds}s"

def calculate_enhanced_metrics(opc_data):
    """Calculate the new power metrics: PoMecD, PoMecGen, PoBatt"""
    
    metrics = {}
    pi = np.pi
    
    # Check if all required fields exist
    required_fields = {
        'RPMd': 'OPC_DsEncoder.outTamburo_SpeedRPM',
        'Td': 'OPC_DsInverters.Torque_ActualValue[2]', 
        'Vi': 'OPC_DsInverters.Velocity_ActualValue[2]',
        'Pinv': 'OPC_DsInverters.Power[2]',
        'Pbatt': 'OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_PWR_SCALED_CALC',
        'Ftot': 'OPC_DsLoadCells.MeasureFloat_SUM'
    }
    
    # Extract data and handle missing fields
    data_dict = {}
    for var_name, field_name in required_fields.items():
        if field_name in opc_data.columns:
            data_dict[var_name] = opc_data[field_name].fillna(0)
        else:
            print(f"  Warning: Field {field_name} not found in OPC data")
            data_dict[var_name] = pd.Series([0] * len(opc_data))
    
    # Calculate power metrics if we have data
    if len(data_dict['RPMd']) > 0:
        
        # PoMecD = RPMd/60 * pi * 0.491 * Ftot * 9.81
        PoMecD = data_dict['RPMd'] / 60 * pi * 0.491 * data_dict['Ftot'] * 9.81
        
        # PoMecGen = Vi / 60 * 2 * pi * Td  
        PoMecGen = data_dict['Vi'] / 60 * 2 * pi * data_dict['Td']
        
        # PoBatt = OPC_CONV_MEAS_FB_LS_PWR_SCALED_CALC (already extracted)
        PoBatt = data_dict['Pbatt']
        
        # Calculate statistics for each metric
        for name, series in [('PoMecD', PoMecD), ('PoMecGen', PoMecGen), ('PoBatt', PoBatt)]:
            # Filter out zero values for better statistics
            clean_series = series[series != 0]
            if len(clean_series) > 0:
                metrics[f'{name}_Mean'] = clean_series.mean()
                metrics[f'{name}_Max'] = clean_series.max()
                metrics[f'{name}_Min'] = clean_series.min()
            else:
                metrics[f'{name}_Mean'] = 0
                metrics[f'{name}_Max'] = 0
                metrics[f'{name}_Min'] = 0
        
        # Add the calculated series to the dataframe
        opc_data = opc_data.copy()
        opc_data['PoMecD'] = PoMecD
        opc_data['PoMecGen'] = PoMecGen  
        opc_data['PoBatt'] = PoBatt
        
        # Also store individual field statistics
        for var_name, series in data_dict.items():
            clean_series = series[series != 0]
            if len(clean_series) > 0:
                metrics[f'{var_name}_Mean'] = clean_series.mean()
                metrics[f'{var_name}_Max'] = clean_series.max()
                metrics[f'{var_name}_Min'] = clean_series.min()
    
    return opc_data, metrics

def create_enhanced_catanzaro_visualizations(folder_name, date, start_time, end_time, flight_name):
    """Create comprehensive visualizations for a Catanzaro flight with enhanced titles and metrics"""
    
    folder_path = os.path.join(base_dir, folder_name)
    
    # Calculate flight duration
    flight_duration = calculate_flight_duration(start_time, end_time)
    
    # Load available data files
    fs_data = pd.DataFrame()
    opc_data = pd.DataFrame()
    gs_data = pd.DataFrame()
    meteo_data = pd.DataFrame()
    
    fs_file = os.path.join(folder_path, 'flight_segment_data.csv')
    if os.path.exists(fs_file):
        fs_data = pd.read_csv(fs_file)
        fs_data['_time'] = pd.to_datetime(fs_data['_time'], format='mixed')
    
    opc_file = os.path.join(folder_path, 'opc_data.csv')
    if os.path.exists(opc_file):
        opc_data = pd.read_csv(opc_file)
        opc_data['_time'] = pd.to_datetime(opc_data['_time'], format='mixed')
    
    gs_file = os.path.join(folder_path, 'ground_segment_data.csv')
    if os.path.exists(gs_file):
        gs_data = pd.read_csv(gs_file)
        gs_data['_time'] = pd.to_datetime(gs_data['_time'], format='mixed')
    
    meteo_file = os.path.join(folder_path, 'meteo_data.csv')
    if os.path.exists(meteo_file):
        meteo_data = pd.read_csv(meteo_file)
        meteo_data['_time'] = pd.to_datetime(meteo_data['_time'], format='mixed')
    
    # Calculate brake status
    brake_engaged_percentage = 0.0
    if not gs_data.empty and 'GROUND_SEGMENT_brake_command' in gs_data.columns:
        brake_engaged_percentage = (gs_data['GROUND_SEGMENT_brake_command'] == 1).mean() * 100
    
    # Process OPC data and calculate enhanced metrics
    enhanced_metrics = {}
    if not opc_data.empty:
        opc_data, enhanced_metrics = calculate_enhanced_metrics(opc_data)
    
    # Create enhanced title with duration and brake status
    enhanced_title = (f'Catanzaro Flight Analysis: {flight_name}\n'
                     f'Date: {date} | Time: {start_time} - {end_time} | Duration: {flight_duration}\n'
                     f'POD Brake Status: {brake_engaged_percentage:.1f}% engaged')
    
    # Create time-based plots with enhanced metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(enhanced_title, fontsize=16, fontweight='bold')
    
    # 1. Force Comparison (FS vs GS) 
    ax1 = axes[0, 0]
    if not fs_data.empty and 'FLIGHT_SEGMENT_loadcells_force' in fs_data.columns:
        fs_force = fs_data['FLIGHT_SEGMENT_loadcells_force'].fillna(0)
        ax1.plot(fs_data['_time'], fs_force, label='FS Force (Flight Segment)', color='blue', linewidth=2, alpha=0.8)
    
    if not opc_data.empty and 'OPC_DsLoadCells.MeasureFloat_SUM' in opc_data.columns:
        gs_force = opc_data['OPC_DsLoadCells.MeasureFloat_SUM'].fillna(0)
        ax1.plot(opc_data['_time'], gs_force, label='GS Force (Ground Station)', color='red', linewidth=2, alpha=0.8)
    
    ax1.set_title('Force Measurements: Flight Segment vs Ground Station')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Force')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Enhanced Power Metrics (PoMecD, PoMecGen, PoBatt)
    ax2 = axes[0, 1]
    if not opc_data.empty and 'PoMecD' in opc_data.columns:
        # Plot the calculated power metrics
        if 'PoMecD' in opc_data.columns:
            pomec_d = opc_data['PoMecD'].fillna(0)
            pomec_d_clean = pomec_d[pomec_d != 0]
            if len(pomec_d_clean) > 0:
                ax2.plot(opc_data['_time'], pomec_d, label=f'PoMecD (Mean: {pomec_d_clean.mean():.1f})', color='purple', alpha=0.8, linewidth=2)
        
        if 'PoMecGen' in opc_data.columns:
            pomec_gen = opc_data['PoMecGen'].fillna(0)
            pomec_gen_clean = pomec_gen[pomec_gen != 0]  
            if len(pomec_gen_clean) > 0:
                ax2.plot(opc_data['_time'], pomec_gen, label=f'PoMecGen (Mean: {pomec_gen_clean.mean():.1f})', color='orange', alpha=0.8, linewidth=2)
        
        if 'PoBatt' in opc_data.columns:
            po_batt = opc_data['PoBatt'].fillna(0)
            po_batt_clean = po_batt[po_batt != 0]
            if len(po_batt_clean) > 0:
                ax2.plot(opc_data['_time'], po_batt, label=f'PoBatt (Mean: {po_batt_clean.mean():.1f})', color='green', alpha=0.8, linewidth=2)
    else:
        ax2.text(0.5, 0.5, 'No Enhanced Power\\nMetrics Available', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    ax2.set_title('Enhanced Power Metrics: PoMecD, PoMecGen, PoBatt')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Power')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Generator Torque and RPM
    ax3 = axes[1, 0]
    if not opc_data.empty:
        # Plot generator torque
        if 'OPC_DsInverters.Torque_ActualValue[2]' in opc_data.columns:
            gen_torque = opc_data['OPC_DsInverters.Torque_ActualValue[2]'].fillna(0)
            ax3.plot(opc_data['_time'], gen_torque, color='purple', linewidth=2, 
                    label=f'Generator Torque (Max: {gen_torque.max():.1f}, Avg: {gen_torque.mean():.1f})')
        
        # Add secondary axis for RPM
        if 'OPC_DsEncoder.outTamburo_SpeedRPM' in opc_data.columns:
            ax3_twin = ax3.twinx()
            rpm = opc_data['OPC_DsEncoder.outTamburo_SpeedRPM'].fillna(0)
            ax3_twin.plot(opc_data['_time'], rpm, color='red', linewidth=1.5, alpha=0.7,
                         label=f'RPM (Max: {rpm.max():.1f})')
            ax3_twin.set_ylabel('RPM', color='red')
            ax3_twin.tick_params(axis='y', labelcolor='red')
    else:
        ax3.text(0.5, 0.5, 'No Generator/RPM\\nData Available', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    ax3.set_title('Generator Torque and Drum RPM')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Generator Torque', color='purple')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax3.tick_params(axis='x', rotation=45)
    ax3.tick_params(axis='y', labelcolor='purple')
    
    # 4. Wind Speed and Direction
    ax4 = axes[1, 1]
    if not meteo_data.empty:
        # Primary y-axis for wind speed
        if 'METEO_meteo.speed' in meteo_data.columns:
            wind_speed = meteo_data['METEO_meteo.speed'].fillna(0)
            line1 = ax4.plot(meteo_data['_time'], wind_speed, color='green', linewidth=2, 
                            label=f'Wind Speed (Avg: {wind_speed.mean():.1f}, Max: {wind_speed.max():.1f})')
            ax4.set_ylabel('Wind Speed (m/s)', color='green')
            ax4.tick_params(axis='y', labelcolor='green')
        
        # Secondary y-axis for wind direction
        if 'METEO_meteo.direction' in meteo_data.columns:
            ax4_twin = ax4.twinx()
            wind_direction = meteo_data['METEO_meteo.direction'].fillna(0)
            line2 = ax4_twin.plot(meteo_data['_time'], wind_direction, color='brown', linewidth=1.5, alpha=0.7,
                                 label='Wind Direction')
            ax4_twin.set_ylabel('Wind Direction (°)', color='brown')
            ax4_twin.tick_params(axis='y', labelcolor='brown')
            ax4_twin.set_ylim(0, 360)
        
        ax4.set_title('Wind Conditions')
    else:
        ax4.text(0.5, 0.5, 'No METEO Data Available', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Wind Conditions')
    
    ax4.set_xlabel('Time')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'enhanced_flight_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save enhanced OPC data with calculated metrics
    if not opc_data.empty and 'PoMecD' in opc_data.columns:
        enhanced_opc_file = os.path.join(folder_path, 'opc_data_enhanced.csv')
        opc_data.to_csv(enhanced_opc_file, index=False)
        print(f"  Saved enhanced OPC data to {enhanced_opc_file}")
    
    # Save enhanced metrics
    if enhanced_metrics:
        enhanced_metrics_file = os.path.join(folder_path, 'enhanced_flight_metrics.txt')
        with open(enhanced_metrics_file, 'w') as f:
            f.write(f"Enhanced Flight Analysis for {flight_name}\n")
            f.write(f"Date: {date}\n")
            f.write(f"Time Range: {start_time} - {end_time}\n")
            f.write(f"Flight Duration: {flight_duration}\n")
            f.write(f"POD Brake Engaged: {brake_engaged_percentage:.2f}%\n")
            f.write(f"Data Points - OPC: {len(opc_data)}\n")
            f.write("\nEnhanced Power Metrics:\n")
            f.write("-" * 50 + "\n")
            
            for metric, value in enhanced_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {value}\n")
        print(f"  Saved enhanced metrics to {enhanced_metrics_file}")
    
    print(f"  Created enhanced visualizations for {flight_name}")

# Process each flight and create enhanced visualizations
print("Creating enhanced visualizations and metrics for Catanzaro flights...")

for i, (date, start_time, end_time, flight_name) in enumerate(flights):
    # Focus on flights 5, 6, 7 as requested
    if flight_name not in ['Flight_5', 'Flight_6', 'Flight_7']:
        continue
        
    # Create folder name
    start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S")
    folder_name = f"{date.replace('-', '_')}_{start_dt.hour:02d}_{start_dt.minute:02d}_{flight_name}"
    
    print(f"\nProcessing {flight_name}: {folder_name}")
    
    # Check if folder exists
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.exists(folder_path):
        create_enhanced_catanzaro_visualizations(folder_name, date, start_time, end_time, flight_name)
    else:
        print(f"  Warning: Folder not found for {flight_name}")

print(f"\nEnhanced analysis complete!")
print(f"Results saved in: {base_dir}")
print("\nNew files created in each flight folder:")
print("  - enhanced_flight_timeseries.png (enhanced visualizations with duration and brake status)")
print("  - enhanced_flight_metrics.txt (calculated PoMecD, PoMecGen, PoBatt metrics)")
print("  - opc_data_enhanced.csv (OPC data with new calculated columns)")
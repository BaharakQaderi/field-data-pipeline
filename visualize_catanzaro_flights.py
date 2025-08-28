#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
from datetime import datetime
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

def create_catanzaro_visualizations(folder_name, date, start_time, end_time, flight_name):
    """Create comprehensive visualizations for a Catanzaro flight"""
    
    folder_path = os.path.join(base_dir, folder_name)
    
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
    
    # Create time-based plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Catanzaro Flight Analysis: {flight_name}\nDate: {date} | Time: {start_time} - {end_time}', fontsize=16, fontweight='bold')
    
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
    
    # 2. Flight Control Torques (FS Left/Right only)
    ax2 = axes[0, 1]
    if not fs_data.empty:
        if 'FLIGHT_SEGMENT_l_torque' in fs_data.columns:
            torque_left = fs_data['FLIGHT_SEGMENT_l_torque'].fillna(0)
            ax2.plot(fs_data['_time'], torque_left, label='Left Control Torque', color='blue', alpha=0.8, linewidth=2)
        
        if 'FLIGHT_SEGMENT_r_torque' in fs_data.columns:
            torque_right = fs_data['FLIGHT_SEGMENT_r_torque'].fillna(0)
            ax2.plot(fs_data['_time'], torque_right, label='Right Control Torque', color='red', alpha=0.8, linewidth=2)
        
        # Add torque difference
        if 'FLIGHT_SEGMENT_l_torque' in fs_data.columns and 'FLIGHT_SEGMENT_r_torque' in fs_data.columns:
            torque_diff = abs(torque_left - torque_right)
            ax2.plot(fs_data['_time'], torque_diff, label='|Left - Right|', color='green', linestyle='--', linewidth=1.5)
    else:
        ax2.text(0.5, 0.5, 'No Flight Control\\nTorque Data', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    ax2.set_title('Flight Control Torques (Small Control Motors)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Control Torque')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Generator Torque (Power Generation System)
    ax3 = axes[1, 0]
    if not opc_data.empty and 'OPC_DsInverters.Torque_ActualValue[2]' in opc_data.columns:
        gen_torque = opc_data['OPC_DsInverters.Torque_ActualValue[2]'].fillna(0)
        ax3.plot(opc_data['_time'], gen_torque, color='purple', linewidth=2, 
                label=f'Generator (Max: {gen_torque.max():.1f}, Avg: {gen_torque.mean():.1f})')
        ax3.axhline(y=gen_torque.mean(), color='purple', linestyle='--', alpha=0.7, label=f'Mean: {gen_torque.mean():.1f}')
        ax3.axhline(y=gen_torque.max(), color='red', linestyle=':', alpha=0.7, label=f'Max: {gen_torque.max():.1f}')
    else:
        ax3.text(0.5, 0.5, 'No Generator\\nTorque Data', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    ax3.set_title('Generator Torque (Large Ground Motor)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Generator Torque')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax3.tick_params(axis='x', rotation=45)
    
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
    plt.savefig(os.path.join(folder_path, 'flight_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Catanzaro Flight Distributions: {flight_name} ({date})', fontsize=16, fontweight='bold')
    
    # Force distributions comparison
    ax1 = axes[0, 0]
    if not fs_data.empty and 'FLIGHT_SEGMENT_loadcells_force' in fs_data.columns:
        fs_force = fs_data['FLIGHT_SEGMENT_loadcells_force'].fillna(0)
        if len(fs_force) > 0:
            ax1.hist(fs_force, bins=30, alpha=0.6, label='FS Force', color='blue', 
                    density=True, weights=np.ones(len(fs_force)) / len(fs_force) * 100)
    
    if not opc_data.empty and 'OPC_DsLoadCells.MeasureFloat_SUM' in opc_data.columns:
        gs_force = opc_data['OPC_DsLoadCells.MeasureFloat_SUM'].fillna(0)
        if len(gs_force) > 0:
            ax1.hist(gs_force, bins=30, alpha=0.6, label='GS Force', color='red',
                    density=True, weights=np.ones(len(gs_force)) / len(gs_force) * 100)
    
    ax1.set_title('Force Distribution: FS vs GS')
    ax1.set_xlabel('Force')
    ax1.set_ylabel('Probability (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # FS Control Torque distribution (Small Control Motors)
    ax2 = axes[0, 1]
    if not fs_data.empty:
        if 'FLIGHT_SEGMENT_l_torque' in fs_data.columns:
            torque_left = fs_data['FLIGHT_SEGMENT_l_torque'].fillna(0)
            torque_left_clean = torque_left[torque_left != 0]
            if len(torque_left_clean) > 0:
                ax2.hist(torque_left_clean, bins=20, alpha=0.6, label='Left Control Torque', color='blue',
                        density=True, weights=np.ones(len(torque_left_clean)) / len(torque_left_clean) * 100)
        
        if 'FLIGHT_SEGMENT_r_torque' in fs_data.columns:
            torque_right = fs_data['FLIGHT_SEGMENT_r_torque'].fillna(0)
            torque_right_clean = torque_right[torque_right != 0]
            if len(torque_right_clean) > 0:
                ax2.hist(torque_right_clean, bins=20, alpha=0.6, label='Right Control Torque', color='red',
                        density=True, weights=np.ones(len(torque_right_clean)) / len(torque_right_clean) * 100)
    else:
        ax2.text(0.5, 0.5, 'No Flight Control\\nTorque Data', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    ax2.set_title('Flight Control Torque Distribution (Small Control Motors)')
    ax2.set_xlabel('Control Torque')
    ax2.set_ylabel('Probability (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Generator torque distribution (Large Ground Motor)
    ax3 = axes[1, 0]
    if not opc_data.empty and 'OPC_DsInverters.Torque_ActualValue[2]' in opc_data.columns:
        gen_torque = opc_data['OPC_DsInverters.Torque_ActualValue[2]'].fillna(0)
        gen_torque_clean = gen_torque[gen_torque != 0]
        if len(gen_torque_clean) > 0:
            ax3.hist(gen_torque_clean, bins=30, alpha=0.7, color='purple', edgecolor='black',
                    density=True, weights=np.ones(len(gen_torque_clean)) / len(gen_torque_clean) * 100)
            ax3.axvline(gen_torque_clean.mean(), color='darkblue', linestyle='--', linewidth=2, 
                       label=f'Mean: {gen_torque_clean.mean():.2f}')
    else:
        ax3.text(0.5, 0.5, 'No Generator\\nTorque Data', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    ax3.set_title('Generator Torque Distribution (Large Ground Motor)')
    ax3.set_xlabel('Generator Torque')
    ax3.set_ylabel('Probability (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Wind speed and direction distribution
    ax4 = axes[1, 1]
    if not meteo_data.empty:
        if 'METEO_meteo.speed' in meteo_data.columns:
            wind_speed = meteo_data['METEO_meteo.speed'].fillna(0)
            wind_speed_clean = wind_speed[wind_speed > 0]
            
            if len(wind_speed_clean) > 0:
                ax4.hist(wind_speed_clean, bins=20, alpha=0.7, color='green', edgecolor='black',
                        density=True, weights=np.ones(len(wind_speed_clean)) / len(wind_speed_clean) * 100)
                ax4.axvline(wind_speed_clean.mean(), color='darkgreen', linestyle='--', linewidth=2, 
                           label=f'Mean: {wind_speed_clean.mean():.2f} m/s')
                ax4.axvline(wind_speed_clean.max(), color='red', linestyle='-', linewidth=2,
                           label=f'Max: {wind_speed_clean.max():.2f} m/s')
                
                ax4.set_title('Wind Speed Distribution')
                ax4.set_xlabel('Wind Speed (m/s)')
                ax4.set_ylabel('Probability (%)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No Wind Data Available', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Wind Speed Distribution')
        else:
            ax4.text(0.5, 0.5, 'Wind Speed Field\\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Wind Speed Distribution')
    else:
        ax4.text(0.5, 0.5, 'No METEO Data Available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Wind Speed Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'flight_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Created visualizations for {flight_name}")

# Process each flight and create visualizations
for i, (date, start_time, end_time, flight_name) in enumerate(flights):
    # Create folder name
    start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S")
    folder_name = f"{date.replace('-', '_')}_{start_dt.hour:02d}_{start_dt.minute:02d}_{flight_name}"
    
    print(f"Creating visualizations for {flight_name}: {folder_name}")
    
    # Check if folder exists
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.exists(folder_path):
        create_catanzaro_visualizations(folder_name, date, start_time, end_time, flight_name)
    else:
        print(f"  Warning: Folder not found for {flight_name}")

print(f"\nAll {len(flights)} Catanzaro flight visualizations created successfully!")
print(f"Results saved in: {base_dir}")
print("\nEach flight folder now contains:")
print("  - flight_metrics.txt (calculated metrics)")
print("  - flight_segment_data.csv / opc_data.csv / ground_segment_data.csv / meteo_data.csv (raw data)")
print("  - flight_timeseries.png (time-series plots with wind conditions)")
print("  - flight_distributions.png (distribution plots with wind speed)")
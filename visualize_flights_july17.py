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

# Flight time ranges for July 17, 2025
flights = [
    ("10:15:08", "10:23:42"),
    ("10:37:37", "10:47:16"),
    ("10:54:17", "11:15:08"),
    ("11:30:35", "11:50:12"),
    ("12:34:24", "12:42:04"),
    ("12:45:20", "12:49:25"),
    ("12:51:41", "12:55:51"),
    ("12:59:20", "13:02:56"),
    ("13:08:09", "13:10:47"),
    ("13:15:52", "13:20:25"),
    ("13:24:51", "13:26:14"),
    ("13:31:12", "13:32:42"),
    ("13:32:55", "13:33:53"),
    ("13:35:45", "13:38:03"),
    ("13:39:23", "13:46:50")
]

# Base directories
base_dir = '/Users/baharakqaderi/field-data-pipeline/flight_analysis_july17'

def create_flight_visualizations(folder_name, flight_data, start_time, end_time):
    """Create comprehensive visualizations for a flight"""
    
    folder_path = os.path.join(base_dir, folder_name)
    
    # Prepare data for plotting
    flight_data['total_force'] = (
        flight_data['Backline_Left_kg'] + 
        flight_data['Backline_Right_kg'] + 
        flight_data['5th_line_kg'] + 
        flight_data['Frontline_kg']
    )
    
    flight_data['back_force_sum'] = flight_data['Backline_Left_kg'] + flight_data['Backline_Right_kg']
    flight_data['back_force_diff'] = abs(flight_data['Backline_Left_kg'] - flight_data['Backline_Right_kg'])
    
    # Create time-based plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Flight Analysis: {folder_name}\nDate: July 17, 2025 | Time: {start_time} - {end_time}', fontsize=16, fontweight='bold')
    
    # 1. Individual Force Components Over Time
    ax1 = axes[0, 0]
    ax1.plot(flight_data['_time'], flight_data['Backline_Left_kg'], label='Backline Left', linewidth=1)
    ax1.plot(flight_data['_time'], flight_data['Backline_Right_kg'], label='Backline Right', linewidth=1)
    ax1.plot(flight_data['_time'], flight_data['5th_line_kg'], label='5th Line', linewidth=1)
    ax1.plot(flight_data['_time'], flight_data['Frontline_kg'], label='Frontline', linewidth=1)
    ax1.set_title('Individual Force Components')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Force (kg)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Total Force Over Time
    ax2 = axes[0, 1]
    ax2.plot(flight_data['_time'], flight_data['total_force'], color='red', linewidth=2, label=f'Max: {flight_data["total_force"].max():.2f} kg')
    ax2.axhline(y=flight_data['total_force'].mean(), color='orange', linestyle='--', alpha=0.7, label=f'Mean: {flight_data["total_force"].mean():.2f} kg')
    ax2.set_title('Total Force Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Total Force (kg)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Back Force Analysis
    ax3 = axes[1, 0]
    ax3.plot(flight_data['_time'], flight_data['back_force_sum'], color='blue', linewidth=2, label='Left + Right')
    ax3.plot(flight_data['_time'], flight_data['back_force_diff'], color='purple', linewidth=1, label='|Left - Right|')
    ax3.axhline(y=flight_data['back_force_sum'].mean(), color='blue', linestyle='--', alpha=0.7)
    ax3.axhline(y=flight_data['back_force_diff'].mean(), color='purple', linestyle='--', alpha=0.7)
    ax3.set_title('Back Force Analysis')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Force (kg)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Torque Analysis
    ax4 = axes[1, 1]
    torque_left = flight_data['FLIGHT_SEGMENT_l_torque'].fillna(0)
    torque_right = flight_data['FLIGHT_SEGMENT_r_torque'].fillna(0)
    
    ax4.plot(flight_data['_time'], torque_left, label='Left Torque', alpha=0.7, linewidth=1)
    ax4.plot(flight_data['_time'], torque_right, label='Right Torque', alpha=0.7, linewidth=1)
    ax4.plot(flight_data['_time'], abs(torque_left - torque_right), label='|Torque Diff|', color='red', linewidth=1.5)
    ax4.set_title('Torque Analysis')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Torque')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'flight_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Flight Distributions: {folder_name} (July 17, 2025)', fontsize=16, fontweight='bold')
    
    # Force distributions
    ax1 = axes[0, 0]
    ax1.hist(flight_data['total_force'], bins=50, alpha=0.7, color='red', edgecolor='black', density=True, weights=np.ones(len(flight_data['total_force'])) / len(flight_data['total_force']) * 100)
    ax1.axvline(flight_data['total_force'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {flight_data["total_force"].mean():.2f}')
    ax1.axvline(flight_data['total_force'].max(), color='darkred', linestyle='-', linewidth=2, label=f'Max: {flight_data["total_force"].max():.2f}')
    ax1.set_title('Total Force Distribution')
    ax1.set_xlabel('Force (kg)')
    ax1.set_ylabel('Probability (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Back force components
    ax2 = axes[0, 1]
    ax2.hist(flight_data['Backline_Left_kg'], bins=30, alpha=0.6, label='Left', color='blue', density=True, weights=np.ones(len(flight_data['Backline_Left_kg'])) / len(flight_data['Backline_Left_kg']) * 100)
    ax2.hist(flight_data['Backline_Right_kg'], bins=30, alpha=0.6, label='Right', color='green', density=True, weights=np.ones(len(flight_data['Backline_Right_kg'])) / len(flight_data['Backline_Right_kg']) * 100)
    ax2.set_title('Backline Force Distribution')
    ax2.set_xlabel('Force (kg)')
    ax2.set_ylabel('Probability (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Force difference
    ax3 = axes[1, 0]
    ax3.hist(flight_data['back_force_diff'], bins=30, alpha=0.7, color='purple', density=True, weights=np.ones(len(flight_data['back_force_diff'])) / len(flight_data['back_force_diff']) * 100)
    ax3.axvline(flight_data['back_force_diff'].mean(), color='indigo', linestyle='--', linewidth=2, 
                label=f'Mean: {flight_data["back_force_diff"].mean():.2f}')
    ax3.set_title('Back Force Difference |Left-Right|')
    ax3.set_xlabel('Force Difference (kg)')
    ax3.set_ylabel('Probability (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Torque distributions
    ax4 = axes[1, 1]
    torque_left_clean = torque_left[torque_left != 0]
    torque_right_clean = torque_right[torque_right != 0]
    
    if len(torque_left_clean) > 0:
        ax4.hist(torque_left_clean, bins=20, alpha=0.6, label='Left Torque', color='orange', density=True, weights=np.ones(len(torque_left_clean)) / len(torque_left_clean) * 100)
    if len(torque_right_clean) > 0:
        ax4.hist(torque_right_clean, bins=20, alpha=0.6, label='Right Torque', color='cyan', density=True, weights=np.ones(len(torque_right_clean)) / len(torque_right_clean) * 100)
    ax4.set_title('Torque Distribution (Non-zero values)')
    ax4.set_xlabel('Torque')
    ax4.set_ylabel('Probability (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'flight_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Created visualizations for {folder_name}")

# Process each flight and create visualizations
for i, (start_time, end_time) in enumerate(flights):
    # Create folder name with date and start hour/minute
    start_dt = datetime.strptime(f"2025-07-17 {start_time}", "%Y-%m-%d %H:%M:%S")
    folder_name = f"2025_07_17_{start_dt.hour:02d}_{start_dt.minute:02d}"
    
    print(f"Creating visualizations for flight {i+1}: {folder_name}")
    
    # Load flight data
    data_file = os.path.join(base_dir, folder_name, 'flight_data.csv')
    
    if os.path.exists(data_file):
        flight_data = pd.read_csv(data_file)
        flight_data['_time'] = pd.to_datetime(flight_data['_time'], format='mixed')
        
        create_flight_visualizations(folder_name, flight_data, start_time, end_time)
    else:
        print(f"  Warning: Data file not found for {folder_name}")

print(f"\nAll {len(flights)} flight visualizations created successfully!")
print(f"Results saved in: {base_dir}")
print("\nEach flight folder now contains:")
print("  - flight_metrics.txt (calculated metrics)")
print("  - flight_data.csv (raw data)")
print("  - flight_timeseries.png (time-series plots)")
print("  - flight_distributions.png (distribution plots)")
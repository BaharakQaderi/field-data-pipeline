#!/usr/bin/env python3
"""
Vaie Flights - Servo Temperature & Torque Time Series Analysis

Creates dual y-axis time series plots for each flight showing:
- Left Y-axis: Servo temperatures (left & right)
- Right Y-axis: Moving averaged torque (2-second window, absolute value)

Covers all flights from July 16 & 17, 2025.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def process_single_flight(flight_path, flight_name, date_str):
    """Process a single flight and create the dual y-axis plot"""
    
    flight_data_file = os.path.join(flight_path, 'flight_data.csv')
    
    if not os.path.exists(flight_data_file):
        print(f"  ⚠️  Flight data file not found: {flight_name}")
        return False
    
    try:
        # Load flight data
        df = pd.read_csv(flight_data_file)
        df['_time'] = pd.to_datetime(df['_time'])
        
        # Check if required columns exist
        required_cols = [
            'FLIGHT_SEGMENT_left_servo_temp',
            'FLIGHT_SEGMENT_right_servo_temp', 
            'FLIGHT_SEGMENT_l_torque',
            'FLIGHT_SEGMENT_r_torque'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠️  Missing columns in {flight_name}: {missing_cols}")
            return False
        
        # Calculate moving averaged torque (2-second window)
        # Assuming ~10Hz sampling rate, 2 seconds ≈ 20 samples
        window_size = 20
        
        # Calculate absolute torque values with moving average
        df['left_torque_abs'] = np.abs(df['FLIGHT_SEGMENT_l_torque'])
        df['right_torque_abs'] = np.abs(df['FLIGHT_SEGMENT_r_torque'])
        
        df['left_torque_ma'] = df['left_torque_abs'].rolling(window=window_size, center=True, min_periods=1).mean()
        df['right_torque_ma'] = df['right_torque_abs'].rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Calculate total absolute torque (combined left + right)
        df['total_torque_abs_ma'] = df['left_torque_ma'] + df['right_torque_ma']
        
        # Flight duration
        duration = (df['_time'].max() - df['_time'].min()).total_seconds()
        duration_str = f"{int(duration//60)}m {int(duration%60)}s"
        
        # Create the dual y-axis plot
        fig, ax1 = plt.subplots(figsize=(16, 8))
        
        # Left y-axis: Servo temperatures
        color_left_temp = '#1f77b4'  # Blue
        color_right_temp = '#ff7f0e' # Orange
        
        line1 = ax1.plot(df['_time'], df['FLIGHT_SEGMENT_left_servo_temp'], 
                        color=color_left_temp, linewidth=2, alpha=0.8,
                        label='Left Servo Temp')
        line2 = ax1.plot(df['_time'], df['FLIGHT_SEGMENT_right_servo_temp'], 
                        color=color_right_temp, linewidth=2, alpha=0.8,
                        label='Right Servo Temp')
        
        ax1.set_xlabel('Time', fontweight='bold')
        ax1.set_ylabel('Servo Temperature [°C]', color='black', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Right y-axis: Moving averaged torque
        ax2 = ax1.twinx()
        color_torque_left = '#2ca02c'   # Green
        color_torque_right = '#d62728'  # Red
        color_torque_total = '#9467bd'  # Purple
        
        line3 = ax2.plot(df['_time'], df['left_torque_ma'], 
                        color=color_torque_left, linewidth=1.5, alpha=0.7, linestyle='--',
                        label='Left Torque (2s MA)')
        line4 = ax2.plot(df['_time'], df['right_torque_ma'], 
                        color=color_torque_right, linewidth=1.5, alpha=0.7, linestyle='--',
                        label='Right Torque (2s MA)')
        line5 = ax2.plot(df['_time'], df['total_torque_abs_ma'], 
                        color=color_torque_total, linewidth=2.5, alpha=0.9,
                        label='Total Torque (2s MA)')
        
        ax2.set_ylabel('Absolute Torque [Nm] (2s Moving Avg)', color='black', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Title and formatting
        start_time = df['_time'].min().strftime('%H:%M:%S')
        end_time = df['_time'].max().strftime('%H:%M:%S')
        
        title = f'Vaie Flight - Servo Temperature & Torque Analysis\\n'
        title += f'{date_str} | {flight_name} | {start_time} - {end_time} | Duration: {duration_str}'
        
        plt.title(title, fontweight='bold', pad=20)
        
        # Combine legends
        lines1 = line1 + line2
        lines2 = line3 + line4 + line5
        labels1 = [l.get_label() for l in lines1]
        labels2 = [l.get_label() for l in lines2]
        
        # Create a single legend
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
        
        # Grid
        ax1.grid(True, alpha=0.6, linewidth=0.8, color='gray')
        ax1.minorticks_on()
        ax1.grid(which='minor', alpha=0.3, linewidth=0.5, color='lightgray')
        
        # Format x-axis for better time display
        ax1.tick_params(axis='x', rotation=45)
        
        # Style
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_filename = f"{flight_name}_servo_temp_torque_timeseries.png"
        output_path = os.path.join(flight_path, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Calculate and print statistics
        stats = {
            'duration_min': duration / 60,
            'data_points': len(df),
            'left_temp_avg': df['FLIGHT_SEGMENT_left_servo_temp'].mean(),
            'left_temp_max': df['FLIGHT_SEGMENT_left_servo_temp'].max(),
            'right_temp_avg': df['FLIGHT_SEGMENT_right_servo_temp'].mean(),
            'right_temp_max': df['FLIGHT_SEGMENT_right_servo_temp'].max(),
            'total_torque_avg': df['total_torque_abs_ma'].mean(),
            'total_torque_max': df['total_torque_abs_ma'].max()
        }
        
        print(f"  ✅ {flight_name}: {duration_str} | "
              f"Temp: L={stats['left_temp_avg']:.1f}°C, R={stats['right_temp_avg']:.1f}°C | "
              f"Torque: avg={stats['total_torque_avg']:.2f}Nm, max={stats['total_torque_max']:.2f}Nm")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {flight_name}: {e}")
        return False

def process_all_flights():
    """Process all flights from July 16 and July 17"""
    
    base_dirs = {
        'July 16, 2025': '/Users/baharakqaderi/field-data-pipeline/flight_analysis_july16',
        'July 17, 2025': '/Users/baharakqaderi/field-data-pipeline/flight_analysis_july17'
    }
    
    print("🚁 Vaie Flights - Servo Temperature & Torque Time Series Analysis")
    print("=" * 80)
    print("Creating dual y-axis plots:")
    print("  📊 Left axis: Servo temperatures (left & right)")
    print("  📈 Right axis: Moving averaged torque (2-second window, absolute)")
    print("=" * 80)
    
    total_processed = 0
    total_successful = 0
    
    for date_name, base_dir in base_dirs.items():
        print(f"\\n🗓️  Processing {date_name} flights...")
        
        if not os.path.exists(base_dir):
            print(f"  ⚠️  Directory not found: {base_dir}")
            continue
        
        # Get all flight directories
        flight_dirs = sorted([d for d in os.listdir(base_dir) 
                             if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('2025_07_')])
        
        print(f"  Found {len(flight_dirs)} flight directories")
        
        for i, flight_dir in enumerate(flight_dirs, 1):
            flight_path = os.path.join(base_dir, flight_dir)
            flight_name = f"Flight_{i}"
            
            total_processed += 1
            success = process_single_flight(flight_path, flight_name, date_name)
            if success:
                total_successful += 1
    
    print("\\n" + "=" * 80)
    print(f"📋 Summary: {total_successful}/{total_processed} flights processed successfully")
    print("\\n🎯 Generated files:")
    print("  - Individual flight plots: *_servo_temp_torque_timeseries.png")
    print("  - Located in respective flight directories")
    print("\\n📊 Each plot shows:")
    print("  • Blue/Orange lines: Left/Right servo temperatures")
    print("  • Green/Red dashed lines: Individual torque components (2s MA)")
    print("  • Purple solid line: Total absolute torque (2s MA)")

if __name__ == "__main__":
    process_all_flights()
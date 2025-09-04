#!/usr/bin/env python3
"""
Vaie Flights - Servo Temperature & Torque Time Series Analysis (Separate Plots)

Creates separate dual y-axis plots for each motor per flight:
- Left Motor Plot: Left servo temp + Left torque (2s moving average)
- Right Motor Plot: Right servo temp + Right torque (2s moving average)

Much clearer and more understandable than combined plots!
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 13
})

def create_motor_plot(df, motor_side, flight_info, output_dir):
    """Create a 3-subplot plot: temperature, torque, and roll percentage histogram"""
    
    # Define columns based on motor side
    if motor_side.lower() == 'left':
        temp_col = 'FLIGHT_SEGMENT_left_servo_temp'
        torque_col = 'FLIGHT_SEGMENT_l_torque'  # Use raw torque, not moving average
        color_temp = '#1f77b4'    # Blue
        color_torque = '#2ca02c'  # Green
        title_side = 'Left Motor'
    else:  # right
        temp_col = 'FLIGHT_SEGMENT_right_servo_temp'
        torque_col = 'FLIGHT_SEGMENT_r_torque'  # Use raw torque, not moving average
        color_temp = '#ff7f0e'    # Orange  
        color_torque = '#d62728'  # Red
        title_side = 'Right Motor'
    
    # Create 3 subplots: temperature (taller), torque, roll percentage histogram
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), 
                                        gridspec_kw={'height_ratios': [1.3, 1, 0.8]})
    
    # Top subplot: Servo temperature (RAW VALUES - no moving average)
    temp_data = df[temp_col].dropna()
    time_temp = df[df[temp_col].notna()]['_time']
    
    ax1.plot(time_temp, temp_data, 
             color=color_temp, linewidth=2.5, alpha=0.8, marker='o', markersize=3)
    
    ax1.set_ylabel('Servo Temperature [°C]', fontweight='bold', fontsize=13)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.grid(True, alpha=0.6, linewidth=0.8, color='gray')
    ax1.minorticks_on()
    ax1.grid(which='minor', alpha=0.3, linewidth=0.5, color='lightgray')
    
    # No legend for temperature plot - statistics will be shown in main title area
    
    # Middle subplot: Torque with moving average (torque is noisy, so MA makes sense)
    df['torque_abs'] = np.abs(df[torque_col])
    window_size = 20  # 2-second window at ~10Hz
    df['torque_ma'] = df['torque_abs'].rolling(window=window_size, center=True, min_periods=1).mean()
    
    ax2.plot(df['_time'], df['torque_ma'], 
             color=color_torque, linewidth=2.5, alpha=0.8,
             label=f'{title_side} Torque (2s MA)')
    
    ax2.set_ylabel('Absolute Torque [Nm] (2s Moving Average)', 
                   fontweight='bold', fontsize=13)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.grid(True, alpha=0.6, linewidth=0.8, color='gray')
    ax2.minorticks_on()
    ax2.grid(which='minor', alpha=0.3, linewidth=0.5, color='lightgray')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    # Bottom subplot: Roll percentage histogram (showing probability percentage)
    roll_data = df['FLIGHT_SEGMENT_roll_percentage'].dropna()
    
    # Create histogram with density=True to show probability percentages
    counts, bins, _ = ax3.hist(roll_data, bins=30, density=True, color='#9467bd', 
                              alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Convert density to percentage
    bin_width = bins[1] - bins[0]
    ax3.clear()  # Clear and redraw with percentage
    
    counts_percent = (counts * bin_width) * 100  # Convert to percentage
    ax3.bar(bins[:-1], counts_percent, width=bin_width, color='#9467bd', 
           alpha=0.7, edgecolor='black', linewidth=0.5, align='edge')
    
    ax3.set_xlabel('Roll Percentage [%]', fontweight='bold', fontsize=13)
    ax3.set_ylabel('Probability [%]', fontweight='bold', fontsize=13)
    ax3.tick_params(axis='both', labelsize=11)
    ax3.grid(True, alpha=0.6, linewidth=0.8, color='gray')
    ax3.minorticks_on()
    ax3.grid(which='minor', alpha=0.3, linewidth=0.5, color='lightgray')
    
    # Add histogram statistics
    roll_mean = roll_data.mean()
    roll_std = roll_data.std()
    ax3.axvline(roll_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {roll_mean:.1f}%')
    ax3.axvline(roll_mean + roll_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label=f'+1σ: {roll_mean+roll_std:.1f}%')
    ax3.axvline(roll_mean - roll_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label=f'-1σ: {roll_mean-roll_std:.1f}%')
    ax3.legend(loc='upper right', framealpha=0.9)
    
    # Title for the entire figure
    start_time = df['_time'].min().strftime('%H:%M:%S')
    end_time = df['_time'].max().strftime('%H:%M:%S')

    title = f'Vaie {title_side} - Temperature, Torque & Roll Analysis | '
    title += f'{flight_info["date"]} | {flight_info["name"]} | '
    title += f'{start_time} - {end_time} | Duration: {flight_info["duration"]}\n\n\n'
    
    fig.suptitle(title, fontweight='bold', fontsize=16, y=0.96)
    
    # Statistics text box below the main title (using figure coordinates)
    temp_avg = temp_data.mean()
    temp_max = temp_data.max()
    temp_min = temp_data.min()
    torque_avg = df['torque_ma'].mean()
    torque_max = df['torque_ma'].max()
    
    stats_text = f'{title_side} Stats | '
    stats_text += f'Temp: {temp_min:.1f}-{temp_max:.1f}°C (avg: {temp_avg:.1f}°C) | '
    stats_text += f'Torque: {torque_max:.2f}Nm max (avg: {torque_avg:.2f}Nm) | '
    stats_text += f'Roll: {roll_mean:.1f}% ± {roll_std:.1f}%'
    
    # Position statistics box below the main title
    fig.text(0.5, 0.92, stats_text, ha='center', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgray', alpha=0.8, edgecolor='gray'))
    
    # Format x-axis only for middle plot (time axis)
    ax2.tick_params(axis='x', rotation=45)
    ax3.tick_params(axis='x', rotation=0)  # No rotation for histogram
    
    # Style - remove top spines
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_filename = f"{flight_info['name']}_{motor_side.lower()}_motor_temp_torque.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'temp_avg': temp_avg,
        'temp_max': temp_max,
        'torque_avg': torque_avg,
        'torque_max': torque_max,
        'roll_mean': roll_mean,
        'roll_std': roll_std
    }

def process_single_flight(flight_path, flight_name, date_str):
    """Process a single flight and create separate left/right motor plots"""
    
    flight_data_file = os.path.join(flight_path, 'flight_data.csv')
    
    if not os.path.exists(flight_data_file):
        print(f"  ⚠️  Flight data file not found: {flight_name}")
        return False
    
    try:
        # Load flight data
        df = pd.read_csv(flight_data_file)
        df['_time'] = pd.to_datetime(df['_time'], format='mixed')
        
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
        
        # Flight info
        duration = (df['_time'].max() - df['_time'].min()).total_seconds()
        duration_str = f"{int(duration//60)}m {int(duration%60)}s"
        
        flight_info = {
            'name': flight_name,
            'date': date_str,
            'duration': duration_str
        }
        
        # Create separate plots for left and right motors
        left_stats = create_motor_plot(df, 'left', flight_info, flight_path)
        right_stats = create_motor_plot(df, 'right', flight_info, flight_path)
        
        print(f"  ✅ {flight_name}: {duration_str}")
        print(f"     Left  - Temp: {left_stats['temp_avg']:.1f}°C (max: {left_stats['temp_max']:.1f}°C) | "
              f"Torque: {left_stats['torque_avg']:.2f}Nm (max: {left_stats['torque_max']:.2f}Nm)")
        print(f"     Right - Temp: {right_stats['temp_avg']:.1f}°C (max: {right_stats['temp_max']:.1f}°C) | "
              f"Torque: {right_stats['torque_avg']:.2f}Nm (max: {right_stats['torque_max']:.2f}Nm)")
        
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
    
    print("🚁 Vaie Flights - Separate Motor Temperature & Torque Analysis")
    print("=" * 80)
    print("Creating SEPARATE dual y-axis plots for better clarity:")
    print("  🔵 Left Motor Plot: Left servo temp + Left torque (2s MA)")
    print("  🟠 Right Motor Plot: Right servo temp + Right torque (2s MA)")
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
    print(f"📊 Generated {total_successful * 2} individual motor plots")
    print("\\n🎯 Generated files per flight:")
    print("  - Flight_X_left_motor_temp_torque.png  (Left motor analysis)")
    print("  - Flight_X_right_motor_temp_torque.png (Right motor analysis)")
    print("\\n✨ Benefits of separate plots:")
    print("  • Clear correlation between each motor's temperature and torque")
    print("  • Easy to compare left vs right motor performance") 
    print("  • Statistics box for each motor")
    print("  • Much more understandable at a glance!")

if __name__ == "__main__":
    process_all_flights()
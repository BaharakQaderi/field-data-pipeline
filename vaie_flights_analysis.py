#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Enhanced plotting parameters for better quality
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

# July 16 flights
july16_flights = [
    ("10:29:25", "10:32:59"),
    ("10:58:01", "11:02:16"), 
    ("11:47:46", "12:18:40"),
    ("13:46:41", "13:51:33"),
    ("14:00:25", "14:03:27"),
    ("14:37:58", "14:46:57")
]

# July 17 flights
july17_flights = [
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

# Time windows for moving averages (in seconds)
t_width = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 60, 90, 120, 180]

base_dir_july16 = '/Users/baharakqaderi/field-data-pipeline/flight_analysis_july16'
base_dir_july17 = '/Users/baharakqaderi/field-data-pipeline/flight_analysis_july17'

def calculate_flight_duration(start_time, end_time):
    """Calculate flight duration in minutes and seconds"""
    start_dt = datetime.strptime(start_time, "%H:%M:%S")
    end_dt = datetime.strptime(end_time, "%H:%M:%S")
    duration = end_dt - start_dt
    
    total_seconds = duration.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    return f"{minutes}m {seconds}s"

def compute_moving_averages_and_max(series, t_width_list, sampling_rate_hz=10):
    """
    Compute moving averages for different time windows and return maximum values
    
    Parameters:
    - series: pandas Series of values
    - t_width_list: list of time windows in seconds
    - sampling_rate_hz: assumed sampling rate (default 10 Hz based on typical OPC data)
    
    Returns:
    - av_values: list of maximum values for each time window
    """
    av_values = []
    
    for window_sec in t_width_list:
        if window_sec == 0:
            # No averaging - use original data
            max_val = series.max()
        else:
            # Convert seconds to number of samples
            window_samples = max(1, int(window_sec * sampling_rate_hz))
            
            # Compute rolling average
            averaged_series = series.rolling(window=window_samples, center=True).mean()
            
            # Get maximum of the averaged series
            max_val = averaged_series.max()
        
        av_values.append(max_val)
    
    return av_values

def get_brake_status_percentage(flight_folder, date):
    """Get brake status percentage from ground segment data"""
    ground_file = os.path.join(flight_folder, 'ground_segment_data.csv')
    
    if os.path.exists(ground_file):
        try:
            ground_data = pd.read_csv(ground_file)
            if 'GROUND_SEGMENT_brake_command' in ground_data.columns and len(ground_data) > 0:
                brake_percentage = (ground_data['GROUND_SEGMENT_brake_command'] == 1).mean() * 100
                print(f"    Brake data found: {len(ground_data)} records, {brake_percentage:.1f}% engaged")
                return brake_percentage
            else:
                print(f"    No brake command column or empty data in {ground_file}")
        except Exception as e:
            print(f"    Error reading brake data: {e}")
    else:
        print(f"    No ground segment file found: {ground_file}")
    
    return 0.0

def analyze_single_flight(flight_folder, flight_name, date, start_time, end_time):
    """Analyze torque and force data for a single flight"""
    
    flight_data_file = os.path.join(flight_folder, 'flight_data.csv')
    
    if not os.path.exists(flight_data_file):
        print(f"Warning: Flight data not found for {flight_name}")
        return None
    
    # Load flight data
    flight_data = pd.read_csv(flight_data_file)
    flight_data['_time'] = pd.to_datetime(flight_data['_time'], format='mixed')
    
    print(f"\nAnalyzing {flight_name}:")
    print(f"  Data points: {len(flight_data)}")
    
    # Calculate absolute torque values: Tl = abs(Torque_left), Tr = abs(Torque_right)
    if 'FLIGHT_SEGMENT_l_torque' in flight_data.columns:
        Tl = abs(flight_data['FLIGHT_SEGMENT_l_torque'].fillna(0))
        print(f"  Left torque range: {Tl.min():.2f} to {Tl.max():.2f}")
    else:
        print(f"  Warning: Left torque data not found")
        Tl = pd.Series([0] * len(flight_data))
    
    if 'FLIGHT_SEGMENT_r_torque' in flight_data.columns:
        Tr = abs(flight_data['FLIGHT_SEGMENT_r_torque'].fillna(0))
        print(f"  Right torque range: {Tr.min():.2f} to {Tr.max():.2f}")
    else:
        print(f"  Warning: Right torque data not found")
        Tr = pd.Series([0] * len(flight_data))
    
    # Calculate force values: Fl = Left back force, Fr = Right back force
    if 'Backline_Left_kg' in flight_data.columns:
        Fl = flight_data['Backline_Left_kg'].fillna(0)
        print(f"  Left back force range: {Fl.min():.2f} to {Fl.max():.2f}")
    else:
        print(f"  Warning: Left back force data not found")
        Fl = pd.Series([0] * len(flight_data))
    
    if 'Backline_Right_kg' in flight_data.columns:
        Fr = flight_data['Backline_Right_kg'].fillna(0)
        print(f"  Right back force range: {Fr.min():.2f} to {Fr.max():.2f}")
    else:
        print(f"  Warning: Right back force data not found")
        Fr = pd.Series([0] * len(flight_data))
    
    # Estimate sampling rate from time data
    if len(flight_data) > 1:
        time_diff = (flight_data['_time'].iloc[-1] - flight_data['_time'].iloc[0]).total_seconds()
        estimated_rate = (len(flight_data) - 1) / time_diff
        print(f"  Estimated sampling rate: {estimated_rate:.2f} Hz")
    else:
        estimated_rate = 10  # Default fallback
    
    # Compute moving averages and get maximum values for each time window
    Av_Tl = compute_moving_averages_and_max(Tl, t_width, estimated_rate)
    Av_Tr = compute_moving_averages_and_max(Tr, t_width, estimated_rate)
    Av_Fl = compute_moving_averages_and_max(Fl, t_width, estimated_rate)
    Av_Fr = compute_moving_averages_and_max(Fr, t_width, estimated_rate)
    
    # Calculate force difference and sum
    force_diff = abs(Fl - Fr)
    force_sum = Fl + Fr
    Av_force_diff = compute_moving_averages_and_max(force_diff, t_width, estimated_rate)
    Av_force_sum = compute_moving_averages_and_max(force_sum, t_width, estimated_rate)
    
    # Get brake status
    brake_percentage = get_brake_status_percentage(flight_folder, date)
    
    print(f"  Computed moving averages for {len(t_width)} time windows")
    
    result = {
        'Av_Tl': Av_Tl,
        'Av_Tr': Av_Tr,
        'Av_Fl': Av_Fl,
        'Av_Fr': Av_Fr,
        'Av_force_diff': Av_force_diff,
        'Av_force_sum': Av_force_sum,
        'date': date,
        'start_time': start_time,
        'end_time': end_time,
        'brake_percentage': brake_percentage,
        'duration': calculate_flight_duration(start_time, end_time)
    }
    
    return result

def create_individual_flight_plots(flight_name, data, flight_folder, log_scale=True):
    """Create individual analysis plots for one flight"""
    
    scale_suffix = "_log" if log_scale else "_linear"
    scale_title = "(Log Scale)" if log_scale else "(Linear Scale)"
    
    # Create the 4 required plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Graph 1: t_width vs Av_Tl and Av_Tr
    ax1 = axes[0, 0]
    ax1.plot(t_width, data['Av_Tl'], 'o-', linewidth=2.5, markersize=8, label='Left Torque (Av_Tl)', 
            color='#FF6B35', markerfacecolor='white', markeredgewidth=2)
    ax1.plot(t_width, data['Av_Tr'], 's-', linewidth=2.5, markersize=8, label='Right Torque (Av_Tr)', 
            color='#004E89', markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_title(f'Graph 1: Torque Analysis {scale_title}', fontweight='bold')
    ax1.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax1.set_ylabel('Maximum Torque Value (Nm)', fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax1.set_facecolor('#f8f9fa')
    
    # Graph 2: t_width vs Left/Right back forces (Fl/Fr)
    ax2 = axes[0, 1]
    ax2.plot(t_width, data['Av_Fl'], 'o-', linewidth=2.5, markersize=8, label='Left Back Force (Fl)', 
            color='#28A745', markerfacecolor='white', markeredgewidth=2)
    ax2.plot(t_width, data['Av_Fr'], 's-', linewidth=2.5, markersize=8, label='Right Back Force (Fr)', 
            color='#DC3545', markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_title(f'Graph 2: Back Force Analysis {scale_title}', fontweight='bold')
    ax2.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax2.set_ylabel('Maximum Back Force (kg)', fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax2.set_facecolor('#f8f9fa')
    
    # Graph 3: t_width vs abs(Fl-Fr)
    ax3 = axes[1, 0]
    ax3.plot(t_width, data['Av_force_diff'], '^-', linewidth=2.5, markersize=8, label='|Left - Right| Force', 
            color='#6F42C1', markerfacecolor='white', markeredgewidth=2)
    
    ax3.set_title(f'Graph 3: Force Difference Analysis {scale_title}', fontweight='bold')
    ax3.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax3.set_ylabel('Maximum |Fl - Fr| (kg)', fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax3.set_facecolor('#f8f9fa')
    
    # Graph 4: t_width vs Fl+Fr
    ax4 = axes[1, 1]
    ax4.plot(t_width, data['Av_force_sum'], 'd-', linewidth=2.5, markersize=8, label='Total Back Force (Fl + Fr)', 
            color='#FD7E14', markerfacecolor='white', markeredgewidth=2)
    
    ax4.set_title(f'Graph 4: Total Back Force Analysis {scale_title}', fontweight='bold')
    ax4.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax4.set_ylabel('Maximum Fl + Fr (kg)', fontweight='bold')
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax4.set_facecolor('#f8f9fa')
    
    # Apply scale to all axes
    for ax in axes.flat:
        if log_scale:
            ax.set_xscale('log')
            ax.set_xticks(t_width[1:])  # Skip 0 for log scale
            ax.set_xticklabels([str(t) for t in t_width[1:]])
        else:
            ax.set_xticks(t_width)
            ax.set_xticklabels([str(t) for t in t_width])
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
    
    # Main title
    plt.suptitle(f'{flight_name} - Torque and Force Analysis {scale_title}\\n'
                f'Date: {data["date"].replace("2025-", "")} | Duration: {data["duration"]} | '
                f'POD Brake Status: {data["brake_percentage"]:.1f}% engaged', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save the plot
    output_file = os.path.join(flight_folder, f'{flight_name}_analysis{scale_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved individual plot: {output_file}")
    
    plt.close()

def create_combined_plots(all_results, output_dir, log_scale=True):
    """Create combined comparison plots separated by day"""
    
    scale_suffix = "_log" if log_scale else "_linear"
    scale_title = "(Log Scale)" if log_scale else "(Linear Scale)"
    
    # Separate results by day
    july16_results = {k: v for k, v in all_results.items() if k.startswith('July16')}
    july17_results = {k: v for k, v in all_results.items() if k.startswith('July17')}
    
    # Create separate plots for each day
    for day_name, day_results in [("July16", july16_results), ("July17", july17_results)]:
        if not day_results:
            continue
            
        colors = plt.cm.tab20(np.linspace(0, 1, len(day_results)))
        markers = ['o', 's', '^', 'v', 'd', 'p', 'h', '*', '8', 'P', 'X', 'D', '<', '>', '1', '2', '3', '4', '+', 'x']
        
        # Graph 1: Combined Torque Analysis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Collect all torque values to determine optimal y-axis range
        all_left_torque = []
        all_right_torque = []
        for data in day_results.values():
            all_left_torque.extend(data['Av_Tl'])
            all_right_torque.extend(data['Av_Tr'])
        
        # Left Torque
        for i, (flight_name, data) in enumerate(day_results.items()):
            marker = markers[i % len(markers)]
            flight_label = flight_name.replace(f"{day_name}_", "")  # Remove day prefix
            ax1.plot(t_width, data['Av_Tl'], marker + '-', linewidth=2.5, markersize=6,
                    label=f'{flight_label} ({data["duration"]}, {data["brake_percentage"]:.1f}%)', color=colors[i],
                    markerfacecolor='white', markeredgewidth=1.5)
        
        ax1.set_title(f'{day_name} - Left Torque Analysis {scale_title}', fontweight='bold', fontsize=16, pad=20)
        ax1.set_xlabel('Averaging Window (seconds)', fontweight='bold')
        ax1.set_ylabel('Maximum Left Torque (Nm)', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
        ax1.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
        ax1.set_facecolor('#f8f9fa')
        
        # Set fixed y-axis range for left torque - simple 0-20 range
        ax1.set_ylim(0, 20)
        
        # Right Torque
        for i, (flight_name, data) in enumerate(day_results.items()):
            marker = markers[i % len(markers)]
            flight_label = flight_name.replace(f"{day_name}_", "")  # Remove day prefix
            ax2.plot(t_width, data['Av_Tr'], marker + '-', linewidth=2.5, markersize=6,
                    label=f'{flight_label} ({data["duration"]}, {data["brake_percentage"]:.1f}%)', color=colors[i],
                    markerfacecolor='white', markeredgewidth=1.5)
        
        ax2.set_title(f'{day_name} - Right Torque Analysis {scale_title}', fontweight='bold', fontsize=16, pad=20)
        ax2.set_xlabel('Averaging Window (seconds)', fontweight='bold')
        ax2.set_ylabel('Maximum Right Torque (Nm)', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
        ax2.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
        ax2.set_facecolor('#f8f9fa')
        
        # Set fixed y-axis range for right torque - simple 0-20 range
        ax2.set_ylim(0, 20)
        
        # Apply scale
        for ax in [ax1, ax2]:
            if log_scale:
                ax.set_xscale('log')
                ax.set_xticks(t_width[1:])
                ax.set_xticklabels([str(t) for t in t_width[1:]])
            else:
                ax.set_xticks(t_width)
                ax.set_xticklabels([str(t) for t in t_width])
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#666666')
            ax.spines['bottom'].set_color('#666666')
        
        plt.suptitle(f'{day_name} Flights - Combined Torque Analysis {scale_title}\\n'
                    'Maximum Torque Values vs Averaging Window', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85, top=0.92)
        
        output_file = os.path.join(output_dir, f'vaie_{day_name.lower()}_combined_torque_analysis{scale_suffix}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {day_name} combined torque plot: {output_file}")
        plt.close()
        
        # Graph 2: Combined Back Force Analysis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Collect all force values to determine optimal y-axis range
        all_left_force = []
        all_right_force = []
        for data in day_results.values():
            all_left_force.extend(data['Av_Fl'])
            all_right_force.extend(data['Av_Fr'])
        
        # Left Back Force
        for i, (flight_name, data) in enumerate(day_results.items()):
            marker = markers[i % len(markers)]
            flight_label = flight_name.replace(f"{day_name}_", "")
            ax1.plot(t_width, data['Av_Fl'], marker + '-', linewidth=2.5, markersize=6,
                    label=f'{flight_label} ({data["duration"]}, {data["brake_percentage"]:.1f}%)', color=colors[i],
                    markerfacecolor='white', markeredgewidth=1.5)
        
        ax1.set_title(f'{day_name} - Left Back Force Analysis {scale_title}', fontweight='bold', fontsize=16, pad=20)
        ax1.set_xlabel('Averaging Window (seconds)', fontweight='bold')
        ax1.set_ylabel('Maximum Left Back Force (kg)', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
        ax1.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
        ax1.set_facecolor('#f8f9fa')
        
        # Set optimized y-axis range for left force
        if all_left_force:
            max_val = max(all_left_force)
            ax1.set_ylim(0, max_val * 1.1)
        
        # Right Back Force
        for i, (flight_name, data) in enumerate(day_results.items()):
            marker = markers[i % len(markers)]
            flight_label = flight_name.replace(f"{day_name}_", "")
            ax2.plot(t_width, data['Av_Fr'], marker + '-', linewidth=2.5, markersize=6,
                    label=f'{flight_label} ({data["duration"]}, {data["brake_percentage"]:.1f}%)', color=colors[i],
                    markerfacecolor='white', markeredgewidth=1.5)
        
        ax2.set_title(f'{day_name} - Right Back Force Analysis {scale_title}', fontweight='bold', fontsize=16, pad=20)
        ax2.set_xlabel('Averaging Window (seconds)', fontweight='bold')
        ax2.set_ylabel('Maximum Right Back Force (kg)', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
        ax2.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
        ax2.set_facecolor('#f8f9fa')
        
        # Set optimized y-axis range for right force
        if all_right_force:
            max_val = max(all_right_force)
            ax2.set_ylim(0, max_val * 1.1)
        
        # Apply scale
        for ax in [ax1, ax2]:
            if log_scale:
                ax.set_xscale('log')
                ax.set_xticks(t_width[1:])
                ax.set_xticklabels([str(t) for t in t_width[1:]])
            else:
                ax.set_xticks(t_width)
                ax.set_xticklabels([str(t) for t in t_width])
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#666666')
            ax.spines['bottom'].set_color('#666666')
        
        plt.suptitle(f'{day_name} Flights - Combined Back Force Analysis {scale_title}\\n'
                    'Maximum Back Force Values vs Averaging Window', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85, top=0.92)
        
        output_file = os.path.join(output_dir, f'vaie_{day_name.lower()}_combined_back_force_analysis{scale_suffix}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {day_name} combined back force plot: {output_file}")
        plt.close()
        
        # Graph 3: Combined Force Difference Analysis
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Collect all force difference values to determine optimal y-axis range
        all_force_diff = []
        for data in day_results.values():
            all_force_diff.extend(data['Av_force_diff'])
        
        for i, (flight_name, data) in enumerate(day_results.items()):
            marker = markers[i % len(markers)]
            flight_label = flight_name.replace(f"{day_name}_", "")
            ax.plot(t_width, data['Av_force_diff'], marker + '-', linewidth=2.5, markersize=6,
                    label=f'{flight_label} ({data["duration"]}, {data["brake_percentage"]:.1f}%)', color=colors[i],
                    markerfacecolor='white', markeredgewidth=1.5)
        
        ax.set_title(f'{day_name} - Force Difference Analysis {scale_title}', fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel('Averaging Window (seconds)', fontweight='bold')
        ax.set_ylabel('Maximum |Fl - Fr| (kg)', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
        ax.set_facecolor('#f8f9fa')
        
        # Set optimized y-axis range for force difference
        if all_force_diff:
            max_val = max(all_force_diff)
            ax.set_ylim(0, max_val * 1.1)
        
        if log_scale:
            ax.set_xscale('log')
            ax.set_xticks(t_width[1:])
            ax.set_xticklabels([str(t) for t in t_width[1:]])
        else:
            ax.set_xticks(t_width)
            ax.set_xticklabels([str(t) for t in t_width])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        
        plt.suptitle(f'{day_name} Flights - Combined Force Difference Analysis {scale_title}\\n'
                    'Maximum Force Difference vs Averaging Window', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85, top=0.92)
        
        output_file = os.path.join(output_dir, f'vaie_{day_name.lower()}_combined_force_diff_analysis{scale_suffix}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {day_name} combined force difference plot: {output_file}")
        plt.close()
        
        # Graph 4: Combined Total Back Force Analysis
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Collect all total force values to determine optimal y-axis range
        all_total_force = []
        for data in day_results.values():
            all_total_force.extend(data['Av_force_sum'])
        
        for i, (flight_name, data) in enumerate(day_results.items()):
            marker = markers[i % len(markers)]
            flight_label = flight_name.replace(f"{day_name}_", "")
            ax.plot(t_width, data['Av_force_sum'], marker + '-', linewidth=2.5, markersize=6,
                    label=f'{flight_label} ({data["duration"]}, {data["brake_percentage"]:.1f}%)', color=colors[i],
                    markerfacecolor='white', markeredgewidth=1.5)
        
        ax.set_title(f'{day_name} - Total Back Force Analysis {scale_title}', fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel('Averaging Window (seconds)', fontweight='bold')
        ax.set_ylabel('Maximum Fl + Fr (kg)', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
        ax.set_facecolor('#f8f9fa')
        
        # Set optimized y-axis range for total force
        if all_total_force:
            max_val = max(all_total_force)
            ax.set_ylim(0, max_val * 1.1)
        
        if log_scale:
            ax.set_xscale('log')
            ax.set_xticks(t_width[1:])
            ax.set_xticklabels([str(t) for t in t_width[1:]])
        else:
            ax.set_xticks(t_width)
            ax.set_xticklabels([str(t) for t in t_width])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        
        plt.suptitle(f'{day_name} Flights - Combined Total Back Force Analysis {scale_title}\\n'
                    'Maximum Total Back Force vs Averaging Window', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85, top=0.92)
        
        output_file = os.path.join(output_dir, f'vaie_{day_name.lower()}_combined_total_force_analysis{scale_suffix}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {day_name} combined total back force plot: {output_file}")
        plt.close()

def main():
    """Main function to run the analysis"""
    
    print("Starting Vaie Flights Torque and Force Analysis...")
    print(f"Time windows: {t_width} seconds")
    
    all_results = {}
    
    # Process July 16 flights
    print("\nProcessing July 16 flights:")
    for i, (start_time, end_time) in enumerate(july16_flights):
        start_dt = datetime.strptime(f"2025-07-16 {start_time}", "%Y-%m-%d %H:%M:%S")
        folder_name = f"2025_07_16_{start_dt.hour:02d}_{start_dt.minute:02d}"
        flight_folder = os.path.join(base_dir_july16, folder_name)
        
        flight_name = f"July16_Flight_{i+1}"
        result = analyze_single_flight(flight_folder, flight_name, "2025-07-16", start_time, end_time)
        
        if result:
            all_results[flight_name] = result
            # Create individual plots for this flight (both log and linear)
            create_individual_flight_plots(flight_name, result, flight_folder, log_scale=True)
            create_individual_flight_plots(flight_name, result, flight_folder, log_scale=False)
    
    # Process July 17 flights
    print("\nProcessing July 17 flights:")
    for i, (start_time, end_time) in enumerate(july17_flights):
        start_dt = datetime.strptime(f"2025-07-17 {start_time}", "%Y-%m-%d %H:%M:%S")
        folder_name = f"2025_07_17_{start_dt.hour:02d}_{start_dt.minute:02d}"
        flight_folder = os.path.join(base_dir_july17, folder_name)
        
        flight_name = f"July17_Flight_{i+1}"
        result = analyze_single_flight(flight_folder, flight_name, "2025-07-17", start_time, end_time)
        
        if result:
            all_results[flight_name] = result
            # Create individual plots for this flight (both log and linear)
            create_individual_flight_plots(flight_name, result, flight_folder, log_scale=True)
            create_individual_flight_plots(flight_name, result, flight_folder, log_scale=False)
    
    if not all_results:
        print("Error: No flight data found!")
        return
    
    # Create combined plots in base directory
    output_dir = '/Users/baharakqaderi/field-data-pipeline'
    print(f"\nCreating combined plots...")
    create_combined_plots(all_results, output_dir, log_scale=True)
    create_combined_plots(all_results, output_dir, log_scale=False)
    
    print(f"\nVaie flights analysis complete!")
    print(f"Analyzed {len(all_results)} flights total")
    print(f"Individual plots saved in respective flight folders")
    print(f"Combined plots saved in: {output_dir}")
    print("\nGenerated plots:")
    print("  Individual flights: 4 graphs per flight (log + linear versions)")
    print("  Combined analysis: 4 combined plots (log + linear versions)")

if __name__ == "__main__":
    main()
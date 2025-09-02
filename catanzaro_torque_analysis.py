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

# Catanzaro flights 5, 6, 7 details
flights_info = {
    'Flight_5': {
        'date': '2025-07-29',
        'start_time': '08:41:05',
        'end_time': '08:47:18',
        'folder': '2025_07_29_08_41_Flight_5'
    },
    'Flight_6': {
        'date': '2025-07-29', 
        'start_time': '09:11:09',
        'end_time': '09:18:14',
        'folder': '2025_07_29_09_11_Flight_6'
    },
    'Flight_7': {
        'date': '2025-07-29',
        'start_time': '09:45:01', 
        'end_time': '09:47:30',
        'folder': '2025_07_29_09_45_Flight_7'
    }
}

# Time windows for moving averages (in seconds)
t_width = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 60, 90, 120, 180]

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

def compute_moving_averages_and_max(torque_series, t_width_list, sampling_rate_hz=10):
    """
    Compute moving averages for different time windows and return maximum values
    
    Parameters:
    - torque_series: pandas Series of torque values
    - t_width_list: list of time windows in seconds
    - sampling_rate_hz: assumed sampling rate (default 10 Hz based on typical OPC data)
    
    Returns:
    - av_values: list of maximum values for each time window
    """
    av_values = []
    
    for window_sec in t_width_list:
        if window_sec == 0:
            # No averaging - use original data
            max_val = torque_series.max()
        else:
            # Convert seconds to number of samples
            window_samples = max(1, int(window_sec * sampling_rate_hz))
            
            # Compute rolling average
            averaged_series = torque_series.rolling(window=window_samples, center=True).mean()
            
            # Get maximum of the averaged series
            max_val = averaged_series.max()
        
        av_values.append(max_val)
    
    return av_values

def analyze_flight_torque(flight_name, flight_info):
    """Analyze torque data for a single flight"""
    
    folder_path = os.path.join(base_dir, flight_info['folder'])
    flight_data_file = os.path.join(folder_path, 'flight_segment_data.csv')
    
    if not os.path.exists(flight_data_file):
        print(f"Warning: Flight segment data not found for {flight_name}")
        return None, None, None
    
    # Load flight segment data
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
    
    print(f"  Computed moving averages for {len(t_width)} time windows")
    
    return Av_Tl, Av_Tr, flight_info

def create_torque_analysis_plots():
    """Create torque analysis plots for all Catanzaro flights"""
    
    print("Starting Catanzaro Torque Analysis...")
    print(f"Time windows: {t_width} seconds")
    
    # Analyze each flight
    results = {}
    for flight_name, flight_info in flights_info.items():
        Av_Tl, Av_Tr, info = analyze_flight_torque(flight_name, flight_info)
        if Av_Tl is not None:
            results[flight_name] = {
                'Av_Tl': Av_Tl,
                'Av_Tr': Av_Tr,
                'info': info
            }
    
    if not results:
        print("Error: No flight data found!")
        return
    
    # Create individual plots for each flight (both log and linear scales)
    for flight_name, data in results.items():
        create_individual_flight_plot(flight_name, data)
        create_individual_flight_plot_linear(flight_name, data)
    
    # Create combined comparison plots (both log and linear scales)
    create_combined_comparison_plot(results)
    create_combined_comparison_plot_linear(results)
    
    print(f"\nTorque analysis complete! Both log-scale and linear-scale plots saved in flight folders and base directory.")

def create_individual_flight_plot(flight_name, data):
    """Create individual torque analysis plot for one flight"""
    
    Av_Tl = data['Av_Tl']
    Av_Tr = data['Av_Tr'] 
    info = data['info']
    
    # Calculate flight duration
    duration = calculate_flight_duration(info['start_time'], info['end_time'])
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot both series
    ax.plot(t_width, Av_Tl, 'o-', linewidth=2.5, markersize=8, label='Left Torque (Av_Tl)', 
            color='#FF6B35', markerfacecolor='white', markeredgewidth=2)
    ax.plot(t_width, Av_Tr, 's-', linewidth=2.5, markersize=8, label='Right Torque (Av_Tr)', 
            color='#004E89', markerfacecolor='white', markeredgewidth=2)
    
    # Enhance the plot
    ax.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax.set_ylabel('Maximum Torque Value (Nm)', fontweight='bold')
    ax.set_title(f'Torque Analysis: {flight_name}\n'
                f'Date: {info["date"]} | Duration: {duration} | '
                f'Time: {info["start_time"]} - {info["end_time"]}', 
                fontweight='bold', pad=20)
    
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax.set_facecolor('#f8f9fa')
    
    # Set x-axis to log scale for better visualization of the time windows
    ax.set_xscale('log')
    ax.set_xticks(t_width[1:])  # Skip 0 for log scale
    ax.set_xticklabels([str(t) for t in t_width[1:]])
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    plt.tight_layout()
    
    # Save the plot
    folder_path = os.path.join(base_dir, info['folder'])
    output_file = os.path.join(folder_path, f'{flight_name}_torque_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved individual plot: {output_file}")
    
    plt.close()

def create_individual_flight_plot_linear(flight_name, data):
    """Create individual torque analysis plot for one flight with linear scale"""
    
    Av_Tl = data['Av_Tl']
    Av_Tr = data['Av_Tr'] 
    info = data['info']
    
    # Calculate flight duration
    duration = calculate_flight_duration(info['start_time'], info['end_time'])
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot both series
    ax.plot(t_width, Av_Tl, 'o-', linewidth=2.5, markersize=8, label='Left Torque (Av_Tl)', 
            color='#FF6B35', markerfacecolor='white', markeredgewidth=2)
    ax.plot(t_width, Av_Tr, 's-', linewidth=2.5, markersize=8, label='Right Torque (Av_Tr)', 
            color='#004E89', markerfacecolor='white', markeredgewidth=2)
    
    # Enhance the plot
    ax.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax.set_ylabel('Maximum Torque Value (Nm)', fontweight='bold')
    ax.set_title(f'Torque Analysis: {flight_name} (Linear Scale)\n'
                f'Date: {info["date"]} | Duration: {duration} | '
                f'Time: {info["start_time"]} - {info["end_time"]}', 
                fontweight='bold', pad=20)
    
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax.set_facecolor('#f8f9fa')
    
    # Use linear scale for x-axis
    ax.set_xticks(t_width)
    ax.set_xticklabels([str(t) for t in t_width])
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    plt.tight_layout()
    
    # Save the plot
    folder_path = os.path.join(base_dir, info['folder'])
    output_file = os.path.join(folder_path, f'{flight_name}_torque_analysis_linear.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved individual linear plot: {output_file}")
    
    plt.close()

def create_combined_comparison_plot_linear(results):
    """Create combined comparison plot for all flights with linear scale"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    colors = ['#FF6B35', '#004E89', '#A23B72']
    markers = ['o', 's', '^']
    
    # Plot Left Torque comparison
    for i, (flight_name, data) in enumerate(results.items()):
        duration = calculate_flight_duration(data['info']['start_time'], data['info']['end_time'])
        ax1.plot(t_width, data['Av_Tl'], markers[i] + '-', linewidth=2.5, markersize=8,
                label=f'{flight_name} (Duration: {duration})', color=colors[i],
                markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_title('Left Torque Analysis - All Catanzaro Flights (Linear Scale)', fontweight='bold', fontsize=16, pad=20)
    ax1.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax1.set_ylabel('Maximum Left Torque (Nm)', fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax1.set_facecolor('#f8f9fa')
    ax1.set_xticks(t_width)
    ax1.set_xticklabels([str(t) for t in t_width])
    
    # Plot Right Torque comparison
    for i, (flight_name, data) in enumerate(results.items()):
        duration = calculate_flight_duration(data['info']['start_time'], data['info']['end_time'])
        ax2.plot(t_width, data['Av_Tr'], markers[i] + '-', linewidth=2.5, markersize=8,
                label=f'{flight_name} (Duration: {duration})', color=colors[i],
                markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_title('Right Torque Analysis - All Catanzaro Flights (Linear Scale)', fontweight='bold', fontsize=16, pad=20)
    ax2.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax2.set_ylabel('Maximum Right Torque (Nm)', fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax2.set_facecolor('#f8f9fa')
    ax2.set_xticks(t_width)
    ax2.set_xticklabels([str(t) for t in t_width])
    
    # Style both subplots
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
    
    plt.suptitle('Catanzaro Flights Torque Analysis Comparison (Linear Scale)\n'
                'Maximum Torque Values vs Averaging Window', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save combined plot
    output_file = os.path.join(base_dir, 'catanzaro_torque_analysis_comparison_linear.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved combined linear comparison plot: {output_file}")
    
    plt.close()

def create_combined_comparison_plot(results):
    """Create combined comparison plot for all flights"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    colors = ['#FF6B35', '#004E89', '#A23B72']
    markers = ['o', 's', '^']
    
    # Plot Left Torque comparison
    for i, (flight_name, data) in enumerate(results.items()):
        duration = calculate_flight_duration(data['info']['start_time'], data['info']['end_time'])
        ax1.plot(t_width, data['Av_Tl'], markers[i] + '-', linewidth=2.5, markersize=8,
                label=f'{flight_name} (Duration: {duration})', color=colors[i],
                markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_title('Left Torque Analysis - All Catanzaro Flights', fontweight='bold', fontsize=16, pad=20)
    ax1.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax1.set_ylabel('Maximum Left Torque (Nm)', fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax1.set_facecolor('#f8f9fa')
    ax1.set_xscale('log')
    ax1.set_xticks(t_width[1:])
    ax1.set_xticklabels([str(t) for t in t_width[1:]])
    
    # Plot Right Torque comparison
    for i, (flight_name, data) in enumerate(results.items()):
        duration = calculate_flight_duration(data['info']['start_time'], data['info']['end_time'])
        ax2.plot(t_width, data['Av_Tr'], markers[i] + '-', linewidth=2.5, markersize=8,
                label=f'{flight_name} (Duration: {duration})', color=colors[i],
                markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_title('Right Torque Analysis - All Catanzaro Flights', fontweight='bold', fontsize=16, pad=20)
    ax2.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax2.set_ylabel('Maximum Right Torque (Nm)', fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax2.set_facecolor('#f8f9fa')
    ax2.set_xscale('log')
    ax2.set_xticks(t_width[1:])
    ax2.set_xticklabels([str(t) for t in t_width[1:]])
    
    # Style both subplots
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
    
    plt.suptitle('Catanzaro Flights Torque Analysis Comparison\n'
                'Maximum Torque Values vs Averaging Window', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save combined plot
    output_file = os.path.join(base_dir, 'catanzaro_torque_analysis_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved combined comparison plot: {output_file}")
    
    plt.close()

if __name__ == "__main__":
    create_torque_analysis_plots()
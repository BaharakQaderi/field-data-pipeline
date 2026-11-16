#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime
import seaborn as sns

# Set style for better looking plots
plt.style.use('default')
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

# Time windows for moving averages (in seconds) - Modified as requested
t_width = [0, 1, 2, 3, 5, 7, 10, 15, 20]

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

def get_brake_status_percentage(ground_data):
    """Get brake status percentage from ground segment data"""
    if 'GROUND_SEGMENT_brake_command' in ground_data.columns and len(ground_data) > 0:
        brake_percentage = (ground_data['GROUND_SEGMENT_brake_command'] == 1).mean() * 100
        return brake_percentage
    return 0.0

def analyze_single_flight(flight_dir):
    """Analyze torque and force data for a single flight"""

    # Find FLIGHT_SEGMENT and GROUND_SEGMENT CSV files
    flight_csv = glob.glob(f"{flight_dir}/FLIGHT_SEGMENT_data_*.csv")
    ground_csv = glob.glob(f"{flight_dir}/GROUND_SEGMENT_data_*.csv")

    if not flight_csv or not ground_csv:
        print(f"Warning: Missing CSV files in {flight_dir}")
        return None

    flight_file = flight_csv[0]
    ground_file = ground_csv[0]

    # Load flight and ground data
    flight_data = pd.read_csv(flight_file)
    ground_data = pd.read_csv(ground_file)
    flight_data['_time'] = pd.to_datetime(flight_data['_time'], format='mixed')

    print(f"\nAnalyzing {os.path.basename(flight_dir)}:")
    print(f"  Flight data points: {len(flight_data)}")
    print(f"  Ground data points: {len(ground_data)}")

    # Extract timing info
    start_time = flight_data['_time'].iloc[0]
    end_time = flight_data['_time'].iloc[-1]
    date_str = start_time.strftime('%Y-%m-%d')
    start_time_str = start_time.strftime('%H:%M:%S')
    end_time_str = end_time.strftime('%H:%M:%S')
    duration_str = calculate_flight_duration(start_time_str, end_time_str)

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
    if 'gs_Backline_Left_kg' in flight_data.columns:
        Fl = flight_data['gs_Backline_Left_kg'].fillna(0)
        print(f"  Left back force range: {Fl.min():.2f} to {Fl.max():.2f} kgf")
    else:
        print(f"  Warning: Left back force data not found")
        Fl = pd.Series([0] * len(flight_data))

    if 'gs_Backline_Right_kg' in flight_data.columns:
        Fr = flight_data['gs_Backline_Right_kg'].fillna(0)
        print(f"  Right back force range: {Fr.min():.2f} to {Fr.max():.2f} kgf")
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
    brake_percentage = get_brake_status_percentage(ground_data)

    print(f"  Computed moving averages for {len(t_width)} time windows")
    print(f"  Brake engagement: {brake_percentage:.1f}%")

    result = {
        'Av_Tl': Av_Tl,
        'Av_Tr': Av_Tr,
        'Av_Fl': Av_Fl,
        'Av_Fr': Av_Fr,
        'Av_force_diff': Av_force_diff,
        'Av_force_sum': Av_force_sum,
        'date': date_str,
        'start_time': start_time_str,
        'end_time': end_time_str,
        'brake_percentage': brake_percentage,
        'duration': duration_str
    }

    return result

def create_individual_flight_plots(flight_name, data, flight_dir, log_scale=True):
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
    ax2.set_ylabel('Maximum Back Force (kgf)', fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax2.set_facecolor('#f8f9fa')

    # Graph 3: t_width vs abs(Fl-Fr)
    ax3 = axes[1, 0]
    ax3.plot(t_width, data['Av_force_diff'], '^-', linewidth=2.5, markersize=8, label='|Left - Right| Force',
            color='#6F42C1', markerfacecolor='white', markeredgewidth=2)

    ax3.set_title(f'Graph 3: Force Difference Analysis {scale_title}', fontweight='bold')
    ax3.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax3.set_ylabel('Maximum |Fl - Fr| (kgf)', fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.6, linewidth=1.0, linestyle='-', color='#cccccc')
    ax3.set_facecolor('#f8f9fa')

    # Graph 4: t_width vs Fl+Fr
    ax4 = axes[1, 1]
    ax4.plot(t_width, data['Av_force_sum'], 'd-', linewidth=2.5, markersize=8, label='Total Back Force (Fl + Fr)',
            color='#FD7E14', markerfacecolor='white', markeredgewidth=2)

    ax4.set_title(f'Graph 4: Total Back Force Analysis {scale_title}', fontweight='bold')
    ax4.set_xlabel('Averaging Window (seconds)', fontweight='bold')
    ax4.set_ylabel('Maximum Fl + Fr (kgf)', fontweight='bold')
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
    plt.suptitle(f'{flight_name} - Torque and Force Analysis {scale_title} | '
                f'Date: {data["date"].replace("2025-", "")} | Duration: {data["duration"]} | '
                f'POD Brake Status: {data["brake_percentage"]:.1f}% engaged',
                fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # Save the plot
    output_file = os.path.join(flight_dir, f'vaie_{flight_name}_analysis{scale_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved VAIE plot: {output_file}")

    plt.close()

def create_vaie_analysis_for_all_flights():
    """Create VAIE-style analysis plots for all flights in individual_flight_segments"""

    print("Creating VAIE-style analysis plots for all flights...")
    print(f"Time windows: {t_width} seconds")
    print("="*80)

    # Find all flight directories
    flight_dirs = sorted([d for d in os.listdir('.') if d.startswith('2025-07-')])

    if not flight_dirs:
        print("No flight directories found!")
        return

    print(f"Found {len(flight_dirs)} flight directories")

    all_results = {}
    flight_counter = 1

    for flight_dir in flight_dirs:
        print(f"\\n[{flight_counter}/{len(flight_dirs)}] Processing: {flight_dir}")

        # Analyze the flight
        result = analyze_single_flight(flight_dir)

        if result:
            flight_name = f"Flight_{flight_counter}"
            all_results[flight_name] = result

            # Create individual VAIE plots for this flight (both log and linear)
            create_individual_flight_plots(flight_name, result, flight_dir, log_scale=True)
            create_individual_flight_plots(flight_name, result, flight_dir, log_scale=False)

        else:
            print(f"  ✗ Failed to analyze {flight_dir}")

        flight_counter += 1

    if not all_results:
        print("Error: No flight data found!")
        return

    print(f"\\n{'='*80}")
    print(f"VAIE-style analysis complete!")
    print(f"Processed {len(all_results)} flights successfully")
    print("Each flight directory now contains:")
    print("  - vaie_Flight_X_analysis_log.png (logarithmic scale)")
    print("  - vaie_Flight_X_analysis_linear.png (linear scale)")

    return all_results

if __name__ == "__main__":
    create_vaie_analysis_for_all_flights()
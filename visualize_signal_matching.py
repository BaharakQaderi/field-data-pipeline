#!/usr/bin/env python3
"""
Visualize signal matching between Raspberry Pi and InfluxDB data
with detailed time windows for confirmation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

def load_raspberry_files_for_date(date):
    """Load the identified Raspberry Pi files for a specific date."""

    raspi_dir = Path("data/FORCES/log_raspi2")

    if date == "2025-07-16":
        # Top 3 files based on signal matching analysis
        filenames = [
            "loadcell_readings_20250715_161829.csv",
            "loadcell_readings_20250715_183235.csv",
            "loadcell_readings_20250715_211757.csv"
        ]
    elif date == "2025-07-17":
        # Top 2 files based on signal matching analysis
        filenames = [
            "loadcell_readings_20250715_224238.csv",
            "loadcell_readings_20250716_003735.csv"
        ]
    else:
        raise ValueError(f"Unsupported date: {date}")

    print(f"Loading Raspberry Pi files for {date}:")

    all_data = []
    for filename in filenames:
        filepath = raspi_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
            df = df.sort_values('datetime')
            df = df[df['total_force_kg'] > 0.001]  # Filter meaningful data
            df['source_file'] = filename
            all_data.append(df)
            print(f"  • {filename}: {len(df)} records")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('datetime')
        print(f"  Total combined: {len(combined_df)} records")
        return combined_df
    else:
        return None

def load_influx_data(date):
    """Load InfluxDB data for a specific date."""

    filepath = Path(f"data/INFLUX/FLIGHT_SEGMENT_data_{date}.csv")
    df = pd.read_csv(filepath)
    df['_time'] = pd.to_datetime(df['_time'], format='mixed')
    df = df.sort_values('_time')
    df = df[df['FLIGHT_SEGMENT_loadcells_force'] > 0.001]  # Filter meaningful data

    print(f"InfluxDB {date}: {len(df)} records")
    return df

def create_signal_comparison_plots(date, time_window_hours=1, num_windows=6):
    """Create detailed signal comparison plots for verification."""

    print(f"\n🔍 Creating Signal Comparison for {date}")
    print("=" * 50)

    # Load data
    raspi_df = load_raspberry_files_for_date(date)
    influx_df = load_influx_data(date)

    if raspi_df is None or len(raspi_df) == 0:
        print("❌ No Raspberry Pi data found!")
        return

    # Determine time windows
    influx_start = influx_df['_time'].min()
    influx_end = influx_df['_time'].max()
    total_hours = (influx_end - influx_start).total_seconds() / 3600

    print(f"InfluxDB time span: {influx_start} to {influx_end} ({total_hours:.1f} hours)")
    print(f"Creating {num_windows} time windows of {time_window_hours} hours each")

    # Create subplots
    fig, axes = plt.subplots(num_windows, 1, figsize=(16, 3 * num_windows))
    if num_windows == 1:
        axes = [axes]

    fig.suptitle(f'Signal Matching Verification - {date}\nRaspberry Pi vs InfluxDB Force Signals',
                 fontsize=16, fontweight='bold')

    for i in range(num_windows):
        # Calculate time window
        window_start = influx_start + pd.Timedelta(hours=i * total_hours / num_windows)
        window_end = window_start + pd.Timedelta(hours=time_window_hours)

        # Filter data to window
        influx_window = influx_df[
            (influx_df['_time'] >= window_start) &
            (influx_df['_time'] <= window_end)
        ].copy()

        # For Raspberry Pi, we need to handle the fact that timestamps might not align perfectly
        # Let's take a representative sample from the same relative position
        raspi_total_duration = (raspi_df['datetime'].max() - raspi_df['datetime'].min())
        raspi_start_offset = pd.Timedelta(hours=i * raspi_total_duration.total_seconds() / 3600 / num_windows)
        raspi_window_start = raspi_df['datetime'].min() + raspi_start_offset
        raspi_window_end = raspi_window_start + pd.Timedelta(hours=time_window_hours)

        raspi_window = raspi_df[
            (raspi_df['datetime'] >= raspi_window_start) &
            (raspi_df['datetime'] <= raspi_window_end)
        ].copy()

        # Plot the signals
        if len(influx_window) > 0:
            axes[i].plot(influx_window['_time'], influx_window['FLIGHT_SEGMENT_loadcells_force'],
                        'r-', alpha=0.7, linewidth=1, label=f'InfluxDB ({len(influx_window)} points)')

        if len(raspi_window) > 0:
            axes[i].plot(raspi_window['datetime'], raspi_window['total_force_kg'],
                        'b-', alpha=0.7, linewidth=1, label=f'Raspberry Pi ({len(raspi_window)} points)')

        # Formatting
        axes[i].set_title(f'Window {i+1}: {window_start.strftime("%H:%M")} - {window_end.strftime("%H:%M")}')
        axes[i].set_ylabel('Force (kg)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Format x-axis
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axes[i].tick_params(axis='x', rotation=45)

        # Add statistics
        if len(influx_window) > 0 and len(raspi_window) > 0:
            influx_mean = influx_window['FLIGHT_SEGMENT_loadcells_force'].mean()
            raspi_mean = raspi_window['total_force_kg'].mean()
            axes[i].text(0.02, 0.98, f'InfluxDB mean: {influx_mean:.1f} kg\nRaspberry mean: {raspi_mean:.1f} kg',
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))

    axes[-1].set_xlabel('Time')
    plt.tight_layout()

    # Save plot
    output_path = f"outputs/signal_verification_{date}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Signal verification plot saved to: {output_path}")
    plt.show()

    # Print summary statistics
    print(f"\n📊 Summary Statistics for {date}:")
    print(f"Raspberry Pi: {len(raspi_df)} records, mean={raspi_df['total_force_kg'].mean():.1f} kg, max={raspi_df['total_force_kg'].max():.1f} kg")
    print(f"InfluxDB: {len(influx_df)} records, mean={influx_df['FLIGHT_SEGMENT_loadcells_force'].mean():.1f} kg, max={influx_df['FLIGHT_SEGMENT_loadcells_force'].max():.1f} kg")

def create_histogram_comparison(date):
    """Create histogram comparison for overall signal characteristics."""

    print(f"\n📊 Creating Histogram Comparison for {date}")

    # Load data
    raspi_df = load_raspberry_files_for_date(date)
    influx_df = load_influx_data(date)

    if raspi_df is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Force distribution comparison
    ax1.hist(raspi_df['total_force_kg'], bins=100, alpha=0.7, color='blue',
             density=True, label=f'Raspberry Pi ({len(raspi_df)} records)')
    ax1.hist(influx_df['FLIGHT_SEGMENT_loadcells_force'], bins=100, alpha=0.7, color='red',
             density=True, label=f'InfluxDB ({len(influx_df)} records)')
    ax1.set_xlabel('Force (kg)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{date} - Force Distribution Comparison')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Force value correlation scatter
    # Sample both datasets to same size for comparison
    sample_size = min(1000, len(raspi_df), len(influx_df))
    raspi_sample = raspi_df['total_force_kg'].sample(sample_size).sort_values().values
    influx_sample = influx_df['FLIGHT_SEGMENT_loadcells_force'].sample(sample_size).sort_values().values

    ax2.scatter(raspi_sample, influx_sample, alpha=0.5, s=10)
    ax2.plot([0, max(raspi_sample.max(), influx_sample.max())],
             [0, max(raspi_sample.max(), influx_sample.max())], 'k--', alpha=0.5, label='Perfect correlation')
    ax2.set_xlabel('Raspberry Pi Force (kg)')
    ax2.set_ylabel('InfluxDB Force (kg)')
    ax2.set_title(f'{date} - Force Values Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'outputs/histogram_comparison_{date}.png', dpi=300, bbox_inches='tight')
    print(f"💾 Histogram comparison saved to: outputs/histogram_comparison_{date}.png")
    plt.show()

def main():
    """Main function to create all verification plots."""

    print("🚁 Signal Matching Verification Tool")
    print("=" * 60)

    # Create plots for both dates
    for date in ["2025-07-16", "2025-07-17"]:
        create_signal_comparison_plots(date, time_window_hours=0.5, num_windows=8)
        create_histogram_comparison(date)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
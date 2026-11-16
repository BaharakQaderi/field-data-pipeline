#!/usr/bin/env python3
"""
Reconstruct and visualize the complete day signals from both Raspberry Pi and InfluxDB
for July 16 and July 17 in separate panels for comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

def load_complete_raspberry_data(date):
    """Load and reconstruct the complete Raspberry Pi data for a given date."""

    raspi_dir = Path("data/FORCES/log_raspi2")

    if date == "2025-07-16":
        filenames = [
            "loadcell_readings_20250715_161829.csv",
            "loadcell_readings_20250715_183235.csv",
            "loadcell_readings_20250715_211757.csv"
        ]
    elif date == "2025-07-17":
        filenames = [
            "loadcell_readings_20250715_224238.csv",
            "loadcell_readings_20250716_003735.csv"
        ]
    else:
        raise ValueError(f"Unsupported date: {date}")

    print(f"\n🔍 Loading complete Raspberry Pi data for {date}:")

    all_data = []
    for i, filename in enumerate(filenames):
        filepath = raspi_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
            df = df.sort_values('datetime')

            # Keep all data, not just meaningful force data for complete reconstruction
            df['file_sequence'] = i + 1
            df['source_file'] = filename
            all_data.append(df)

            print(f"  File {i+1}: {filename}")
            print(f"    Total records: {len(df)}")
            print(f"    Time span: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"    Force range: {df['total_force_kg'].min():.1f} - {df['total_force_kg'].max():.1f} kg")
            print(f"    Records with force > 0: {len(df[df['total_force_kg'] > 0])}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('datetime')
        print(f"  📊 Combined total: {len(combined_df)} records")
        print(f"  📅 Complete time span: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")

        duration_hours = (combined_df['datetime'].max() - combined_df['datetime'].min()).total_seconds() / 3600
        print(f"  ⏱️ Total duration: {duration_hours:.2f} hours")

        return combined_df
    else:
        return None

def load_complete_influx_data(date):
    """Load complete InfluxDB data for a given date."""

    filepath = Path(f"data/INFLUX/FLIGHT_SEGMENT_data_{date}.csv")
    df = pd.read_csv(filepath)
    df['_time'] = pd.to_datetime(df['_time'], format='mixed')
    df = df.sort_values('_time')

    print(f"\n🌐 Loading complete InfluxDB data for {date}:")
    print(f"  Total records: {len(df)}")
    print(f"  Time span: {df['_time'].min()} to {df['_time'].max()}")

    # Show force data statistics
    force_data = df[df['FLIGHT_SEGMENT_loadcells_force'].notna()]
    meaningful_force = force_data[force_data['FLIGHT_SEGMENT_loadcells_force'] > 0]

    print(f"  Records with force data: {len(force_data)}")
    print(f"  Records with meaningful force (>0): {len(meaningful_force)}")
    if len(meaningful_force) > 0:
        print(f"  Force range: {meaningful_force['FLIGHT_SEGMENT_loadcells_force'].min():.1f} - {meaningful_force['FLIGHT_SEGMENT_loadcells_force'].max():.1f} kg")

    duration_hours = (df['_time'].max() - df['_time'].min()).total_seconds() / 3600
    print(f"  Total duration: {duration_hours:.2f} hours")

    return df

def create_full_day_comparison(date):
    """Create a comprehensive full-day comparison plot."""

    print(f"\n📊 Creating Full Day Reconstruction for {date}")
    print("=" * 60)

    # Load data
    raspi_df = load_complete_raspberry_data(date)
    influx_df = load_complete_influx_data(date)

    if raspi_df is None:
        print("❌ No Raspberry Pi data found!")
        return

    # Create figure with 4 panels
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'Complete Day Signal Reconstruction - {date}\nRaspberry Pi vs InfluxDB Comparison',
                 fontsize=16, fontweight='bold')

    # Panel 1: Raspberry Pi complete signal (including zeros)
    axes[0].plot(raspi_df['datetime'], raspi_df['total_force_kg'],
                'b-', linewidth=0.5, alpha=0.8, label=f'All data ({len(raspi_df)} points)')
    axes[0].set_title('Raspberry Pi - Complete Signal (All Data Points)', fontweight='bold')
    axes[0].set_ylabel('Force (kg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Color code different files
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for i, file_num in enumerate(raspi_df['file_sequence'].unique()):
        if pd.notna(file_num):
            file_data = raspi_df[raspi_df['file_sequence'] == file_num]
            if i < len(colors):
                axes[0].scatter(file_data['datetime'], file_data['total_force_kg'],
                              c=colors[i], s=0.1, alpha=0.5, label=f'File {int(file_num)}')

    # Panel 2: Raspberry Pi meaningful force only (> 0)
    raspi_meaningful = raspi_df[raspi_df['total_force_kg'] > 0.001]
    if len(raspi_meaningful) > 0:
        axes[1].plot(raspi_meaningful['datetime'], raspi_meaningful['total_force_kg'],
                    'b-', linewidth=0.8, alpha=0.8, label=f'Meaningful force ({len(raspi_meaningful)} points)')
        axes[1].set_title('Raspberry Pi - Meaningful Force Data Only (> 0.001 kg)', fontweight='bold')
        axes[1].set_ylabel('Force (kg)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Panel 3: InfluxDB complete signal
    influx_forces = influx_df['FLIGHT_SEGMENT_loadcells_force'].fillna(0)
    axes[2].plot(influx_df['_time'], influx_forces,
                'r-', linewidth=0.5, alpha=0.8, label=f'All data ({len(influx_df)} points)')
    axes[2].set_title('InfluxDB - Complete Signal (All Data Points)', fontweight='bold')
    axes[2].set_ylabel('Force (kg)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Panel 4: InfluxDB meaningful force only
    influx_meaningful = influx_df[influx_df['FLIGHT_SEGMENT_loadcells_force'] > 0.001]
    if len(influx_meaningful) > 0:
        axes[3].plot(influx_meaningful['_time'], influx_meaningful['FLIGHT_SEGMENT_loadcells_force'],
                    'r-', linewidth=0.8, alpha=0.8, label=f'Meaningful force ({len(influx_meaningful)} points)')
        axes[3].set_title('InfluxDB - Meaningful Force Data Only (> 0.001 kg)', fontweight='bold')
        axes[3].set_ylabel('Force (kg)')
        axes[3].set_xlabel('Time')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Format all x-axes
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_fontsize(10)

    plt.tight_layout()

    # Save plot
    output_path = f"outputs/full_day_reconstruction_{date}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Full day reconstruction saved to: {output_path}")
    plt.show()

    # Print detailed statistics
    print(f"\n📈 Detailed Statistics for {date}:")
    print("-" * 50)

    if len(raspi_meaningful) > 0:
        print(f"Raspberry Pi (meaningful data):")
        print(f"  Records: {len(raspi_meaningful)}")
        print(f"  Time span: {raspi_meaningful['datetime'].min()} to {raspi_meaningful['datetime'].max()}")
        print(f"  Mean force: {raspi_meaningful['total_force_kg'].mean():.2f} kg")
        print(f"  Force range: {raspi_meaningful['total_force_kg'].min():.1f} - {raspi_meaningful['total_force_kg'].max():.1f} kg")

    if len(influx_meaningful) > 0:
        print(f"InfluxDB (meaningful data):")
        print(f"  Records: {len(influx_meaningful)}")
        print(f"  Time span: {influx_meaningful['_time'].min()} to {influx_meaningful['_time'].max()}")
        print(f"  Mean force: {influx_meaningful['FLIGHT_SEGMENT_loadcells_force'].mean():.2f} kg")
        print(f"  Force range: {influx_meaningful['FLIGHT_SEGMENT_loadcells_force'].min():.1f} - {influx_meaningful['FLIGHT_SEGMENT_loadcells_force'].max():.1f} kg")

    # Check for data gaps
    if len(raspi_meaningful) > 0:
        raspi_time_diffs = raspi_meaningful['datetime'].diff().dt.total_seconds()
        large_gaps = raspi_time_diffs[raspi_time_diffs > 300]  # Gaps > 5 minutes
        print(f"Raspberry Pi large gaps (>5min): {len(large_gaps)}")

    if len(influx_meaningful) > 0:
        influx_time_diffs = influx_meaningful['_time'].diff().dt.total_seconds()
        large_gaps = influx_time_diffs[influx_time_diffs > 300]  # Gaps > 5 minutes
        print(f"InfluxDB large gaps (>5min): {len(large_gaps)}")

def create_side_by_side_comparison():
    """Create side-by-side comparison for both days."""

    fig, axes = plt.subplots(2, 2, figsize=(24, 12))
    fig.suptitle('Complete Signal Overview - July 16 vs July 17\nRaspberry Pi (left) vs InfluxDB (right)',
                 fontsize=16, fontweight='bold')

    dates = ["2025-07-16", "2025-07-17"]
    colors = [('blue', 'red'), ('darkblue', 'darkred')]

    for i, date in enumerate(dates):
        raspi_df = load_complete_raspberry_data(date)
        influx_df = load_complete_influx_data(date)

        if raspi_df is not None:
            raspi_meaningful = raspi_df[raspi_df['total_force_kg'] > 0.001]
            axes[i, 0].plot(raspi_meaningful['datetime'], raspi_meaningful['total_force_kg'],
                          colors[i][0], linewidth=0.5, alpha=0.8)
            axes[i, 0].set_title(f'{date} - Raspberry Pi Signal', fontweight='bold')
            axes[i, 0].set_ylabel('Force (kg)')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        influx_meaningful = influx_df[influx_df['FLIGHT_SEGMENT_loadcells_force'] > 0.001]
        if len(influx_meaningful) > 0:
            axes[i, 1].plot(influx_meaningful['_time'], influx_meaningful['FLIGHT_SEGMENT_loadcells_force'],
                          colors[i][1], linewidth=0.5, alpha=0.8)
            axes[i, 1].set_title(f'{date} - InfluxDB Signal', fontweight='bold')
            axes[i, 1].set_ylabel('Force (kg)')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Format x-axes
    for i in range(2):
        for j in range(2):
            axes[i, j].tick_params(axis='x', rotation=45)
            if i == 1:  # Bottom row
                axes[i, j].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig('outputs/side_by_side_signal_comparison.png', dpi=300, bbox_inches='tight')
    print(f"💾 Side-by-side comparison saved to: outputs/side_by_side_signal_comparison.png")
    plt.show()

def main():
    """Main function to create all reconstruction plots."""

    print("🚁 Full Day Signal Reconstruction Tool")
    print("=" * 60)

    # Create detailed plots for each day
    for date in ["2025-07-16", "2025-07-17"]:
        create_full_day_comparison(date)
        print("\n" + "="*60)

    # Create side-by-side overview
    print("\n🔄 Creating Side-by-Side Overview...")
    create_side_by_side_comparison()

if __name__ == "__main__":
    main()
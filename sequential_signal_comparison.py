#!/usr/bin/env python3
"""
Sequential signal comparison - arrange Raspberry Pi files in order
and compare with InfluxDB data using sequential index instead of timestamps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_raspi_files_sequentially(date):
    """Load Raspberry Pi files in sequential order, ignoring timestamps."""

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

    print(f"\n🔍 Loading Raspberry Pi files sequentially for {date}:")

    all_data = []
    sequential_index = 0

    for i, filename in enumerate(filenames):
        filepath = raspi_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df = df[df['total_force_kg'] > 0.001]  # Only meaningful data

            # Create sequential index
            df['sequential_index'] = range(sequential_index, sequential_index + len(df))
            df['file_number'] = i + 1
            df['source_file'] = filename

            all_data.append(df)
            sequential_index += len(df)

            print(f"  File {i+1}: {filename}")
            print(f"    Records: {len(df)}")
            print(f"    Index range: {df['sequential_index'].min()} - {df['sequential_index'].max()}")
            print(f"    Force range: {df['total_force_kg'].min():.1f} - {df['total_force_kg'].max():.1f} kg")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"  📊 Total sequential records: {len(combined_df)}")
        return combined_df
    else:
        return None

def load_influx_sequential(date):
    """Load InfluxDB data and create sequential index."""

    filepath = Path(f"data/INFLUX/FLIGHT_SEGMENT_data_{date}.csv")
    df = pd.read_csv(filepath)
    df['_time'] = pd.to_datetime(df['_time'], format='mixed')
    df = df.sort_values('_time')
    df = df[df['FLIGHT_SEGMENT_loadcells_force'] > 0.001]  # Only meaningful data

    # Create sequential index
    df['sequential_index'] = range(len(df))

    print(f"\n🌐 InfluxDB {date}:")
    print(f"  Records: {len(df)}")
    print(f"  Force range: {df['FLIGHT_SEGMENT_loadcells_force'].min():.1f} - {df['FLIGHT_SEGMENT_loadcells_force'].max():.1f} kg")

    return df

def create_sequential_comparison(date):
    """Create comparison using sequential indices."""

    print(f"\n📊 Creating Sequential Comparison for {date}")
    print("=" * 50)

    # Load data
    raspi_df = load_raspi_files_sequentially(date)
    influx_df = load_influx_sequential(date)

    if raspi_df is None:
        return

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Sequential Signal Comparison - {date}\nRaspberry Pi vs InfluxDB (Sequential Order)',
                 fontsize=16, fontweight='bold')

    # Panel 1: Raspberry Pi sequential signal
    axes[0].plot(raspi_df['sequential_index'], raspi_df['total_force_kg'],
                'b-', linewidth=0.5, alpha=0.8, label=f'Raspberry Pi ({len(raspi_df)} points)')
    axes[0].set_title('Raspberry Pi - Sequential Force Signal', fontweight='bold')
    axes[0].set_ylabel('Force (kg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Color different files
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for file_num in raspi_df['file_number'].unique():
        file_data = raspi_df[raspi_df['file_number'] == file_num]
        color = colors[(file_num - 1) % len(colors)]
        axes[0].axvspan(file_data['sequential_index'].min(), file_data['sequential_index'].max(),
                       alpha=0.1, color=color, label=f'File {file_num}')

    # Panel 2: InfluxDB sequential signal
    axes[1].plot(influx_df['sequential_index'], influx_df['FLIGHT_SEGMENT_loadcells_force'],
                'r-', linewidth=0.5, alpha=0.8, label=f'InfluxDB ({len(influx_df)} points)')
    axes[1].set_title('InfluxDB - Sequential Force Signal', fontweight='bold')
    axes[1].set_ylabel('Force (kg)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Overlay comparison (normalized)
    # Normalize both signals to same length for comparison
    min_length = min(len(raspi_df), len(influx_df))

    raspi_normalized = np.interp(np.linspace(0, 1, min_length),
                                np.linspace(0, 1, len(raspi_df)),
                                raspi_df['total_force_kg'].values)

    influx_normalized = np.interp(np.linspace(0, 1, min_length),
                                 np.linspace(0, 1, len(influx_df)),
                                 influx_df['FLIGHT_SEGMENT_loadcells_force'].values)

    axes[2].plot(range(min_length), raspi_normalized, 'b-', linewidth=0.8, alpha=0.7, label='Raspberry Pi (normalized)')
    axes[2].plot(range(min_length), influx_normalized, 'r-', linewidth=0.8, alpha=0.7, label='InfluxDB (normalized)')
    axes[2].set_title('Normalized Overlay Comparison', fontweight='bold')
    axes[2].set_ylabel('Force (kg)')
    axes[2].set_xlabel('Normalized Sequential Index')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = f"outputs/sequential_comparison_{date}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Sequential comparison saved to: {output_path}")
    plt.show()

    # Statistics
    print(f"\n📈 Statistics for {date}:")
    print(f"Raspberry Pi: mean={raspi_df['total_force_kg'].mean():.1f} kg, max={raspi_df['total_force_kg'].max():.1f} kg")
    print(f"InfluxDB: mean={influx_df['FLIGHT_SEGMENT_loadcells_force'].mean():.1f} kg, max={influx_df['FLIGHT_SEGMENT_loadcells_force'].max():.1f} kg")

    # Correlation of normalized signals
    correlation = np.corrcoef(raspi_normalized, influx_normalized)[0, 1]
    print(f"Normalized signal correlation: {correlation:.3f}")

def main():
    """Main function."""

    print("🚁 Sequential Signal Comparison Tool")
    print("=" * 50)

    for date in ["2025-07-16", "2025-07-17"]:
        create_sequential_comparison(date)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
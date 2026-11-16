#!/usr/bin/env python3
"""
Verify sampling frequencies and time intervals between Raspberry Pi FORCES data
and InfluxDB FLIGHT_SEGMENT loadcells data to understand if alignment makes sense.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def analyze_sampling_frequency():
    """Analyze sampling frequencies of both data sources."""

    print("🔍 Analyzing Sampling Frequencies")
    print("=" * 60)

    # 1. Analyze Raspberry Pi FORCES data
    print("\n📡 RASPBERRY PI FORCES DATA:")
    print("-" * 40)

    forces_dir = Path("data/FORCES")
    loadcell_files = sorted(list(forces_dir.glob("loadcell_readings_*.csv")))

    for i, filepath in enumerate(loadcell_files[:2]):  # First 2 files (July 16)
        print(f"\nFile {i+1}: {filepath.name}")

        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df = df.sort_values('datetime')

        # Calculate time differences between consecutive readings
        time_diffs = df['datetime'].diff().dt.total_seconds()
        time_diffs = time_diffs.dropna()

        print(f"  Total records: {len(df)}")
        print(f"  Time span: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).total_seconds():.1f} seconds")
        print(f"  Mean interval: {time_diffs.mean():.3f} seconds")
        print(f"  Median interval: {time_diffs.median():.3f} seconds")
        print(f"  Min interval: {time_diffs.min():.3f} seconds")
        print(f"  Max interval: {time_diffs.max():.3f} seconds")
        print(f"  Estimated frequency: {1/time_diffs.mean():.2f} Hz")

        # Check for consistent sampling
        consistent_samples = (time_diffs < 0.2).sum()  # Within 0.2 seconds
        print(f"  Consistent samples (<0.2s): {consistent_samples}/{len(time_diffs)} ({100*consistent_samples/len(time_diffs):.1f}%)")

    # 2. Analyze InfluxDB data
    print(f"\n🌐 INFLUXDB FLIGHT_SEGMENT DATA:")
    print("-" * 40)

    influx_file = "data/INFLUX/FLIGHT_SEGMENT_data_2025-07-16.csv"
    df_influx = pd.read_csv(influx_file)
    df_influx['_time'] = pd.to_datetime(df_influx['_time'], format='mixed')

    # Filter to records with force data
    df_influx_forces = df_influx[df_influx['FLIGHT_SEGMENT_loadcells_force'].notna()].copy()
    df_influx_forces = df_influx_forces[df_influx_forces['FLIGHT_SEGMENT_loadcells_force'] > 0.001]
    df_influx_forces = df_influx_forces.sort_values('_time')

    time_diffs_influx = df_influx_forces['_time'].diff().dt.total_seconds()
    time_diffs_influx = time_diffs_influx.dropna()

    print(f"  Total records with force: {len(df_influx_forces)}")
    print(f"  Time span: {df_influx_forces['_time'].min()} to {df_influx_forces['_time'].max()}")
    print(f"  Duration: {(df_influx_forces['_time'].max() - df_influx_forces['_time'].min()).total_seconds():.1f} seconds")
    print(f"  Mean interval: {time_diffs_influx.mean():.3f} seconds")
    print(f"  Median interval: {time_diffs_influx.median():.3f} seconds")
    print(f"  Min interval: {time_diffs_influx.min():.3f} seconds")
    print(f"  Max interval: {time_diffs_influx.max():.3f} seconds")
    print(f"  Estimated frequency: {1/time_diffs_influx.mean():.2f} Hz")

    consistent_samples_influx = (time_diffs_influx < 0.2).sum()
    print(f"  Consistent samples (<0.2s): {consistent_samples_influx}/{len(time_diffs_influx)} ({100*consistent_samples_influx/len(time_diffs_influx):.1f}%)")

    # 3. Compare value distributions and uniqueness
    print(f"\n📊 VALUE DISTRIBUTION ANALYSIS:")
    print("-" * 40)

    # Load July 16 forces data
    forces_files_july16 = loadcell_files[:2]
    all_forces = []
    for f in forces_files_july16:
        df = pd.read_csv(f)
        df = df[df['total_force_kg'] > 0.001]
        all_forces.extend(df['total_force_kg'].values)

    forces_values = np.array(all_forces)
    influx_values = df_influx_forces['FLIGHT_SEGMENT_loadcells_force'].values

    print(f"Raspberry Pi FORCES:")
    print(f"  Total values: {len(forces_values)}")
    print(f"  Unique values: {len(np.unique(forces_values))}")
    print(f"  Value range: {forces_values.min():.2f} - {forces_values.max():.2f} kg")
    print(f"  Most common values (top 5):")
    unique_forces, counts_forces = np.unique(forces_values, return_counts=True)
    top_indices = np.argsort(counts_forces)[-5:]
    for idx in reversed(top_indices):
        print(f"    {unique_forces[idx]:.1f} kg: {counts_forces[idx]} times")

    print(f"\nInfluxDB FLIGHT_SEGMENT:")
    print(f"  Total values: {len(influx_values)}")
    print(f"  Unique values: {len(np.unique(influx_values))}")
    print(f"  Value range: {influx_values.min():.2f} - {influx_values.max():.2f} kg")
    print(f"  Most common values (top 5):")
    unique_influx, counts_influx = np.unique(influx_values, return_counts=True)
    top_indices = np.argsort(counts_influx)[-5:]
    for idx in reversed(top_indices):
        print(f"    {unique_influx[idx]:.1f} kg: {counts_influx[idx]} times")

    # 4. Check overlap in force values
    print(f"\n🔄 VALUE OVERLAP ANALYSIS:")
    print("-" * 40)

    # Round to 1 decimal for comparison
    forces_rounded = np.round(forces_values, 1)
    influx_rounded = np.round(influx_values, 1)

    forces_set = set(forces_rounded)
    influx_set = set(influx_rounded)

    overlap = forces_set.intersection(influx_set)

    print(f"Raspberry Pi unique force values (rounded): {len(forces_set)}")
    print(f"InfluxDB unique force values (rounded): {len(influx_set)}")
    print(f"Overlapping force values: {len(overlap)}")
    print(f"Overlap percentage: {100 * len(overlap) / min(len(forces_set), len(influx_set)):.1f}%")

    if len(overlap) < 100:  # If small overlap, show some examples
        print(f"Sample overlapping values: {sorted(list(overlap))[:10]}")

    # 5. Create visualization
    create_frequency_plots(forces_values, influx_values,
                          time_diffs.values if 'time_diffs' in locals() else None,
                          time_diffs_influx.values)

def create_frequency_plots(forces_values, influx_values, forces_time_diffs, influx_time_diffs):
    """Create plots showing sampling frequency analysis."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sampling Frequency and Value Distribution Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Force value distributions
    axes[0, 0].hist(forces_values, bins=50, alpha=0.7, color='blue', label='Raspberry Pi FORCES')
    axes[0, 0].hist(influx_values, bins=50, alpha=0.7, color='red', label='InfluxDB FLIGHT_SEGMENT')
    axes[0, 0].set_xlabel('Force Values (kg)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Force Value Distributions')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')

    # Plot 2: Time interval distributions
    if forces_time_diffs is not None:
        axes[0, 1].hist(forces_time_diffs[forces_time_diffs < 10], bins=50, alpha=0.7, color='blue', label='Raspberry Pi')
    axes[0, 1].hist(influx_time_diffs[influx_time_diffs < 10], bins=50, alpha=0.7, color='red', label='InfluxDB')
    axes[0, 1].set_xlabel('Time Interval (seconds)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Time Interval Distributions (< 10s)')
    axes[0, 1].legend()

    # Plot 3: Force values over time (sample)
    sample_size = min(1000, len(forces_values))
    axes[1, 0].plot(range(sample_size), forces_values[:sample_size], 'b-', alpha=0.7, label='Raspberry Pi')
    sample_size_influx = min(1000, len(influx_values))
    axes[1, 0].plot(range(sample_size_influx), influx_values[:sample_size_influx], 'r-', alpha=0.7, label='InfluxDB')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Force (kg)')
    axes[1, 0].set_title('Force Values Over Time (First 1000 samples)')
    axes[1, 0].legend()

    # Plot 4: Value uniqueness comparison
    forces_unique = len(np.unique(forces_values))
    influx_unique = len(np.unique(influx_values))

    categories = ['Raspberry Pi\nFORCES', 'InfluxDB\nFLIGHT_SEGMENT']
    total_values = [len(forces_values), len(influx_values)]
    unique_values = [forces_unique, influx_unique]

    x = np.arange(len(categories))
    width = 0.35

    axes[1, 1].bar(x - width/2, total_values, width, label='Total Values', alpha=0.7)
    axes[1, 1].bar(x + width/2, unique_values, width, label='Unique Values', alpha=0.7)
    axes[1, 1].set_xlabel('Data Source')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Total vs Unique Values')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('outputs/sampling_frequency_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Analysis plots saved to: outputs/sampling_frequency_analysis.png")
    plt.show()

if __name__ == "__main__":
    analyze_sampling_frequency()
#!/usr/bin/env python3
"""
Investigate the correct mapping between loadcell files and flight dates
by plotting total forces patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def plot_forces_investigation():
    """Plot total forces to help determine correct file mapping."""

    print("🔍 Investigating File Mapping")
    print("=" * 50)

    # Load InfluxDB data for both dates
    influx_16 = pd.read_csv("data/INFLUX/FLIGHT_SEGMENT_data_2025-07-16.csv")
    influx_17 = pd.read_csv("data/INFLUX/FLIGHT_SEGMENT_data_2025-07-17.csv")

    influx_16['_time'] = pd.to_datetime(influx_16['_time'], format='mixed')
    influx_17['_time'] = pd.to_datetime(influx_17['_time'], format='mixed')

    # Filter meaningful force data
    influx_16_forces = influx_16[influx_16['FLIGHT_SEGMENT_loadcells_force'] > 0.001].copy()
    influx_17_forces = influx_17[influx_17['FLIGHT_SEGMENT_loadcells_force'] > 0.001].copy()

    print(f"July 16 InfluxDB records with force: {len(influx_16_forces)}")
    print(f"July 17 InfluxDB records with force: {len(influx_17_forces)}")

    # Load all loadcell files
    forces_dir = Path("data/FORCES")
    loadcell_files = sorted(list(forces_dir.glob("loadcell_readings_*.csv")))

    print(f"\nLoadcell files found: {len(loadcell_files)}")
    for i, f in enumerate(loadcell_files):
        print(f"  {i+1}. {f.name}")

    # Create comprehensive comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Force Pattern Investigation: InfluxDB vs Loadcell Files', fontsize=16, fontweight='bold')

    # Plot InfluxDB force patterns
    axes[0, 0].hist(influx_16_forces['FLIGHT_SEGMENT_loadcells_force'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('July 16 InfluxDB - Total Force Distribution')
    axes[0, 0].set_xlabel('Total Force (kg)')
    axes[0, 0].set_ylabel('Count')

    axes[0, 1].hist(influx_17_forces['FLIGHT_SEGMENT_loadcells_force'], bins=50, alpha=0.7, color='red')
    axes[0, 1].set_title('July 17 InfluxDB - Total Force Distribution')
    axes[0, 1].set_xlabel('Total Force (kg)')
    axes[0, 1].set_ylabel('Count')

    # Plot loadcell file force patterns
    colors = ['orange', 'green', 'purple', 'brown', 'pink']

    for i, loadcell_file in enumerate(loadcell_files):
        print(f"\nLoading {loadcell_file.name}...")
        df = pd.read_csv(loadcell_file)
        df = df[df['total_force_kg'] > 0.001]

        print(f"  Records with meaningful force: {len(df)}")
        print(f"  Force range: {df['total_force_kg'].min():.2f} - {df['total_force_kg'].max():.2f} kg")
        print(f"  Mean force: {df['total_force_kg'].mean():.2f} kg")

        row = 1 + (i // 2)
        col = i % 2

        if row < 3:  # Only plot first 4 files
            axes[row, col].hist(df['total_force_kg'], bins=50, alpha=0.7, color=colors[i % len(colors)])
            axes[row, col].set_title(f'Loadcell File {i+1}: {loadcell_file.name}')
            axes[row, col].set_xlabel('Total Force (kg)')
            axes[row, col].set_ylabel('Count')

    # Hide unused subplot
    if len(loadcell_files) < 4:
        axes[2, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig('outputs/force_pattern_investigation.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Investigation plot saved to: outputs/force_pattern_investigation.png")
    plt.show()

    # Print summary statistics for comparison
    print(f"\n📊 Summary Statistics:")
    print(f"July 16 InfluxDB:")
    print(f"  Mean force: {influx_16_forces['FLIGHT_SEGMENT_loadcells_force'].mean():.2f} kg")
    print(f"  Force range: {influx_16_forces['FLIGHT_SEGMENT_loadcells_force'].min():.2f} - {influx_16_forces['FLIGHT_SEGMENT_loadcells_force'].max():.2f} kg")

    print(f"July 17 InfluxDB:")
    print(f"  Mean force: {influx_17_forces['FLIGHT_SEGMENT_loadcells_force'].mean():.2f} kg")
    print(f"  Force range: {influx_17_forces['FLIGHT_SEGMENT_loadcells_force'].min():.2f} - {influx_17_forces['FLIGHT_SEGMENT_loadcells_force'].max():.2f} kg")

def test_mapping_correlation():
    """Test correlation between different mappings."""

    print(f"\n🧪 Testing Different File Mappings")
    print("=" * 50)

    # Load InfluxDB data
    influx_16 = pd.read_csv("data/INFLUX/FLIGHT_SEGMENT_data_2025-07-16.csv")
    influx_17 = pd.read_csv("data/INFLUX/FLIGHT_SEGMENT_data_2025-07-17.csv")

    influx_16_forces = influx_16[influx_16['FLIGHT_SEGMENT_loadcells_force'] > 0.001]['FLIGHT_SEGMENT_loadcells_force'].values
    influx_17_forces = influx_17[influx_17['FLIGHT_SEGMENT_loadcells_force'] > 0.001]['FLIGHT_SEGMENT_loadcells_force'].values

    # Load loadcell files
    forces_dir = Path("data/FORCES")
    loadcell_files = sorted(list(forces_dir.glob("loadcell_readings_*.csv")))

    loadcell_data = []
    for f in loadcell_files:
        df = pd.read_csv(f)
        forces = df[df['total_force_kg'] > 0.001]['total_force_kg'].values
        loadcell_data.append(forces)
        print(f"{f.name}: {len(forces)} records")

    # Test your proposed mapping: first 2 for July 16, last 3 for July 17
    print(f"\n📋 Testing Your Proposed Mapping:")
    print(f"  July 16 <- Files 1,2: {loadcell_files[0].name}, {loadcell_files[1].name}")
    print(f"  July 17 <- Files 3,4,5: {loadcell_files[2].name}, {loadcell_files[3].name}, {loadcell_files[4].name}")

    # Combine forces for proposed mapping
    july16_combined = np.concatenate([loadcell_data[0], loadcell_data[1]])
    july17_combined = np.concatenate([loadcell_data[2], loadcell_data[3], loadcell_data[4]])

    print(f"\n📊 Force Statistics Comparison:")
    print(f"July 16 InfluxDB: mean={np.mean(influx_16_forces):.2f}, std={np.std(influx_16_forces):.2f}")
    print(f"July 16 Loadcells: mean={np.mean(july16_combined):.2f}, std={np.std(july16_combined):.2f}")
    print(f"July 17 InfluxDB: mean={np.mean(influx_17_forces):.2f}, std={np.std(influx_17_forces):.2f}")
    print(f"July 17 Loadcells: mean={np.mean(july17_combined):.2f}, std={np.std(july17_combined):.2f}")

if __name__ == "__main__":
    plot_forces_investigation()
    test_mapping_correlation()
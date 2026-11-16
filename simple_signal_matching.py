#!/usr/bin/env python3
"""
Simple signal matching to identify which Raspberry Pi files correspond to which flight days.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    """Simple signal matching analysis."""

    print("🚁 Simple Signal Matching Analysis")
    print("=" * 50)

    # Load InfluxDB data
    influx_16 = pd.read_csv("data/INFLUX/FLIGHT_SEGMENT_data_2025-07-16.csv")
    influx_17 = pd.read_csv("data/INFLUX/FLIGHT_SEGMENT_data_2025-07-17.csv")

    influx_16['_time'] = pd.to_datetime(influx_16['_time'], format='mixed')
    influx_17['_time'] = pd.to_datetime(influx_17['_time'], format='mixed')

    # Filter meaningful data
    influx_16_forces = influx_16[influx_16['FLIGHT_SEGMENT_loadcells_force'] > 0.001]['FLIGHT_SEGMENT_loadcells_force'].values
    influx_17_forces = influx_17[influx_17['FLIGHT_SEGMENT_loadcells_force'] > 0.001]['FLIGHT_SEGMENT_loadcells_force'].values

    print(f"InfluxDB July 16: {len(influx_16_forces)} records, mean={np.mean(influx_16_forces):.1f} kg")
    print(f"InfluxDB July 17: {len(influx_17_forces)} records, mean={np.mean(influx_17_forces):.1f} kg")

    # Load Raspberry Pi files from log_raspi2
    raspi_dir = Path("data/FORCES/log_raspi2")
    csv_files = sorted(list(raspi_dir.glob("loadcell_readings_*.csv")))

    print(f"\nAnalyzing {len(csv_files)} Raspberry Pi files...")

    candidates = []

    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            df = df[df['total_force_kg'] > 0.001]  # Filter meaningful data

            if len(df) < 100:  # Skip files with very little data
                continue

            forces = df['total_force_kg'].values
            mean_force = np.mean(forces)
            max_force = np.max(forces)
            duration = len(df)

            # Compare with July 16
            mean_diff_16 = abs(mean_force - np.mean(influx_16_forces)) / np.mean(influx_16_forces)
            max_diff_16 = abs(max_force - np.max(influx_16_forces)) / np.max(influx_16_forces)
            score_16 = -(mean_diff_16 + max_diff_16) / 2  # Higher score = better match

            # Compare with July 17
            mean_diff_17 = abs(mean_force - np.mean(influx_17_forces)) / np.mean(influx_17_forces)
            max_diff_17 = abs(max_force - np.max(influx_17_forces)) / np.max(influx_17_forces)
            score_17 = -(mean_diff_17 + max_diff_17) / 2  # Higher score = better match

            candidates.append({
                'filename': filepath.name,
                'records': len(df),
                'mean_force': mean_force,
                'max_force': max_force,
                'score_16': score_16,
                'score_17': score_17,
                'best_match': '16' if score_16 > score_17 else '17'
            })

            print(f"{filepath.name}: {len(df)} records, mean={mean_force:.1f} kg, max={max_force:.1f} kg")
            print(f"  Score vs July 16: {score_16:.3f}, Score vs July 17: {score_17:.3f}")
            print(f"  Best match: July {candidates[-1]['best_match']}")

        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
            continue

    # Sort by number of records (prioritize files with more data)
    candidates.sort(key=lambda x: x['records'], reverse=True)

    print(f"\n🎯 TOP CANDIDATES (by data volume):")
    print("=" * 50)

    july_16_files = []
    july_17_files = []

    for i, candidate in enumerate(candidates[:10]):  # Top 10 by data volume
        match_symbol = "📅" if candidate['best_match'] == '16' else "📆"
        print(f"{i+1:2d}. {match_symbol} {candidate['filename']}")
        print(f"    Records: {candidate['records']}, Mean: {candidate['mean_force']:.1f} kg, Max: {candidate['max_force']:.1f} kg")
        print(f"    Best match: July {candidate['best_match']} (16: {candidate['score_16']:.3f}, 17: {candidate['score_17']:.3f})")

        if candidate['best_match'] == '16':
            july_16_files.append(candidate['filename'])
        else:
            july_17_files.append(candidate['filename'])

    print(f"\n📋 RECOMMENDED MAPPING:")
    print("=" * 30)
    print(f"July 16 files:")
    for f in july_16_files:
        print(f"  • {f}")
    print(f"July 17 files:")
    for f in july_17_files:
        print(f"  • {f}")

    # Create a simple comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.hist(influx_16_forces, bins=50, alpha=0.7, color='blue', label='July 16 InfluxDB')
    ax1.hist(influx_17_forces, bins=50, alpha=0.7, color='red', label='July 17 InfluxDB')
    ax1.set_xlabel('Force (kg)')
    ax1.set_ylabel('Count')
    ax1.set_title('InfluxDB Force Distributions')
    ax1.legend()
    ax1.set_yscale('log')

    # Plot the top candidates
    colors = ['blue' if c['best_match'] == '16' else 'red' for c in candidates[:5]]
    labels = [f"{c['filename'][:20]}..." for c in candidates[:5]]
    means = [c['mean_force'] for c in candidates[:5]]
    maxes = [c['max_force'] for c in candidates[:5]]

    x = np.arange(len(means))
    width = 0.35

    ax2.bar(x - width/2, means, width, color=[c + '70' for c in colors], label='Mean Force')  # Add transparency
    ax2.bar(x + width/2, maxes, width, color=colors, alpha=0.7, label='Max Force')
    ax2.set_xlabel('Top Raspberry Pi Files')
    ax2.set_ylabel('Force (kg)')
    ax2.set_title('Top Raspberry Pi Files Force Statistics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('outputs/simple_signal_matching.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Comparison plot saved to: outputs/simple_signal_matching.png")
    plt.show()

if __name__ == "__main__":
    main()
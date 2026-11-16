#!/usr/bin/env python3
"""
Signal matching analysis to determine which Raspberry Pi files correspond
to July 16 vs July 17 flights based on force signal patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr
import sys

def load_raspberry_data():
    """Load all Raspberry Pi data files sequentially."""

    print("🔍 Loading Raspberry Pi Data Files")
    print("=" * 50)

    # Check for the log_raspi2 directory
    raspi_dir = Path("data/FORCES/log_raspi2")
    if not raspi_dir.exists():
        # Fallback to the original location
        raspi_dir = Path("data/FORCES")
        print(f"⚠️  log_raspi2 directory not found, using: {raspi_dir}")
    else:
        print(f"📁 Using Raspberry Pi data from: {raspi_dir}")

    # Load all CSV files
    csv_files = sorted(list(raspi_dir.glob("loadcell_readings_*.csv")))

    if len(csv_files) == 0:
        print("❌ No loadcell_readings files found!")
        return None

    print(f"Found {len(csv_files)} Raspberry Pi files:")

    raspi_data = {}

    for i, filepath in enumerate(csv_files):
        print(f"  {i+1}. {filepath.name}")

        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df = df.sort_values('datetime')

        # Filter meaningful force data
        meaningful_data = df[df['total_force_kg'] > 0.001].copy()

        raspi_data[filepath.name] = {
            'filepath': filepath,
            'data': meaningful_data,
            'total_records': len(df),
            'meaningful_records': len(meaningful_data),
            'time_start': df['datetime'].min(),
            'time_end': df['datetime'].max(),
            'duration_hours': (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600,
            'force_range': (meaningful_data['total_force_kg'].min(), meaningful_data['total_force_kg'].max()),
            'mean_force': meaningful_data['total_force_kg'].mean()
        }

        print(f"     Records: {len(meaningful_data)}/{len(df)} meaningful")
        print(f"     Duration: {raspi_data[filepath.name]['duration_hours']:.1f} hours")
        print(f"     Force range: {raspi_data[filepath.name]['force_range'][0]:.1f} - {raspi_data[filepath.name]['force_range'][1]:.1f} kg")
        print(f"     Mean force: {raspi_data[filepath.name]['mean_force']:.1f} kg")

    return raspi_data

def load_influx_data():
    """Load InfluxDB data for both flight days."""

    print(f"\n🌐 Loading InfluxDB Data Files")
    print("=" * 50)

    influx_data = {}

    for date in ["2025-07-16", "2025-07-17"]:
        filepath = Path(f"data/INFLUX/FLIGHT_SEGMENT_data_{date}.csv")

        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            continue

        print(f"Loading {date}...")

        df = pd.read_csv(filepath)
        df['_time'] = pd.to_datetime(df['_time'], format='mixed')
        df = df.sort_values('_time')

        # Filter meaningful force data
        force_data = df[df['FLIGHT_SEGMENT_loadcells_force'].notna()].copy()
        meaningful_data = force_data[force_data['FLIGHT_SEGMENT_loadcells_force'] > 0.001].copy()

        influx_data[date] = {
            'data': meaningful_data,
            'total_records': len(df),
            'force_records': len(force_data),
            'meaningful_records': len(meaningful_data),
            'time_start': meaningful_data['_time'].min(),
            'time_end': meaningful_data['_time'].max(),
            'duration_hours': (meaningful_data['_time'].max() - meaningful_data['_time'].min()).total_seconds() / 3600,
            'force_range': (meaningful_data['FLIGHT_SEGMENT_loadcells_force'].min(), meaningful_data['FLIGHT_SEGMENT_loadcells_force'].max()),
            'mean_force': meaningful_data['FLIGHT_SEGMENT_loadcells_force'].mean()
        }

        print(f"  Records: {len(meaningful_data)}/{len(df)} meaningful")
        print(f"  Duration: {influx_data[date]['duration_hours']:.1f} hours")
        print(f"  Force range: {influx_data[date]['force_range'][0]:.1f} - {influx_data[date]['force_range'][1]:.1f} kg")
        print(f"  Mean force: {influx_data[date]['mean_force']:.1f} kg")

    return influx_data

def compute_signal_correlation(raspi_data, influx_data):
    """Compute correlation between Raspberry Pi and InfluxDB signals."""

    print(f"\n🔗 Computing Signal Correlations")
    print("=" * 50)

    correlation_matrix = {}

    for raspi_filename, raspi_info in raspi_data.items():
        correlation_matrix[raspi_filename] = {}

        print(f"\nAnalyzing {raspi_filename}:")

        raspi_forces = raspi_info['data']['total_force_kg'].values

        for influx_date, influx_info in influx_data.items():
            influx_forces = influx_info['data']['FLIGHT_SEGMENT_loadcells_force'].values

            # Method 1: Direct correlation of force histograms
            raspi_hist, bins = np.histogram(raspi_forces, bins=50, density=True)
            influx_hist, _ = np.histogram(influx_forces, bins=bins, density=True)

            hist_corr, hist_p = pearsonr(raspi_hist, influx_hist)

            # Method 2: Statistical similarity
            # Compare means and standard deviations
            raspi_mean, raspi_std = np.mean(raspi_forces), np.std(raspi_forces)
            influx_mean, influx_std = np.mean(influx_forces), np.std(influx_forces)

            mean_diff = abs(raspi_mean - influx_mean) / max(raspi_mean, influx_mean)
            std_diff = abs(raspi_std - influx_std) / max(raspi_std, influx_std)

            # Method 3: Value overlap analysis
            raspi_values = set(np.round(raspi_forces, 1))
            influx_values = set(np.round(influx_forces, 1))

            overlap = len(raspi_values.intersection(influx_values))
            overlap_ratio = overlap / min(len(raspi_values), len(influx_values))

            # Method 4: Force range similarity
            raspi_range = raspi_info['force_range']
            influx_range = influx_info['force_range']

            range_similarity = 1 - abs(raspi_range[1] - influx_range[1]) / max(raspi_range[1], influx_range[1])

            # Composite score
            composite_score = (hist_corr + overlap_ratio + range_similarity - mean_diff - std_diff) / 3

            correlation_matrix[raspi_filename][influx_date] = {
                'histogram_correlation': hist_corr,
                'histogram_p_value': hist_p,
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'value_overlap_ratio': overlap_ratio,
                'range_similarity': range_similarity,
                'composite_score': composite_score
            }

            print(f"  vs {influx_date}:")
            print(f"    Histogram correlation: {hist_corr:.3f} (p={hist_p:.3f})")
            print(f"    Mean difference: {mean_diff:.3f}")
            print(f"    Std difference: {std_diff:.3f}")
            print(f"    Value overlap: {overlap_ratio:.3f}")
            print(f"    Range similarity: {range_similarity:.3f}")
            print(f"    Composite score: {composite_score:.3f}")

    return correlation_matrix

def determine_best_mapping(correlation_matrix):
    """Determine the best mapping based on correlation scores."""

    print(f"\n🎯 Determining Best File Mapping")
    print("=" * 50)

    # Create mapping recommendation
    july16_candidates = []
    july17_candidates = []

    for raspi_file, correlations in correlation_matrix.items():
        july16_score = correlations['2025-07-16']['composite_score']
        july17_score = correlations['2025-07-17']['composite_score']

        if july16_score > july17_score:
            july16_candidates.append((raspi_file, july16_score))
        else:
            july17_candidates.append((raspi_file, july17_score))

    # Sort by score
    july16_candidates.sort(key=lambda x: x[1], reverse=True)
    july17_candidates.sort(key=lambda x: x[1], reverse=True)

    print("Recommended mapping based on signal correlation:")
    print(f"\nJuly 16 (2025-07-16):")
    for i, (filename, score) in enumerate(july16_candidates):
        print(f"  {i+1}. {filename} (score: {score:.3f})")

    print(f"\nJuly 17 (2025-07-17):")
    for i, (filename, score) in enumerate(july17_candidates):
        print(f"  {i+1}. {filename} (score: {score:.3f})")

    return july16_candidates, july17_candidates

def create_signal_comparison_plots(raspi_data, influx_data, correlation_matrix):
    """Create visualization of signal patterns for comparison."""

    print(f"\n📊 Creating Signal Comparison Plots")
    print("=" * 50)

    n_raspi = len(raspi_data)
    fig, axes = plt.subplots(n_raspi + 2, 2, figsize=(16, 4 * (n_raspi + 2)))
    fig.suptitle('Signal Pattern Matching Analysis', fontsize=16, fontweight='bold')

    # Plot InfluxDB signals first
    for i, (date, info) in enumerate(influx_data.items()):
        forces = info['data']['FLIGHT_SEGMENT_loadcells_force'].values

        # Force histogram
        axes[i, 0].hist(forces, bins=50, alpha=0.7, color='red', label=f'InfluxDB {date}')
        axes[i, 0].set_title(f'InfluxDB {date} - Force Distribution')
        axes[i, 0].set_xlabel('Force (kg)')
        axes[i, 0].set_ylabel('Count')
        axes[i, 0].set_yscale('log')
        axes[i, 0].legend()

        # Force time series (sample)
        sample_size = min(5000, len(forces))
        axes[i, 1].plot(range(sample_size), forces[:sample_size], 'r-', alpha=0.7, linewidth=0.5)
        axes[i, 1].set_title(f'InfluxDB {date} - Force Time Series (sample)')
        axes[i, 1].set_xlabel('Sample Index')
        axes[i, 1].set_ylabel('Force (kg)')

    # Plot Raspberry Pi signals
    for i, (filename, info) in enumerate(raspi_data.items()):
        row = i + 2
        forces = info['data']['total_force_kg'].values

        # Force histogram
        axes[row, 0].hist(forces, bins=50, alpha=0.7, color='blue', label=f'Raspberry Pi')
        axes[row, 0].set_title(f'{filename} - Force Distribution')
        axes[row, 0].set_xlabel('Force (kg)')
        axes[row, 0].set_ylabel('Count')
        axes[row, 0].set_yscale('log')
        axes[row, 0].legend()

        # Add correlation scores as text
        july16_score = correlation_matrix[filename]['2025-07-16']['composite_score']
        july17_score = correlation_matrix[filename]['2025-07-17']['composite_score']
        axes[row, 0].text(0.7, 0.8, f'July 16 score: {july16_score:.3f}\nJuly 17 score: {july17_score:.3f}',
                         transform=axes[row, 0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

        # Force time series (sample)
        sample_size = min(5000, len(forces))
        axes[row, 1].plot(range(sample_size), forces[:sample_size], 'b-', alpha=0.7, linewidth=0.5)
        axes[row, 1].set_title(f'{filename} - Force Time Series (sample)')
        axes[row, 1].set_xlabel('Sample Index')
        axes[row, 1].set_ylabel('Force (kg)')

    plt.tight_layout()
    plt.savefig('outputs/signal_matching_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Signal comparison plots saved to: outputs/signal_matching_analysis.png")
    plt.show()

def main():
    """Main analysis function."""

    print("🚁 Raspberry Pi to InfluxDB Signal Matching Analysis")
    print("=" * 60)

    # Load all data
    raspi_data = load_raspberry_data()
    if raspi_data is None:
        return

    influx_data = load_influx_data()
    if len(influx_data) == 0:
        return

    # Compute correlations
    correlation_matrix = compute_signal_correlation(raspi_data, influx_data)

    # Determine best mapping
    july16_mapping, july17_mapping = determine_best_mapping(correlation_matrix)

    # Create visualizations
    create_signal_comparison_plots(raspi_data, influx_data, correlation_matrix)

    print(f"\n✅ Signal matching analysis complete!")
    print(f"Review the plots and correlation scores to determine the correct file mapping.")

if __name__ == "__main__":
    main()
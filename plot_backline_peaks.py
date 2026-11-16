#!/usr/bin/env python3
"""
Plot backline left and right forces from a specific loadcell file
to analyze their peak patterns and timing variations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

def plot_backline_peaks(filename="loadcell_readings_20250715_183235.csv"):
    """Plot backline left and right forces showing peak patterns."""

    print(f"📊 Analyzing Backline Forces from: {filename}")
    print("=" * 60)

    # Load the specific file
    filepath = Path("data/FORCES") / filename
    if not filepath.exists():
        # Try the log_raspi2 directory
        filepath = Path("data/FORCES/log_raspi2") / filename

    if not filepath.exists():
        print(f"❌ File not found: {filename}")
        return

    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df = df.sort_values('datetime')

    print(f"Total records: {len(df)}")
    print(f"Time span: {df['datetime'].min()} to {df['datetime'].max()}")

    duration_seconds = (df['datetime'].max() - df['datetime'].min()).total_seconds()
    print(f"Duration: {duration_seconds/3600:.2f} hours ({duration_seconds/60:.1f} minutes)")

    # Filter meaningful data
    meaningful_df = df[df['total_force_kg'] > 0.001].copy()
    print(f"Records with meaningful force: {len(meaningful_df)}")

    if len(meaningful_df) == 0:
        print("❌ No meaningful force data found!")
        return

    # Statistics
    print(f"\nBackline Left stats:")
    print(f"  Range: {meaningful_df['Backline_Left_kg'].min():.3f} to {meaningful_df['Backline_Left_kg'].max():.3f} kg")
    print(f"  Mean: {meaningful_df['Backline_Left_kg'].mean():.3f} kg")

    print(f"\nBackline Right stats:")
    print(f"  Range: {meaningful_df['Backline_Right_kg'].min():.3f} to {meaningful_df['Backline_Right_kg'].max():.3f} kg")
    print(f"  Mean: {meaningful_df['Backline_Right_kg'].mean():.3f} kg")

    # Create comprehensive plots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle(f'Backline Forces Analysis - {filename}\nPeak Patterns and Timing Variations',
                 fontsize=14, fontweight='bold')

    # Panel 1: Complete time series
    axes[0].plot(meaningful_df['datetime'], meaningful_df['Backline_Left_kg'],
                'b-', linewidth=0.5, alpha=0.8, label='Backline Left')
    axes[0].plot(meaningful_df['datetime'], meaningful_df['Backline_Right_kg'],
                'r-', linewidth=0.5, alpha=0.8, label='Backline Right')
    axes[0].set_title('Complete Backline Forces Time Series', fontweight='bold')
    axes[0].set_ylabel('Force (kg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    # Panel 2: Zoom into first 10 minutes to see details
    time_window = meaningful_df['datetime'].min() + pd.Timedelta(minutes=10)
    zoom_df = meaningful_df[meaningful_df['datetime'] <= time_window]

    if len(zoom_df) > 0:
        axes[1].plot(zoom_df['datetime'], zoom_df['Backline_Left_kg'],
                    'b-', linewidth=1, alpha=0.8, label='Backline Left')
        axes[1].plot(zoom_df['datetime'], zoom_df['Backline_Right_kg'],
                    'r-', linewidth=1, alpha=0.8, label='Backline Right')
        axes[1].set_title('First 10 Minutes - Detail View of Peak Patterns', fontweight='bold')
        axes[1].set_ylabel('Force (kg)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    # Panel 3: Scatter plot to show correlation/relationship
    axes[2].scatter(meaningful_df['Backline_Left_kg'], meaningful_df['Backline_Right_kg'],
                   alpha=0.3, s=1, c=range(len(meaningful_df)), cmap='viridis')
    axes[2].set_xlabel('Backline Left (kg)')
    axes[2].set_ylabel('Backline Right (kg)')
    axes[2].set_title('Backline Left vs Right Correlation (colored by time progression)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # Add diagonal line for reference
    max_force = max(meaningful_df['Backline_Left_kg'].max(), meaningful_df['Backline_Right_kg'].max())
    min_force = min(meaningful_df['Backline_Left_kg'].min(), meaningful_df['Backline_Right_kg'].min())
    axes[2].plot([min_force, max_force], [min_force, max_force], 'k--', alpha=0.5, label='Equal forces line')
    axes[2].legend()

    # Panel 4: Difference between left and right (timing analysis)
    meaningful_df['backline_diff'] = meaningful_df['Backline_Left_kg'] - meaningful_df['Backline_Right_kg']
    axes[3].plot(meaningful_df['datetime'], meaningful_df['backline_diff'],
                'g-', linewidth=0.5, alpha=0.8, label='Left - Right')
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[3].set_title('Backline Force Difference (Left - Right) - Shows Timing Variations', fontweight='bold')
    axes[3].set_ylabel('Force Difference (kg)')
    axes[3].set_xlabel('Time')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    # Format x-axes
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_fontsize(9)

    plt.tight_layout()

    # Save plot
    output_path = f"outputs/backline_peaks_analysis_{filename.replace('.csv', '.png')}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Backline peaks analysis saved to: {output_path}")
    plt.show()

    # Additional analysis
    print(f"\n🔍 Peak Analysis:")

    # Find peaks (simple threshold-based)
    left_threshold = meaningful_df['Backline_Left_kg'].quantile(0.95)  # Top 5%
    right_threshold = meaningful_df['Backline_Right_kg'].quantile(0.95)

    left_peaks = meaningful_df[meaningful_df['Backline_Left_kg'] > left_threshold]
    right_peaks = meaningful_df[meaningful_df['Backline_Right_kg'] > right_threshold]

    print(f"Left peaks (top 5%): {len(left_peaks)} events above {left_threshold:.3f} kg")
    print(f"Right peaks (top 5%): {len(right_peaks)} events above {right_threshold:.3f} kg")

    # Timing variation analysis
    diff_stats = meaningful_df['backline_diff'].describe()
    print(f"\nTiming Variation Analysis (Left - Right):")
    print(f"  Mean difference: {diff_stats['mean']:.3f} kg")
    print(f"  Std deviation: {diff_stats['std']:.3f} kg")
    print(f"  Range: {diff_stats['min']:.3f} to {diff_stats['max']:.3f} kg")

    # Check for systematic timing differences
    positive_diff = (meaningful_df['backline_diff'] > 0.1).sum()
    negative_diff = (meaningful_df['backline_diff'] < -0.1).sum()
    print(f"  Left > Right (>0.1kg): {positive_diff} records ({100*positive_diff/len(meaningful_df):.1f}%)")
    print(f"  Right > Left (<-0.1kg): {negative_diff} records ({100*negative_diff/len(meaningful_df):.1f}%)")

def main():
    """Main function."""

    print("🚁 Backline Peak Analysis Tool")
    print("=" * 50)

    # Analyze the specific file requested
    plot_backline_peaks("loadcell_readings_20250715_183235.csv")

if __name__ == "__main__":
    main()
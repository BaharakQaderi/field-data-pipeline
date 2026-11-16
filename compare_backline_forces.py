#!/usr/bin/env python3
"""
Compare backline forces between original Raspberry Pi data and merged data
to validate that timing variations are preserved.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

def load_original_forces_data(date="2025-07-16"):
    """Load original FORCES data for a specific date."""

    forces_dir = Path("data/FORCES")

    # Correct mapping based on force pattern analysis:
    if date == "2025-07-16":
        files = ["loadcell_readings_20250715_161829.csv",
                "loadcell_readings_20250715_183235.csv"]
    elif date == "2025-07-17":
        files = ["loadcell_readings_20250715_211757.csv",
                "loadcell_readings_20250715_224238.csv",
                "loadcell_readings_20250716_003735.csv"]
    else:
        raise ValueError(f"Unsupported date: {date}")

    # Load and combine all FORCES files
    all_forces = []
    for filename in files:
        filepath = forces_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
            df = df[df['total_force_kg'] > 0.001]  # Filter meaningful data
            df['source_file'] = filename
            all_forces.append(df)
            print(f"Loaded {len(df)} records from {filename}")

    combined_df = pd.concat(all_forces, ignore_index=True)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

    print(f"Total original FORCES records for {date}: {len(combined_df)}")
    return combined_df

def load_merged_data(date="2025-07-16"):
    """Load merged data for a specific date."""

    filepath = Path(f"outputs/merged_flight_data_{date}.csv")
    if not filepath.exists():
        raise FileNotFoundError(f"Merged data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df['_time'] = pd.to_datetime(df['_time'], format='mixed')

    # Filter to only matched records with meaningful force data
    matched_df = df[df['forces_matched'] == True].copy()
    matched_df = matched_df[matched_df['FLIGHT_SEGMENT_loadcells_force'] > 0.001]

    print(f"Total matched records for {date}: {len(matched_df)}")
    return matched_df

def plot_backline_comparison(date="2025-07-16", time_window_minutes=30):
    """
    Create a comparison plot of backline forces.

    Args:
        date: Date to analyze
        time_window_minutes: Time window to plot (in minutes from start)
    """

    print(f"🔍 Creating backline forces comparison for {date}")
    print("=" * 60)

    # Load data
    original_df = load_original_forces_data(date)
    merged_df = load_merged_data(date)

    # Get time range for plotting
    start_time_orig = original_df['datetime'].min()
    end_time_orig = start_time_orig + pd.Timedelta(minutes=time_window_minutes)

    start_time_merged = merged_df['_time'].min()
    end_time_merged = start_time_merged + pd.Timedelta(minutes=time_window_minutes)

    # Filter to time window
    orig_window = original_df[
        (original_df['datetime'] >= start_time_orig) &
        (original_df['datetime'] <= end_time_orig)
    ].copy()

    merged_window = merged_df[
        (merged_df['_time'] >= start_time_merged) &
        (merged_df['_time'] <= end_time_merged)
    ].copy()

    print(f"Original data window: {len(orig_window)} records")
    print(f"Merged data window: {len(merged_window)} records")

    # Create the comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'Backline Forces Comparison - {date}\n(First {time_window_minutes} minutes)',
                 fontsize=16, fontweight='bold')

    # Panel 1: Original Raspberry Pi Data
    ax1.plot(orig_window['datetime'], orig_window['Backline_Left_kg'],
             'b-', linewidth=1, alpha=0.8, label='Backline Left')
    ax1.plot(orig_window['datetime'], orig_window['Backline_Right_kg'],
             'r-', linewidth=1, alpha=0.8, label='Backline Right')

    ax1.set_title('Original Raspberry Pi Data (with timing variations)', fontweight='bold')
    ax1.set_ylabel('Force (kg)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Panel 2: Merged Data (Fixed)
    ax2.plot(merged_window['_time'], merged_window['Backline_Left_kg'],
             'b-', linewidth=1, alpha=0.8, label='Backline Left')
    ax2.plot(merged_window['_time'], merged_window['Backline_Right_kg'],
             'r-', linewidth=1, alpha=0.8, label='Backline Right')

    ax2.set_title('Merged Data (with timing fix applied)', fontweight='bold')
    ax2.set_ylabel('Force (kg)')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Add statistical comparison
    print(f"\n📊 Statistical Comparison:")
    print(f"Original Data:")
    print(f"  Backline Left  - Mean: {orig_window['Backline_Left_kg'].mean():.3f}, Std: {orig_window['Backline_Left_kg'].std():.3f}")
    print(f"  Backline Right - Mean: {orig_window['Backline_Right_kg'].mean():.3f}, Std: {orig_window['Backline_Right_kg'].std():.3f}")
    print(f"Merged Data:")
    print(f"  Backline Left  - Mean: {merged_window['Backline_Left_kg'].mean():.3f}, Std: {merged_window['Backline_Left_kg'].std():.3f}")
    print(f"  Backline Right - Mean: {merged_window['Backline_Right_kg'].mean():.3f}, Std: {merged_window['Backline_Right_kg'].std():.3f}")

    # Check for timing variations (should see different values at consecutive times)
    orig_left_diff = orig_window['Backline_Left_kg'].diff().abs().mean()
    orig_right_diff = orig_window['Backline_Right_kg'].diff().abs().mean()
    merged_left_diff = merged_window['Backline_Left_kg'].diff().abs().mean()
    merged_right_diff = merged_window['Backline_Right_kg'].diff().abs().mean()

    print(f"\n🔄 Timing Variation Analysis:")
    print(f"Original - Mean absolute differences between consecutive readings:")
    print(f"  Backline Left: {orig_left_diff:.6f} kg")
    print(f"  Backline Right: {orig_right_diff:.6f} kg")
    print(f"Merged - Mean absolute differences between consecutive readings:")
    print(f"  Backline Left: {merged_left_diff:.6f} kg")
    print(f"  Backline Right: {merged_right_diff:.6f} kg")

    if merged_left_diff > 0.001 and merged_right_diff > 0.001:
        print("✅ GOOD: Merged data shows timing variations (fix working!)")
    else:
        print("⚠️  WARNING: Merged data may still have timing issues")

    plt.tight_layout()

    # Save the plot
    output_path = Path("outputs") / f"backline_forces_comparison_{date}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Plot saved to: {output_path}")

    plt.show()

    return orig_window, merged_window

def main():
    """Main function to run the comparison."""

    # You can modify these parameters
    date = "2025-07-16"  # or "2025-07-17"
    time_window = 120 # minutes

    print(f"🚁 Backline Forces Comparison Tool")
    print(f"   Date: {date}")
    print(f"   Time window: {time_window} minutes")

    try:
        orig_data, merged_data = plot_backline_comparison(date, time_window)

        print(f"\n🎯 Comparison completed successfully!")
        print(f"   You can now visually inspect the plots to see if timing variations match.")

        # Also create a plot for July 17 if July 16 was processed
        if date == "2025-07-16":
            print(f"\n🔄 Processing July 17 data as well...")
            plot_backline_comparison("2025-07-17", time_window)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
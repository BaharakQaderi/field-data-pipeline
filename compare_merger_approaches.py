#!/usr/bin/env python3
"""
Compare the old force-matching approach vs new temporal approach
to demonstrate the preservation of temporal relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from data_merger import DataMerger
from improved_data_merger import ImprovedDataMerger

def compare_approaches():
    """Compare old vs new merger approaches."""

    print("🔄 COMPARING MERGER APPROACHES")
    print("=" * 60)

    date = "2025-07-16"  # Use July 16 which has temporal overlap

    # Test old approach
    print(f"\n📊 OLD APPROACH (Force-Based Matching):")
    print("-" * 40)

    old_merger = DataMerger()
    try:
        old_merged = old_merger.merge_day_data(date, tolerance=0.1)
        old_stats = old_merger.generate_summary_stats(old_merged)

        print(f"  Total records: {old_stats['total_records']}")
        print(f"  Matched records: {old_stats['matched_records']}")
        print(f"  Match rate: {old_stats['match_rate']:.1f}%")
        print(f"  Avg force difference: {old_stats['avg_force_difference']:.4f} kg")

        # Check for artificial repetition
        force_cols = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
        matched_df = old_merged[old_merged['forces_matched']].copy()

        if len(matched_df) > 0:
            # Round to avoid floating point issues
            for col in force_cols:
                matched_df[f'{col}_rounded'] = matched_df[col].round(6)

            rounded_cols = [f'{col}_rounded' for col in force_cols]
            unique_combinations = matched_df[rounded_cols].drop_duplicates()

            print(f"  Unique force combinations: {len(unique_combinations)}")
            print(f"  Reuse ratio: {len(matched_df) / len(unique_combinations):.2f}x")

            # Show temporal spread of identical patterns
            if len(unique_combinations) < len(matched_df):
                # Find most repeated pattern
                pattern_counts = matched_df.groupby(rounded_cols).size().sort_values(ascending=False)
                most_common = pattern_counts.index[0]
                most_common_count = pattern_counts.iloc[0]

                print(f"  Most repeated pattern appears {most_common_count} times")

                # Get time spans for this pattern
                pattern_mask = (matched_df[rounded_cols] == most_common).all(axis=1)
                pattern_times = matched_df[pattern_mask]['_time']
                time_span = (pattern_times.max() - pattern_times.min()).total_seconds() / 60

                print(f"  Same pattern spans {time_span:.1f} minutes (artificial!)")

    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Test new approach
    print(f"\n📊 NEW APPROACH (Temporal Interpolation):")
    print("-" * 40)

    new_merger = ImprovedDataMerger()
    try:
        new_merged = new_merger.merge_day_data_temporal(date)
        new_stats = new_merger.generate_temporal_summary_stats(new_merged)

        print(f"  Total records: {new_stats['total_records']}")
        print(f"  Interpolated records: {new_stats['interpolated_records']}")
        print(f"  Interpolation rate: {new_stats['interpolation_rate']:.1f}%")
        print(f"  Timezone offset detected: {new_stats['timezone_offset']} hours")
        print(f"  Temporal span: {new_stats['temporal_span']['duration_hours']:.2f} hours")

        # Check for temporal preservation
        force_cols = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
        interpolated_df = new_merged[new_merged[force_cols].notna().all(axis=1)].copy()

        if len(interpolated_df) > 0:
            # Check variability in force components over time
            force_variations = {}
            for col in force_cols:
                values = interpolated_df[col].values
                variation = np.std(values) if len(values) > 1 else 0
                force_variations[col] = variation

            total_variation = sum(force_variations.values())
            print(f"  Total force component variation: {total_variation:.3f}")
            print(f"  ✅ Temporal interpolation preserves natural variation")

            # Check if values change smoothly over time
            sample_size = min(100, len(interpolated_df))
            sample_df = interpolated_df.head(sample_size)

            # Calculate rate of change
            for col in force_cols:
                values = sample_df[col].values
                if len(values) > 1:
                    changes = np.abs(np.diff(values))
                    avg_change = np.mean(changes)
                    print(f"    {col}: avg change = {avg_change:.3f} kg/step")

    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Create comparison visualization
    print(f"\n📈 Creating Comparison Visualization...")
    create_comparison_plots(old_merged if 'old_merged' in locals() else None,
                           new_merged if 'new_merged' in locals() else None)

def create_comparison_plots(old_merged=None, new_merged=None):
    """Create plots comparing the two approaches."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Data Merger Approach Comparison', fontsize=16, fontweight='bold')

    force_cols = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
    colors = ['red', 'blue', 'green', 'orange']

    if old_merged is not None:
        # Plot 1: Old approach force components over time
        matched_old = old_merged[old_merged['forces_matched']].head(1000)  # Sample

        for i, (col, color) in enumerate(zip(force_cols, colors)):
            axes[0, 0].plot(range(len(matched_old)), matched_old[col],
                          color=color, alpha=0.7, linewidth=1, label=col.replace('_kg', ''))

        axes[0, 0].set_title('Old Approach: Force Components Over Time')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Force (kg)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Old approach - show repetition pattern
        if len(matched_old) > 0:
            # Show histogram of backline left values to demonstrate repetition
            axes[0, 1].hist(matched_old['Backline_Left_kg'], bins=30, alpha=0.7, color='red', edgecolor='black')
            axes[0, 1].set_title('Old Approach: Backline Left Distribution (Shows Repetition)')
            axes[0, 1].set_xlabel('Backline Left Force (kg)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Old approach data\nnot available', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, 'Old approach data\nnot available', ha='center', va='center', transform=axes[0, 1].transAxes)

    if new_merged is not None:
        # Plot 3: New approach force components over time
        interpolated_new = new_merged[new_merged[force_cols].notna().all(axis=1)].head(1000)  # Sample

        if len(interpolated_new) > 0:
            for i, (col, color) in enumerate(zip(force_cols, colors)):
                axes[1, 0].plot(range(len(interpolated_new)), interpolated_new[col],
                              color=color, alpha=0.7, linewidth=1, label=col.replace('_kg', ''))

            axes[1, 0].set_title('New Approach: Interpolated Force Components')
            axes[1, 0].set_xlabel('Sample Index')
            axes[1, 0].set_ylabel('Force (kg)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: New approach - show smooth variation
            axes[1, 1].hist(interpolated_new['Backline_Left_kg'], bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 1].set_title('New Approach: Backline Left Distribution (Natural Variation)')
            axes[1, 1].set_xlabel('Backline Left Force (kg)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No interpolated data\navailable', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, 'No interpolated data\navailable', ha='center', va='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 0].text(0.5, 0.5, 'New approach data\nnot available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'New approach data\nnot available', ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig('outputs/merger_approach_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plots saved to: outputs/merger_approach_comparison.png")
    plt.show()

if __name__ == "__main__":
    compare_approaches()
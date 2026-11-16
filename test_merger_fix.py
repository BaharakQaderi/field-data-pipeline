#!/usr/bin/env python3
"""
Test script to validate the data merger fix on a small sample.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')
from src.data_merger import DataMerger

def test_small_sample():
    """Test the merger fix on a small data sample."""

    print("🧪 Testing Data Merger Fix on Small Sample")
    print("=" * 50)

    merger = DataMerger()

    # Test with July 16 data (smaller sample)
    date = "2025-07-16"

    try:
        print(f"📊 Loading InfluxDB data for {date}...")
        influx_df = merger.load_influx_data(date)

        # Take only first 50 records for testing
        influx_sample = influx_df.head(50).copy()
        print(f"   Sample size: {len(influx_sample)} records")

        print(f"📂 Loading FORCES data for {date}...")
        forces_files = merger.get_forces_files_for_date(date)
        print(f"   Files: {[f.name for f in forces_files]}")

        # Load and combine FORCES data
        all_forces_data = []
        for forces_file in forces_files:
            forces_df = merger.load_forces_data(forces_file)
            forces_df['source_file'] = forces_file.name
            all_forces_data.append(forces_df)

        combined_forces_df = pd.concat(all_forces_data, ignore_index=True)
        print(f"   Combined FORCES records: {len(combined_forces_df)}")

        print(f"🔄 Running alignment with fix...")
        merged_df = merger.align_forces_with_influx(influx_sample, combined_forces_df)

        # Analyze results
        print(f"\n📈 Results Analysis:")
        print(f"   Total records: {len(merged_df)}")
        print(f"   Matched records: {merged_df['forces_matched'].sum()}")
        print(f"   Match rate: {merged_df['forces_matched'].mean() * 100:.1f}%")

        # Check for duplicated force patterns (the main issue we're fixing)
        matched_df = merged_df[merged_df['forces_matched']].copy()

        if len(matched_df) > 0:
            # Count unique combinations of force components
            force_cols = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']

            # Round to avoid floating point precision issues
            for col in force_cols:
                matched_df[f'{col}_rounded'] = matched_df[col].round(6)

            rounded_cols = [f'{col}_rounded' for col in force_cols]
            unique_combinations = matched_df[rounded_cols].drop_duplicates()

            print(f"   Unique force combinations: {len(unique_combinations)}")
            print(f"   Matched records: {len(matched_df)}")
            print(f"   Reuse ratio: {len(matched_df) / len(unique_combinations):.2f}")

            # Show sample of results
            print(f"\n📋 Sample Results (first 10 matched records):")
            display_cols = ['_time', 'FLIGHT_SEGMENT_loadcells_force'] + force_cols + ['force_difference']
            sample_results = matched_df[display_cols].head(10)

            for i, (idx, row) in enumerate(sample_results.iterrows()):
                time_str = str(row['_time'])[-15:]  # Last 15 chars of timestamp
                print(f"   {i+1:2d}. {time_str} | Total: {row['FLIGHT_SEGMENT_loadcells_force']:4.1f} | "
                      f"BL_L: {row['Backline_Left_kg']:6.3f} | BL_R: {row['Backline_Right_kg']:6.3f} | "
                      f"5th: {row['5th_line_kg']:6.3f} | FL: {row['Frontline_kg']:6.3f}")

            # Check if the fix worked (look for variation in force components)
            if len(unique_combinations) == len(matched_df):
                print(f"\n✅ SUCCESS: All matched records have unique force combinations!")
                print(f"   The timing variation issue has been fixed.")
            elif len(unique_combinations) > 1:
                print(f"\n⚠️  PARTIAL SUCCESS: Some unique combinations, but still some reuse.")
                print(f"   This might be expected if FORCES data has fewer records than InfluxDB.")
            else:
                print(f"\n❌ ISSUE: Only one unique combination found - fix may not be working.")

        # Save test results
        output_path = Path("outputs") / f"test_merged_sample_{date}.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"\n💾 Test results saved to: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    test_small_sample()
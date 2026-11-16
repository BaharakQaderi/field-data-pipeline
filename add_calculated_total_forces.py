import pandas as pd
import os
import glob

def add_calculated_total_forces():
    """Add calculated_total_forces column to all individual flight segment CSV files"""

    print("Adding calculated_total_forces column to individual flight segment files...")
    print("="*70)

    # Find all CSV files in subdirectories
    csv_files = glob.glob('*/FLIGHT_SEGMENT_data_*.csv')

    if not csv_files:
        print("No CSV files found in subdirectories!")
        return

    print(f"Found {len(csv_files)} files to process")

    for i, csv_file in enumerate(sorted(csv_files), 1):
        print(f"\n[{i}/{len(csv_files)}] Processing: {csv_file}")

        try:
            # Load the CSV file
            df = pd.read_csv(csv_file)
            print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

            # Check if required columns exist
            required_columns = ['FLIGHT_SEGMENT_loadcells_force', 'gs_5th_line_kg']

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"  ✗ Missing columns: {missing_columns}")
                continue

            # Calculate calculated_total_forces
            df['calculated_total_forces'] = df['FLIGHT_SEGMENT_loadcells_force'] - df['gs_5th_line_kg']

            print(f"  ✓ Added calculated_total_forces column")
            print(f"    Formula: FLIGHT_SEGMENT_loadcells_force - gs_5th_line_kg")

            # Show example calculation
            example_idx = 0
            flight_seg_force = df['FLIGHT_SEGMENT_loadcells_force'].iloc[example_idx]
            gs_5th_line = df['gs_5th_line_kg'].iloc[example_idx]
            calculated_total = df['calculated_total_forces'].iloc[example_idx]

            print(f"    Example: {flight_seg_force:.3f} - {gs_5th_line:.3f} = {calculated_total:.3f} kg")
            print(f"    Min: {df['calculated_total_forces'].min():.3f} kg, Max: {df['calculated_total_forces'].max():.3f} kg")

            # Save back to the same file
            df.to_csv(csv_file, index=False)
            print(f"  ✓ Saved updated file with {len(df.columns)} columns")

        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {str(e)}")

    print(f"\n{'='*70}")
    print("calculated_total_forces column addition complete!")

if __name__ == "__main__":
    add_calculated_total_forces()
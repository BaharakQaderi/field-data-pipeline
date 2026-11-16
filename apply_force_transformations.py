import pandas as pd
import os
import glob

def apply_force_transformations():
    """Apply force transformations to all individual flight segment CSV files"""

    print("Applying force transformations to individual flight segment files...")
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
            print(f"  Loaded {len(df):,} rows")

            # Check if required columns exist
            required_columns = ['gs_Backline_Left_kg', 'gs_Backline_Right_kg',
                              'gs_5th_line_kg', 'gs_Frontline_kg']

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"  ✗ Missing columns: {missing_columns}")
                continue

            # Transformation 1: Divide gs_5th_line_kg by 2
            original_5th_line = df['gs_5th_line_kg'].copy()
            df['gs_5th_line_kg'] = df['gs_5th_line_kg'] / 2
            print(f"  ✓ Divided gs_5th_line_kg by 2")
            print(f"    Example: {original_5th_line.iloc[0]:.3f} → {df['gs_5th_line_kg'].iloc[0]:.3f}")

            # Transformation 2: Calculate gs_total_force
            df['gs_total_force'] = (df['gs_Backline_Left_kg'] +
                                   df['gs_Backline_Right_kg'] +
                                   df['gs_5th_line_kg'] +
                                   df['gs_Frontline_kg'])

            print(f"  ✓ Added gs_total_force column")
            print(f"    Example total force: {df['gs_total_force'].iloc[0]:.3f} kg")
            print(f"    Min: {df['gs_total_force'].min():.3f} kg, Max: {df['gs_total_force'].max():.3f} kg")

            # Save back to the same file
            df.to_csv(csv_file, index=False)
            print(f"  ✓ Saved updated file with {len(df.columns)} columns")

        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {str(e)}")

    print(f"\n{'='*70}")
    print("Force transformations complete!")

if __name__ == "__main__":
    apply_force_transformations()
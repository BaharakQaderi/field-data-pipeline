import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

def create_individual_flight_segments():
    """Create individual FLIGHT_SEGMENT CSV files for each aligned flight"""

    # Process July 16th flights
    july16_aligned_files = glob.glob('aligned_flights/flight_*_aligned_data.csv')
    july16_gs_file = '/Users/baharakqaderi/field-data-pipeline/data/INFLUX/FLIGHT_SEGMENT_data_2025-07-16.csv'

    # Process July 17th flights
    july17_aligned_files = glob.glob('aligned_flights_july17/july17_flight_*_aligned_data.csv')
    july17_gs_file = '/Users/baharakqaderi/field-data-pipeline/data/INFLUX/FLIGHT_SEGMENT_data_2025-07-17.csv'

    # Create output directory
    output_dir = 'individual_flight_segments'
    os.makedirs(output_dir, exist_ok=True)

    def process_flights(aligned_files, gs_file, date_prefix):
        print(f"\n{'='*60}")
        print(f"PROCESSING {date_prefix.upper()} FLIGHTS")
        print(f"{'='*60}")

        # Load the original GS data to get all columns
        print(f"Loading original GS data: {gs_file}")
        gs_data = pd.read_csv(gs_file)
        gs_data['_time'] = pd.to_datetime(gs_data['_time'], format='mixed')
        print(f"Original GS data columns: {list(gs_data.columns)}")

        for aligned_file in sorted(aligned_files):
            print(f"\nProcessing: {os.path.basename(aligned_file)}")

            # Load aligned data
            aligned_data = pd.read_csv(aligned_file)

            # Extract start time from gs_time column (first timestamp)
            start_time = pd.to_datetime(aligned_data['gs_time'].iloc[0])
            start_time_str = start_time.strftime('%Y_%m_%d_%H_%M')

            # Create output filename
            output_filename = f'FLIGHT_SEGMENT_data_{start_time_str}.csv'
            output_path = os.path.join(output_dir, output_filename)

            # Get time range for this flight
            start_ts = pd.to_datetime(aligned_data['gs_time'].iloc[0])
            end_ts = pd.to_datetime(aligned_data['gs_time'].iloc[-1])

            # Filter original GS data for this flight's time range
            flight_gs_mask = (gs_data['_time'] >= start_ts) & (gs_data['_time'] <= end_ts)
            original_flight_gs = gs_data[flight_gs_mask].copy()

            print(f"    Original flight data points: {len(original_flight_gs):,}")
            print(f"    Aligned data points: {len(aligned_data):,}")

            # Create base dataframe with aligned timestamps
            flight_segment_data = pd.DataFrame({
                '_time': aligned_data['gs_time']
            })

            # Add all original GS columns by interpolating them to aligned timestamps
            for col in gs_data.columns:
                if col == '_time':
                    continue

                if col == 'FLIGHT_SEGMENT_loadcells_force':
                    # Use our interpolated total force
                    flight_segment_data[col] = aligned_data['gs_total_force']
                elif col == 'FLIGHT_SEGMENT_roll_percentage':
                    # Use our interpolated roll percentage
                    flight_segment_data[col] = aligned_data['gs_roll_percentage']
                else:
                    # Interpolate other fields from original data
                    if len(original_flight_gs) > 1:
                        valid_mask = ~original_flight_gs[col].isna()
                        if valid_mask.sum() > 1:
                            from scipy import interpolate
                            # Convert timestamps to numeric for interpolation
                            original_ts_numeric = original_flight_gs.loc[valid_mask, '_time'].astype('int64')
                            aligned_ts_numeric = pd.to_datetime(aligned_data['gs_time'], format='mixed').astype('int64')

                            interp_func = interpolate.interp1d(
                                original_ts_numeric,
                                original_flight_gs.loc[valid_mask, col],
                                kind='linear',
                                bounds_error=False,
                                fill_value=np.nan
                            )
                            flight_segment_data[col] = interp_func(aligned_ts_numeric)
                        else:
                            flight_segment_data[col] = np.nan
                    else:
                        flight_segment_data[col] = np.nan

            # Add the interpolated force components with gs_ prefix
            flight_segment_data['gs_Backline_Left_kg'] = aligned_data['gs_Backline_Left_kg']
            flight_segment_data['gs_Backline_Right_kg'] = aligned_data['gs_Backline_Right_kg']
            flight_segment_data['gs_5th_line_kg'] = aligned_data['gs_5th_line_kg']
            flight_segment_data['gs_Frontline_kg'] = aligned_data['gs_Frontline_kg']

            # Add unified time for reference
            flight_segment_data['unified_time'] = aligned_data['unified_time']

            # Save the individual flight segment
            flight_segment_data.to_csv(output_path, index=False)

            print(f"  ✓ Saved: {output_filename}")
            print(f"    Time range: {start_ts} to {end_ts}")
            print(f"    Data points: {len(flight_segment_data):,}")
            print(f"    Total columns: {len(flight_segment_data.columns)}")

    # Process both days
    if july16_aligned_files:
        process_flights(july16_aligned_files, july16_gs_file, '2025-07-16')

    if july17_aligned_files:
        process_flights(july17_aligned_files, july17_gs_file, '2025-07-17')

    # Create summary
    created_files = glob.glob(f'{output_dir}/FLIGHT_SEGMENT_data_*.csv')

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total individual flight segments created: {len(created_files)}")
    print(f"Output directory: {output_dir}/")
    print("\nCreated files:")
    for file_path in sorted(created_files):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"  {filename} ({file_size:.2f} MB)")

    return created_files

if __name__ == "__main__":
    created_files = create_individual_flight_segments()
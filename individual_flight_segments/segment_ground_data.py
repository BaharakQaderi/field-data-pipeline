import pandas as pd
import os
from datetime import datetime

def segment_ground_data():
    """Segment ground segment data according to flight time ranges and place in flight directories"""

    print("Segmenting ground segment data by flight time ranges...")
    print("="*70)

    # Flight time ranges for July 16th, 2025
    july16_flights = [
        {"start": "2025-07-16 10:29:25", "end": "2025-07-16 10:32:59", "dir": "2025-07-16-10-29"},
        {"start": "2025-07-16 10:58:01", "end": "2025-07-16 11:02:16", "dir": "2025-07-16-10-58"},
        {"start": "2025-07-16 11:47:46", "end": "2025-07-16 12:18:40", "dir": "2025-07-16-11-47"},
        {"start": "2025-07-16 13:46:41", "end": "2025-07-16 13:51:33", "dir": "2025-07-16-13-46"},
        {"start": "2025-07-16 14:00:25", "end": "2025-07-16 14:03:27", "dir": "2025-07-16-14-00"},
        {"start": "2025-07-16 14:37:58", "end": "2025-07-16 14:46:57", "dir": "2025-07-16-14-37"}
    ]

    # Flight time ranges for July 17th, 2025
    july17_flights = [
        {"start": "2025-07-17 10:15:08", "end": "2025-07-17 10:23:42", "dir": "2025-07-17-10-15"},
        {"start": "2025-07-17 10:37:37", "end": "2025-07-17 10:47:17", "dir": "2025-07-17-10-37"},
        {"start": "2025-07-17 10:54:17", "end": "2025-07-17 11:15:08", "dir": "2025-07-17-10-54"},
        {"start": "2025-07-17 11:30:35", "end": "2025-07-17 11:50:12", "dir": "2025-07-17-11-30"},
        {"start": "2025-07-17 12:34:24", "end": "2025-07-17 12:42:04", "dir": "2025-07-17-12-34"},
        {"start": "2025-07-17 12:45:20", "end": "2025-07-17 12:49:25", "dir": "2025-07-17-12-45"},
        {"start": "2025-07-17 12:51:41", "end": "2025-07-17 12:55:51", "dir": "2025-07-17-12-51"},
        {"start": "2025-07-17 12:59:20", "end": "2025-07-17 13:02:56", "dir": "2025-07-17-12-59"},
        {"start": "2025-07-17 13:08:09", "end": "2025-07-17 13:10:47", "dir": "2025-07-17-13-08"},
        {"start": "2025-07-17 13:15:52", "end": "2025-07-17 13:20:25", "dir": "2025-07-17-13-15"},
        {"start": "2025-07-17 13:24:51", "end": "2025-07-17 13:26:15", "dir": "2025-07-17-13-24"},
        {"start": "2025-07-17 13:31:12", "end": "2025-07-17 13:32:43", "dir": "2025-07-17-13-31"},
        {"start": "2025-07-17 13:32:55", "end": "2025-07-17 13:33:54", "dir": "2025-07-17-13-32"},
        {"start": "2025-07-17 13:35:45", "end": "2025-07-17 13:38:03", "dir": "2025-07-17-13-35"},
        {"start": "2025-07-17 13:39:23", "end": "2025-07-17 13:46:51", "dir": "2025-07-17-13-39"}
    ]

    def process_ground_data(gs_file, flights, date_str):
        """Process ground segment data for a specific date"""
        print(f"\nProcessing {date_str} ground segment data...")
        print(f"Loading: {gs_file}")

        try:
            # Load ground segment data
            gs_data = pd.read_csv(gs_file)
            gs_data['_time'] = pd.to_datetime(gs_data['_time'], format='mixed')
            print(f"Loaded {len(gs_data):,} records with {len(gs_data.columns)} columns")

            for i, flight in enumerate(flights, 1):
                print(f"\n  [{i}/{len(flights)}] Processing flight: {flight['dir']}")

                # Parse time ranges and ensure timezone compatibility
                start_time = pd.to_datetime(flight['start'])
                end_time = pd.to_datetime(flight['end'])

                # Make start and end times timezone-aware to match gs_data
                if gs_data['_time'].dt.tz is not None:
                    start_time = start_time.tz_localize('UTC')
                    end_time = end_time.tz_localize('UTC')

                print(f"    Time range: {start_time} to {end_time}")

                # Filter ground segment data for this flight's time range
                flight_mask = (gs_data['_time'] >= start_time) & (gs_data['_time'] <= end_time)
                flight_gs_data = gs_data[flight_mask].copy()

                print(f"    Found {len(flight_gs_data):,} records in time range")

                if len(flight_gs_data) == 0:
                    print(f"    ⚠️  No ground segment data found for this time range")
                    continue

                # Create output filename
                output_file = os.path.join(flight['dir'], f"GROUND_SEGMENT_data_{flight['dir']}.csv")

                # Check if directory exists
                if not os.path.exists(flight['dir']):
                    print(f"    ✗ Directory {flight['dir']} does not exist!")
                    continue

                # Save the segmented ground data
                flight_gs_data.to_csv(output_file, index=False)

                print(f"    ✓ Saved: {output_file}")
                print(f"      Records: {len(flight_gs_data):,}")
                print(f"      Time span: {flight_gs_data['_time'].min()} to {flight_gs_data['_time'].max()}")

        except Exception as e:
            print(f"✗ Error processing {gs_file}: {str(e)}")

    # Ground segment data files
    july16_gs_file = '/Users/baharakqaderi/field-data-pipeline/data/INFLUX/GROUND_SEGMENT_data_2025-07-16.csv'
    july17_gs_file = '/Users/baharakqaderi/field-data-pipeline/data/INFLUX/GROUND_SEGMENT_data_2025-07-17.csv'

    # Process July 16th data
    process_ground_data(july16_gs_file, july16_flights, "July 16th")

    # Process July 17th data
    process_ground_data(july17_gs_file, july17_flights, "July 17th")

    print(f"\n{'='*70}")
    print("Ground segment data segmentation complete!")

if __name__ == "__main__":
    segment_ground_data()
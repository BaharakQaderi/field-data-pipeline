import pandas as pd
import numpy as np
from flight_data_alignment import FlightDataAligner
import os

def process_july17_flights():
    """Process July 17th flight alignments and save each to separate CSV files"""

    # Flight data mapping from TODO.txt for July 17th
    flights = [
        {
            'name': 'july17_flight_001',
            'gs_start': '2025-07-17 10:15:08',
            'gs_end': '2025-07-17 10:23:42',
            'fs_start': '2025-07-15 22:54:36',
            'fs_end': '2025-07-15 23:03:10',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_224238.csv'
        },
        {
            'name': 'july17_flight_002',
            'gs_start': '2025-07-17 10:37:37',
            'gs_end': '2025-07-17 10:47:17',
            'fs_start': '2025-07-15 23:17:07',
            'fs_end': '2025-07-15 23:26:45',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_224238.csv'
        },
        {
            'name': 'july17_flight_003',
            'gs_start': '2025-07-17 10:54:17',
            'gs_end': '2025-07-17 11:15:08',
            'fs_start': '2025-07-15 23:33:44',
            'fs_end': '2025-07-15 23:54:36',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_224238.csv'
        },
        {
            'name': 'july17_flight_004',
            'gs_start': '2025-07-17 11:30:35',
            'gs_end': '2025-07-17 11:50:12',
            'fs_start': '2025-07-16 00:10:03',
            'fs_end': '2025-07-16 00:29:40',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_224238.csv'
        },
        {
            'name': 'july17_flight_005',
            'gs_start': '2025-07-17 12:34:24',
            'gs_end': '2025-07-17 12:42:04',
            'fs_start': '2025-07-16 01:07:23',
            'fs_end': '2025-07-16 01:14:58',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        },
        {
            'name': 'july17_flight_006',
            'gs_start': '2025-07-17 12:45:20',
            'gs_end': '2025-07-17 12:49:25',
            'fs_start': '2025-07-16 01:17:58',
            'fs_end': '2025-07-16 01:22:18',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        },
        {
            'name': 'july17_flight_007',
            'gs_start': '2025-07-17 12:51:41',
            'gs_end': '2025-07-17 12:55:51',
            'fs_start': '2025-07-16 01:24:35',
            'fs_end': '2025-07-16 01:28:44',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        },
        {
            'name': 'july17_flight_008',
            'gs_start': '2025-07-17 12:59:20',
            'gs_end': '2025-07-17 13:02:56',
            'fs_start': '2025-07-16 01:32:11',
            'fs_end': '2025-07-16 01:35:47',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        },
        {
            'name': 'july17_flight_009',
            'gs_start': '2025-07-17 13:08:09',
            'gs_end': '2025-07-17 13:10:47',
            'fs_start': '2025-07-16 01:41:03',
            'fs_end': '2025-07-16 01:43:41',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        },
        {
            'name': 'july17_flight_010',
            'gs_start': '2025-07-17 13:15:52',
            'gs_end': '2025-07-17 13:20:25',
            'fs_start': '2025-07-16 01:48:46',
            'fs_end': '2025-07-16 01:53:19',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        },
        {
            'name': 'july17_flight_011',
            'gs_start': '2025-07-17 13:24:51',
            'gs_end': '2025-07-17 13:26:15',
            'fs_start': '2025-07-16 01:57:45',
            'fs_end': '2025-07-16 01:59:09',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        },
        {
            'name': 'july17_flight_012',
            'gs_start': '2025-07-17 13:31:12',
            'gs_end': '2025-07-17 13:32:43',
            'fs_start': '2025-07-16 02:04:08',
            'fs_end': '2025-07-16 02:05:37',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        },
        {
            'name': 'july17_flight_013',
            'gs_start': '2025-07-17 13:32:55',
            'gs_end': '2025-07-17 13:33:54',
            'fs_start': '2025-07-16 02:05:48',
            'fs_end': '2025-07-16 02:06:49',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        },
        {
            'name': 'july17_flight_014',
            'gs_start': '2025-07-17 13:35:45',
            'gs_end': '2025-07-17 13:38:03',
            'fs_start': '2025-07-16 02:08:39',
            'fs_end': '2025-07-16 02:10:56',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        },
        {
            'name': 'july17_flight_015',
            'gs_start': '2025-07-17 13:39:23',
            'gs_end': '2025-07-17 13:46:51',
            'fs_start': '2025-07-16 02:12:16',
            'fs_end': '2025-07-16 02:19:45',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250716_003735.csv'
        }
    ]

    # GS data file for July 17th
    gs_file = '/Users/baharakqaderi/field-data-pipeline/data/INFLUX/FLIGHT_SEGMENT_data_2025-07-17.csv'

    # Create output directory for July 17th aligned flight data
    output_dir = 'aligned_flights_july17'
    os.makedirs(output_dir, exist_ok=True)

    # Process each flight
    results = {}

    for i, flight in enumerate(flights, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING {flight['name'].upper()} ({i}/15)")
        print(f"{'='*80}")

        try:
            # Initialize aligner
            aligner = FlightDataAligner()

            # Load FS data
            print(f"Loading FS data from: {flight['fs_file']}")
            print(f"FS time range: {flight['fs_start']} to {flight['fs_end']}")
            aligner.load_fs_data(flight['fs_file'], flight['fs_start'], flight['fs_end'])

            # Load GS data
            print(f"Loading GS data from: {gs_file}")
            print(f"GS time range: {flight['gs_start']} to {flight['gs_end']}")
            aligner.load_gs_data(gs_file, flight['gs_start'], flight['gs_end'])

            # Create unified time vector
            print("Creating unified time vector...")
            unified_time = aligner.create_unified_time_vector(method='gs_based')

            # Interpolate force components
            print("Interpolating force components...")
            aligned_data = aligner.interpolate_force_components(unified_time)

            # Analyze alignment quality
            print("Analyzing alignment quality...")
            quality_metrics = aligner.analyze_alignment_quality()

            # Create detailed backline statistics
            print("Creating detailed backline statistics...")
            backline_stats = aligner.create_detailed_comparison_stats()

            # Save aligned data to CSV
            output_file = os.path.join(output_dir, f'{flight["name"]}_aligned_data.csv')
            aligned_data.to_csv(output_file, index=False)
            print(f"Aligned data saved to: {output_file}")

            # Create interactive plots
            interactive_plot_file = os.path.join(output_dir, f'{flight["name"]}_backline_comparison.html')
            aligner.plot_interactive_backline_comparison(interactive_plot_file)

            side_by_side_plot_file = os.path.join(output_dir, f'{flight["name"]}_side_by_side_comparison.html')
            aligner.plot_side_by_side_comparison(side_by_side_plot_file)

            # Store results
            results[flight['name']] = {
                'aligner': aligner,
                'aligned_data': aligned_data,
                'quality_metrics': quality_metrics,
                'backline_stats': backline_stats,
                'output_file': output_file
            }

            print(f"✓ {flight['name']} processed successfully!")

        except Exception as e:
            print(f"✗ Error processing {flight['name']}: {str(e)}")
            results[flight['name']] = {'error': str(e)}

    # Create summary report
    print(f"\n{'='*80}")
    print("JULY 17TH PROCESSING SUMMARY")
    print(f"{'='*80}")

    summary_data = []
    for flight_name, result in results.items():
        if 'error' not in result:
            quality = result['quality_metrics']
            summary_data.append({
                'Flight': flight_name,
                'Status': 'Success',
                'Correlation': f"{quality['correlation']:.4f}",
                'RMSE': f"{quality['rmse']:.4f} kg",
                'MAE': f"{quality['mae']:.4f} kg",
                'Valid Points': quality['valid_points'],
                'Output File': result['output_file']
            })
        else:
            summary_data.append({
                'Flight': flight_name,
                'Status': 'Failed',
                'Error': result['error']
            })

    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, 'july17_alignment_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    print(f"\nJuly 17th alignment summary saved to: {summary_file}")
    print(f"All aligned flight data saved in: {output_dir}/")

    return results

if __name__ == "__main__":
    results = process_july17_flights()
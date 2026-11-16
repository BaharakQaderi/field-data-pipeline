import pandas as pd
import numpy as np
from flight_data_alignment import FlightDataAligner
import os

def process_multiple_flights():
    """Process multiple flight alignments and save each to separate CSV files"""

    # Flight data mapping from TODO.txt
    flights = [
        {
            'name': 'flight_001',
            'gs_start': '2025-07-16 10:29:25',
            'gs_end': '2025-07-16 10:32:59',
            'fs_start': '2025-07-15 17:32:50',
            'fs_end': '2025-07-15 17:36:18',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_161829.csv'
        },
        {
            'name': 'flight_002',
            'gs_start': '2025-07-16 10:58:01',
            'gs_end': '2025-07-16 11:02:16',
            'fs_start': '2025-07-15 18:01:10',
            'fs_end': '2025-07-15 18:05:33',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_161829.csv'
        },
        {
            'name': 'flight_003',
            'gs_start': '2025-07-16 11:47:46',
            'gs_end': '2025-07-16 12:18:40',
            'fs_start': '2025-07-15 18:50:05',
            'fs_end': '2025-07-15 19:20:58',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_183235.csv'
        },
        {
            'name': 'flight_004',
            'gs_start': '2025-07-16 13:46:41',
            'gs_end': '2025-07-16 13:51:33',
            'fs_start': '2025-07-15 20:40:01',
            'fs_end': '2025-07-15 20:53:51',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_183235.csv'
        },
        {
            'name': 'flight_005',
            'gs_start': '2025-07-16 14:00:25',
            'gs_end': '2025-07-16 14:03:27',
            'fs_start': '2025-07-15 21:02:43',
            'fs_end': '2025-07-15 21:05:27',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_183235.csv'
        },
        {
            'name': 'flight_006',
            'gs_start': '2025-07-16 14:37:58',
            'gs_end': '2025-07-16 14:46:57',
            'fs_start': '2025-07-15 21:31:50',
            'fs_end': '2025-07-15 21:40:46',
            'fs_file': '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_211757.csv'
        }
    ]

    # GS data file (same for all flights)
    gs_file = '/Users/baharakqaderi/field-data-pipeline/data/INFLUX/FLIGHT_SEGMENT_data_2025-07-16.csv'

    # Create output directory for aligned flight data
    output_dir = 'aligned_flights'
    os.makedirs(output_dir, exist_ok=True)

    # Process each flight
    results = {}

    for i, flight in enumerate(flights, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING {flight['name'].upper()} ({i}/6)")
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
    print("PROCESSING SUMMARY")
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
    summary_file = os.path.join(output_dir, 'alignment_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    print(f"\nAlignment summary saved to: {summary_file}")
    print(f"All aligned flight data saved in: {output_dir}/")

    return results

if __name__ == "__main__":
    results = process_multiple_flights()
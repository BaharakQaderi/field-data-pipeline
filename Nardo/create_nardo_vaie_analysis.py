import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob

# Configure matplotlib for better looking plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 18
})

def compute_moving_averages_and_max(data, force_column, t_width_values):
    """Compute moving averages and maximum values for different time windows"""
    results = []

    for t_width in t_width_values:
        if t_width == 0:
            # For t_width = 0, use original data
            max_value = data[force_column].max()
        else:
            # Convert t_width seconds to number of data points
            # Assuming approximately 10 Hz sampling rate for Nardo data
            window_size = max(1, int(t_width * 10))

            # Calculate rolling mean
            rolling_mean = data[force_column].rolling(window=window_size, center=True).mean()
            max_value = rolling_mean.max()

        results.append(max_value)

    return results

def create_nardo_vaie_analysis():
    """Create VAIE-style moving average analysis for Nardo flights (forces only)"""

    print("Creating VAIE-style moving average analysis for Nardo flights...")
    print("="*80)

    # Process both Jan and May directories
    base_dirs = {
        'Jan': '/Users/baharakqaderi/field-data-pipeline/Nardo/Main_flights_data_Jan',
        'May': '/Users/baharakqaderi/field-data-pipeline/Nardo/Main_flights_data_May'
    }

    # VAIE configuration - same as individual_flight_segments
    t_width = [0, 1, 2, 3, 5, 7, 10, 15, 20]  # seconds

    flight_counter = 1

    for month, base_dir in base_dirs.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING {month.upper()} FLIGHTS")
        print(f"{'='*60}")

        if not os.path.exists(base_dir):
            print(f"Directory not found: {base_dir}")
            continue

        # Find all flight directories
        flight_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('2025_')])

        if not flight_dirs:
            print(f"No flight directories found in {base_dir}")
            continue

        print(f"Found {len(flight_dirs)} flight directories in {month}")

        for flight_dir in flight_dirs:
            flight_path = os.path.join(base_dir, flight_dir)
            print(f"\n[{flight_counter}] Processing: {flight_dir}")

            try:
                # Find required CSV files
                plc_file = os.path.join(flight_path, 'PLC_processed_data.csv')

                if not os.path.exists(plc_file):
                    print(f"  ✗ Missing PLC_processed_data.csv in {flight_dir}")
                    continue

                # Load data
                plc_data = pd.read_csv(plc_file)
                print(f"  PLC data: {len(plc_data):,} records")

                # Calculate total pulling force and backline forces
                plc_data['total_force'] = (plc_data['force_back_right'] +
                                         plc_data['force_5th_line'] +
                                         plc_data['force_front'] +
                                         plc_data['force_back_left'])

                plc_data['backline_left'] = plc_data['force_back_left']
                plc_data['backline_right'] = plc_data['force_back_right']
                plc_data['backline_diff'] = np.abs(plc_data['backline_left'] - plc_data['backline_right'])
                plc_data['backline_sum'] = plc_data['backline_left'] + plc_data['backline_right']

                # Extract flight information
                flight_name = f"Flight_{flight_counter}"

                # Parse date and time from directory name
                parts = flight_dir.split('_')
                date = f"{parts[0]}-{parts[1]}-{parts[2]}"
                time_start = f"{parts[3]}:{parts[4]}:{parts[5]}"

                # Calculate flight duration
                if '_time' in plc_data.columns:
                    plc_data['_time'] = pd.to_datetime(plc_data['_time'], format='mixed')
                    start_time = plc_data['_time'].iloc[0]
                    end_time = plc_data['_time'].iloc[-1]
                    duration = end_time - start_time
                    flight_duration = f"{duration.total_seconds()/60:.1f}min"
                else:
                    flight_duration = "Unknown"

                # Calculate moving averages for each force component
                back_force_left_results = compute_moving_averages_and_max(plc_data, 'backline_left', t_width)
                back_force_right_results = compute_moving_averages_and_max(plc_data, 'backline_right', t_width)
                force_diff_results = compute_moving_averages_and_max(plc_data, 'backline_diff', t_width)
                total_back_force_results = compute_moving_averages_and_max(plc_data, 'backline_sum', t_width)

                # Create VAIE-style plots (3 graphs instead of 4, no torque)
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'{flight_name} - Force Analysis | Date: {date[-5:]} | Duration: {flight_duration}',
                           fontsize=16, fontweight='bold')

                # Graph 1: Back Force Analysis (Left and Right)
                axes[0, 0].plot(t_width, back_force_left_results, 'go-', linewidth=2, markersize=6, label='Left Back Force (kgf)')
                axes[0, 0].plot(t_width, back_force_right_results, 'ro-', linewidth=2, markersize=6, label='Right Back Force (kgf)')
                axes[0, 0].set_title('Graph 1: Back Force Analysis', fontweight='bold')
                axes[0, 0].set_xlabel('Averaging Window (seconds)')
                axes[0, 0].set_ylabel('Maximum Back Force (kgf)')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_xticks(t_width)

                # Graph 2: Force Difference Analysis
                axes[0, 1].plot(t_width, force_diff_results, 'bo-', linewidth=2, markersize=6, label='|Left - Right| Force')
                axes[0, 1].set_title('Graph 2: Force Difference Analysis', fontweight='bold')
                axes[0, 1].set_xlabel('Averaging Window (seconds)')
                axes[0, 1].set_ylabel('Maximum Diff Force (kgf)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_xticks(t_width)

                # Graph 3: Total Back Force Analysis
                axes[1, 0].plot(t_width, total_back_force_results, 'o-', color='orange', linewidth=2, markersize=6, label='Total Back Force (L + R)')
                axes[1, 0].set_title('Graph 3: Total Back Force Analysis', fontweight='bold')
                axes[1, 0].set_xlabel('Averaging Window (seconds)')
                axes[1, 0].set_ylabel('Maximum Back Force (kgf)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_xticks(t_width)

                # Graph 4: Summary Statistics (text-based)
                axes[1, 1].axis('off')
                summary_text = f"""
Flight Summary Statistics:

Total Data Points: {len(plc_data):,}
Flight Duration: {flight_duration}
Date: {date}
Month: {month}

Force Statistics (kgf):
• Max Left Back: {plc_data['backline_left'].max():.1f}
• Max Right Back: {plc_data['backline_right'].max():.1f}
• Max Total Back: {plc_data['backline_sum'].max():.1f}
• Max Force Diff: {plc_data['backline_diff'].max():.1f}
• Max Total Force: {plc_data['total_force'].max():.1f}

Average Forces (kgf):
• Avg Left Back: {plc_data['backline_left'].mean():.1f}
• Avg Right Back: {plc_data['backline_right'].mean():.1f}
• Avg Total Back: {plc_data['backline_sum'].mean():.1f}
• Avg Force Diff: {plc_data['backline_diff'].mean():.1f}
• Avg Total Force: {plc_data['total_force'].mean():.1f}
                """
                axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                               fontsize=11, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

                plt.tight_layout()

                # Save the plot
                plot_filename = f"{flight_path}/nardo_vaie_{flight_name.lower()}_forces_analysis.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"  ✓ Saved VAIE force analysis plot: {plot_filename}")

                # Create detailed results CSV
                results_data = {
                    'averaging_window_seconds': t_width,
                    'max_left_back_force_kgf': back_force_left_results,
                    'max_right_back_force_kgf': back_force_right_results,
                    'max_force_difference_kgf': force_diff_results,
                    'max_total_back_force_kgf': total_back_force_results
                }

                results_df = pd.DataFrame(results_data)
                csv_filename = f"{flight_path}/nardo_vaie_{flight_name.lower()}_forces_results.csv"
                results_df.to_csv(csv_filename, index=False)
                print(f"  ✓ Saved VAIE force results CSV: {csv_filename}")

                # Create detailed text report
                report_filename = f"{flight_path}/nardo_vaie_{flight_name.lower()}_forces_report.txt"
                with open(report_filename, 'w') as f:
                    f.write("Nardo VAIE-Style Force Analysis Report\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Flight Information:\n")
                    f.write(f"  Flight Number: {flight_name}\n")
                    f.write(f"  Date: {date}\n")
                    f.write(f"  Start Time: {time_start}\n")
                    f.write(f"  Duration: {flight_duration}\n")
                    f.write(f"  Month: {month}\n")
                    f.write(f"  Data Points: {len(plc_data):,}\n\n")

                    f.write("Analysis Configuration:\n")
                    f.write("-"*25 + "\n")
                    f.write(f"Averaging Windows: {t_width} seconds\n")
                    f.write("Force Components Analyzed:\n")
                    f.write("  - Left Backline Force\n")
                    f.write("  - Right Backline Force\n")
                    f.write("  - Backline Force Difference (|Left - Right|)\n")
                    f.write("  - Total Backline Force (Left + Right)\n\n")

                    f.write("Moving Average Results:\n")
                    f.write("-"*25 + "\n")
                    f.write("Window(s) | Left Back | Right Back | Force Diff | Total Back\n")
                    f.write("-"*60 + "\n")
                    for i, tw in enumerate(t_width):
                        f.write(f"{tw:7.0f}   | {back_force_left_results[i]:8.2f}  | {back_force_right_results[i]:9.2f}  | "
                               f"{force_diff_results[i]:9.2f}  | {total_back_force_results[i]:9.2f}\n")

                print(f"  ✓ Saved VAIE force report: {report_filename}")

            except Exception as e:
                print(f"  ✗ Error processing {flight_dir}: {str(e)}")

            flight_counter += 1

    print(f"\n{'='*80}")
    print("Nardo VAIE-style force analysis complete!")
    print("Each flight directory now contains:")
    print("  - nardo_vaie_flight_X_forces_analysis.png (VAIE-style force plots)")
    print("  - nardo_vaie_flight_X_forces_results.csv (numerical results)")
    print("  - nardo_vaie_flight_X_forces_report.txt (detailed report)")

if __name__ == "__main__":
    create_nardo_vaie_analysis()
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

def calculate_pod_efficiency():
    """Calculate POD mechanical efficiency for all individual flight segments"""

    print("Calculating POD mechanical efficiency for all individual flight segments...")
    print("="*80)

    # Create output directory
    output_dir = '/Users/baharakqaderi/field-data-pipeline/individual_flight_segments/pod_efficiency_analysis'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Find all flight segment directories
    base_dir = '/Users/baharakqaderi/field-data-pipeline/individual_flight_segments'
    flight_dirs = sorted([d for d in os.listdir(base_dir)
                         if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('2025-')])

    if not flight_dirs:
        print("No flight directories found!")
        return

    print(f"Found {len(flight_dirs)} flight directories")

    # Initialize results storage
    all_efficiency_results = []

    for flight_dir in flight_dirs:
        flight_path = os.path.join(base_dir, flight_dir)
        print(f"\nProcessing: {flight_dir}")

        try:
            # Find FLIGHT_SEGMENT and GROUND_SEGMENT CSV files
            flight_csv_files = glob.glob(os.path.join(flight_path, '*FLIGHT_SEGMENT*.csv'))
            ground_csv_files = glob.glob(os.path.join(flight_path, '*GROUND_SEGMENT*.csv'))

            if not flight_csv_files:
                print(f"  ✗ No FLIGHT_SEGMENT CSV file found in {flight_dir}")
                continue

            if not ground_csv_files:
                print(f"  ✗ No GROUND_SEGMENT CSV file found in {flight_dir}")
                continue

            flight_csv_file = flight_csv_files[0]
            ground_csv_file = ground_csv_files[0]

            flight_data = pd.read_csv(flight_csv_file)
            ground_data = pd.read_csv(ground_csv_file)

            print(f"  Loaded flight data: {len(flight_data):,} records")
            print(f"  Loaded ground data: {len(ground_data):,} records")

            # Merge flight and ground data on time
            if '_time' in flight_data.columns and '_time' in ground_data.columns:
                flight_data['_time'] = pd.to_datetime(flight_data['_time'], format='mixed')
                ground_data['_time'] = pd.to_datetime(ground_data['_time'], format='mixed')

                # Merge using nearest time matching
                flight_data = pd.merge_asof(
                    flight_data.sort_values('_time'),
                    ground_data[['_time', 'GROUND_SEGMENT_brake_command']].sort_values('_time'),
                    on='_time',
                    direction='nearest'
                )
                print(f"  Merged data: {len(flight_data):,} records")
            else:
                print(f"  ✗ Missing _time column for merging")
                continue

            # Extract flight information from directory name
            parts = flight_dir.split('-')
            if len(parts) >= 4:
                date = f"{parts[0]}-{parts[1]}-{parts[2]}"
                time_start = f"{parts[3]}:{parts[4] if len(parts) > 4 else '00'}"
                flight_name = f"Flight_{flight_dir.replace('-', '_')}"
            else:
                date = "Unknown"
                time_start = "Unknown"
                flight_name = f"Flight_{flight_dir}"

            # Check required columns
            required_cols = ['FLIGHT_SEGMENT_l_torque', 'FLIGHT_SEGMENT_r_torque', 'GROUND_SEGMENT_brake_command',
                           'gs_Backline_Left_kg', 'gs_Backline_Right_kg']
            missing_cols = [col for col in required_cols if col not in flight_data.columns]
            if missing_cols:
                print(f"  ✗ Missing required columns: {missing_cols}")
                continue

            # Filter out invalid data (NaN values)
            valid_data = flight_data.dropna(subset=required_cols)
            if len(valid_data) == 0:
                print(f"  ✗ No valid data after removing NaN values")
                continue

            print(f"  Data points after NaN removal: {len(valid_data):,}")

            # Filter out records with torque under 0.7
            # Remove records where both left and right torque are under 0.7
            torque_filter = (np.abs(valid_data['FLIGHT_SEGMENT_l_torque']) >= 0.7) | \
                           (np.abs(valid_data['FLIGHT_SEGMENT_r_torque']) >= 0.7)
            valid_data = valid_data[torque_filter]

            if len(valid_data) == 0:
                print(f"  ✗ No valid data after torque filtering (torque >= 0.7)")
                continue

            print(f"  Valid data points (torque >= 0.7): {len(valid_data):,}")

            # Calculate actuated force on motor side and kite force based on brake status
            def calculate_forces(row):
                left_torque = row['FLIGHT_SEGMENT_l_torque']
                right_torque = row['FLIGHT_SEGMENT_r_torque']
                brake_command = row['GROUND_SEGMENT_brake_command']
                backline_left = row['gs_Backline_Left_kg']
                backline_right = row['gs_Backline_Right_kg']

                if brake_command == 1:  # Brake engaged
                    # F_motor = |T_left + T_right| / ((0.032/2) * 9.81)
                    motor_force = np.abs(left_torque + right_torque) / ((0.032/2) * 9.81)
                    # F_kite = |Backline_Right - Backline_Left|
                    kite_force = np.abs(backline_right - backline_left)

                    # For brake engaged, left and right are combined
                    motor_force_left = motor_force_right = motor_force / 2  # Split for consistency
                    kite_force_left = kite_force_right = kite_force / 2     # Split for consistency

                else:  # Brake not engaged - calculate separately for left and right
                    # F_motor_left = |T_left| / ((0.032/2) * 9.81)
                    # F_motor_right = |T_right| / ((0.032/2) * 9.81)
                    motor_force_left = np.abs(left_torque) / ((0.032/2) * 9.81)
                    motor_force_right = np.abs(right_torque) / ((0.032/2) * 9.81)

                    # For brake not engaged, we calculate left and right efficiencies separately
                    # The "overall" efficiency will be the average of left and right efficiencies
                    motor_force = (motor_force_left + motor_force_right) / 2

                    # F_kite_left = backline_left, F_kite_right = backline_right
                    # Each side operates independently when brake is not engaged
                    kite_force_left = backline_left
                    kite_force_right = backline_right

                    # For overall calculation, we should NOT average the forces
                    # Instead, we'll calculate individual efficiencies and then average those
                    # But for compatibility with existing code, we'll keep this as a placeholder
                    kite_force = kite_force_left  # This will be recalculated properly below

                return motor_force, kite_force, motor_force_left, motor_force_right, kite_force_left, kite_force_right

            # Calculate motor force and kite force for each data point
            valid_data = valid_data.copy()
            forces_result = valid_data.apply(calculate_forces, axis=1, result_type='expand')
            valid_data['motor_force_kgf'] = forces_result[0]
            valid_data['kite_force_kgf'] = forces_result[1]
            valid_data['motor_force_left_kgf'] = forces_result[2]
            valid_data['motor_force_right_kgf'] = forces_result[3]
            valid_data['kite_force_left_kgf'] = forces_result[4]
            valid_data['kite_force_right_kgf'] = forces_result[5]

            # Calculate left and right efficiencies separately first
            valid_data['efficiency_left_percent'] = np.where(
                valid_data['motor_force_left_kgf'] != 0,
                (valid_data['kite_force_left_kgf'] / valid_data['motor_force_left_kgf']) * 100,
                np.nan
            )

            valid_data['efficiency_right_percent'] = np.where(
                valid_data['motor_force_right_kgf'] != 0,
                (valid_data['kite_force_right_kgf'] / valid_data['motor_force_right_kgf']) * 100,
                np.nan
            )

            # Calculate overall efficiency based on brake status
            def calculate_overall_efficiency(row):
                brake_command = row['GROUND_SEGMENT_brake_command']
                if brake_command == 1:  # Brake engaged
                    # Use the combined force calculation
                    if row['motor_force_kgf'] != 0:
                        return (row['kite_force_kgf'] / row['motor_force_kgf']) * 100
                    else:
                        return np.nan
                else:  # Brake not engaged
                    # Average the left and right efficiencies
                    left_eff = row['efficiency_left_percent']
                    right_eff = row['efficiency_right_percent']
                    if not np.isnan(left_eff) and not np.isnan(right_eff):
                        return (left_eff + right_eff) / 2
                    elif not np.isnan(left_eff):
                        return left_eff
                    elif not np.isnan(right_eff):
                        return right_eff
                    else:
                        return np.nan

            valid_data['efficiency_percent'] = valid_data.apply(calculate_overall_efficiency, axis=1)

            # Remove infinite and invalid efficiency values
            valid_efficiency = valid_data['efficiency_percent'].replace([np.inf, -np.inf], np.nan).dropna()

            print(f"  Data points before noise filtering: {len(valid_data):,}")

            # Additional noise filtering:
            # 1. Remove records with motor forces under 25 kgf
            # 2. Remove records with efficiency ratios above 100%
            noise_filter = (
                (valid_data['motor_force_kgf'] >= 25) &
                (valid_data['efficiency_percent'] <= 100) &
                (~np.isnan(valid_data['efficiency_percent']))
            )

            valid_data_clean = valid_data[noise_filter]

            if len(valid_data_clean) == 0:
                print(f"  ✗ No valid data after noise filtering (motor >= 25, efficiency <= 100%)")
                continue

            print(f"  Data points after noise filtering: {len(valid_data_clean):,}")

            # Update valid_data to use the cleaned data
            valid_data = valid_data_clean
            valid_efficiency = valid_data['efficiency_percent'].replace([np.inf, -np.inf], np.nan).dropna()

            if len(valid_efficiency) == 0:
                print(f"  ✗ No valid efficiency calculations")
                continue

            # Separate data by brake status
            brake_engaged = valid_data[valid_data['GROUND_SEGMENT_brake_command'] == 1]
            brake_not_engaged = valid_data[valid_data['GROUND_SEGMENT_brake_command'] != 1]

            # Calculate statistics
            efficiency_stats = {
                'flight_name': flight_name,
                'date': date,
                'time_start': time_start,
                'total_points': len(valid_data),
                'valid_efficiency_points': len(valid_efficiency),
                'brake_engaged_points': len(brake_engaged),
                'brake_not_engaged_points': len(brake_not_engaged),
                'overall_avg_efficiency': valid_efficiency.mean(),
                'overall_median_efficiency': valid_efficiency.median(),
                'overall_std_efficiency': valid_efficiency.std(),
                'overall_min_efficiency': valid_efficiency.min(),
                'overall_max_efficiency': valid_efficiency.max(),
                'avg_kite_force': valid_data['kite_force_kgf'].mean(),
                'avg_motor_force': valid_data['motor_force_kgf'].mean(),
            }

            # Brake engaged statistics
            if len(brake_engaged) > 0:
                brake_engaged_eff = brake_engaged['efficiency_percent'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(brake_engaged_eff) > 0:
                    efficiency_stats.update({
                        'brake_engaged_avg_efficiency': brake_engaged_eff.mean(),
                        'brake_engaged_median_efficiency': brake_engaged_eff.median(),
                        'brake_engaged_std_efficiency': brake_engaged_eff.std(),
                        'brake_engaged_avg_kite_force': brake_engaged['kite_force_kgf'].mean(),
                        'brake_engaged_avg_motor_force': brake_engaged['motor_force_kgf'].mean(),
                    })
                else:
                    efficiency_stats.update({
                        'brake_engaged_avg_efficiency': np.nan,
                        'brake_engaged_median_efficiency': np.nan,
                        'brake_engaged_std_efficiency': np.nan,
                        'brake_engaged_avg_kite_force': np.nan,
                        'brake_engaged_avg_motor_force': np.nan,
                    })
            else:
                efficiency_stats.update({
                    'brake_engaged_avg_efficiency': np.nan,
                    'brake_engaged_median_efficiency': np.nan,
                    'brake_engaged_std_efficiency': np.nan,
                    'brake_engaged_avg_kite_force': np.nan,
                    'brake_engaged_avg_motor_force': np.nan,
                })

            # Brake not engaged statistics
            if len(brake_not_engaged) > 0:
                brake_not_engaged_eff = brake_not_engaged['efficiency_percent'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(brake_not_engaged_eff) > 0:
                    efficiency_stats.update({
                        'brake_not_engaged_avg_efficiency': brake_not_engaged_eff.mean(),
                        'brake_not_engaged_median_efficiency': brake_not_engaged_eff.median(),
                        'brake_not_engaged_std_efficiency': brake_not_engaged_eff.std(),
                        'brake_not_engaged_avg_kite_force': brake_not_engaged['kite_force_kgf'].mean(),
                        'brake_not_engaged_avg_motor_force': brake_not_engaged['motor_force_kgf'].mean(),
                    })
                else:
                    efficiency_stats.update({
                        'brake_not_engaged_avg_efficiency': np.nan,
                        'brake_not_engaged_median_efficiency': np.nan,
                        'brake_not_engaged_std_efficiency': np.nan,
                        'brake_not_engaged_avg_kite_force': np.nan,
                        'brake_not_engaged_avg_motor_force': np.nan,
                    })
            else:
                efficiency_stats.update({
                    'brake_not_engaged_avg_efficiency': np.nan,
                    'brake_not_engaged_median_efficiency': np.nan,
                    'brake_not_engaged_std_efficiency': np.nan,
                    'brake_not_engaged_avg_kite_force': np.nan,
                    'brake_not_engaged_avg_motor_force': np.nan,
                })

            all_efficiency_results.append(efficiency_stats)

            # Calculate brake engagement percentage
            brake_engaged_percentage = (len(brake_engaged) / len(valid_data)) * 100

            # Check if we have brake not engaged data to create separate plots
            has_brake_not_engaged = len(brake_not_engaged) > 0

            if has_brake_not_engaged:
                # Create plots with separate left/right analysis for brake not engaged
                fig, axes = plt.subplots(4, 2, figsize=(16, 24))
                fig.suptitle(f'POD Mechanical Efficiency Analysis: {flight_name} ({date} {time_start}) | Brake Engaged: {brake_engaged_percentage:.1f}%',
                            fontsize=16, fontweight='bold')
            else:
                # Standard plot for brake engaged only
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'POD Mechanical Efficiency Analysis: {flight_name} ({date} {time_start}) | Brake Engaged: {brake_engaged_percentage:.1f}%',
                            fontsize=16, fontweight='bold')

            # 1. Efficiency distribution histogram
            axes[0, 0].hist(valid_efficiency, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            axes[0, 0].set_title('Efficiency Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('Efficiency (%)')
            axes[0, 0].set_ylabel('Probability Density')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axvline(valid_efficiency.mean(), color='red', linestyle='--',
                              label=f'Mean: {valid_efficiency.mean():.1f}%')
            axes[0, 0].legend()

            # 2. Kite vs Motor Force scatter plot
            axes[0, 1].scatter(valid_data['motor_force_kgf'], valid_data['kite_force_kgf'],
                              alpha=0.6, s=10, color='green')
            axes[0, 1].set_title('Kite Force vs Motor Force', fontweight='bold')
            axes[0, 1].set_xlabel('Motor Force (kgf)')
            axes[0, 1].set_ylabel('Kite Force (kgf)')
            axes[0, 1].grid(True, alpha=0.3)

            # Add ideal efficiency lines
            max_force = max(valid_data['motor_force_kgf'].max(), valid_data['kite_force_kgf'].max())
            x_line = np.linspace(0, max_force, 100)
            axes[0, 1].plot(x_line, x_line, 'r--', label='100% Efficiency', alpha=0.7)
            axes[0, 1].plot(x_line, 0.8*x_line, 'orange', linestyle='--', label='80% Efficiency', alpha=0.7)
            axes[0, 1].plot(x_line, 0.6*x_line, 'purple', linestyle='--', label='60% Efficiency', alpha=0.7)
            axes[0, 1].legend()

            # 3. Efficiency by brake status
            if len(brake_engaged) > 0 and len(brake_not_engaged) > 0:
                brake_engaged_eff_clean = brake_engaged['efficiency_percent'].replace([np.inf, -np.inf], np.nan).dropna()
                brake_not_engaged_eff_clean = brake_not_engaged['efficiency_percent'].replace([np.inf, -np.inf], np.nan).dropna()

                if len(brake_engaged_eff_clean) > 0 and len(brake_not_engaged_eff_clean) > 0:
                    axes[1, 0].hist(brake_engaged_eff_clean, bins=30, alpha=0.6, label='Brake Engaged',
                                   color='red', edgecolor='black', density=True)
                    axes[1, 0].hist(brake_not_engaged_eff_clean, bins=30, alpha=0.6, label='Brake Not Engaged',
                                   color='blue', edgecolor='black', density=True)
                    axes[1, 0].set_title('Efficiency by Brake Status', fontweight='bold')
                    axes[1, 0].set_xlabel('Efficiency (%)')
                    axes[1, 0].set_ylabel('Probability Density')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    axes[1, 0].text(0.5, 0.5, 'Insufficient data for brake status comparison',
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Efficiency by Brake Status', fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'No brake status variation in data',
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Efficiency by Brake Status', fontweight='bold')

            # 4. Time series of efficiency (using index as proxy for time)
            axes[1, 1].plot(valid_data.index, valid_data['efficiency_percent'], 'b-', alpha=0.7, linewidth=1)
            axes[1, 1].set_title('Efficiency Over Time', fontweight='bold')
            axes[1, 1].set_xlabel('Data Point Index')
            axes[1, 1].set_ylabel('Efficiency (%)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(valid_efficiency.mean(), color='red', linestyle='--',
                              label=f'Mean: {valid_efficiency.mean():.1f}%')
            axes[1, 1].legend()

            # Add additional plots for brake not engaged cases
            if has_brake_not_engaged:
                # 5. Left side efficiency for brake not engaged
                brake_not_engaged_left_eff = brake_not_engaged['efficiency_left_percent'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(brake_not_engaged_left_eff) > 0:
                    axes[2, 0].hist(brake_not_engaged_left_eff, bins=30, alpha=0.7, color='green', edgecolor='black', density=True)
                    axes[2, 0].set_title('Left Side Efficiency (Brake Not Engaged)', fontweight='bold')
                    axes[2, 0].set_xlabel('Efficiency (%)')
                    axes[2, 0].set_ylabel('Probability Density')
                    axes[2, 0].grid(True, alpha=0.3)
                    axes[2, 0].axvline(brake_not_engaged_left_eff.mean(), color='red', linestyle='--',
                                      label=f'Mean: {brake_not_engaged_left_eff.mean():.1f}%')
                    axes[2, 0].legend()
                else:
                    axes[2, 0].text(0.5, 0.5, 'No valid left efficiency data\n(Brake Not Engaged)',
                                   ha='center', va='center', transform=axes[2, 0].transAxes)
                    axes[2, 0].set_title('Left Side Efficiency (Brake Not Engaged)', fontweight='bold')

                # 6. Right side efficiency for brake not engaged
                brake_not_engaged_right_eff = brake_not_engaged['efficiency_right_percent'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(brake_not_engaged_right_eff) > 0:
                    axes[2, 1].hist(brake_not_engaged_right_eff, bins=30, alpha=0.7, color='orange', edgecolor='black', density=True)
                    axes[2, 1].set_title('Right Side Efficiency (Brake Not Engaged)', fontweight='bold')
                    axes[2, 1].set_xlabel('Efficiency (%)')
                    axes[2, 1].set_ylabel('Probability Density')
                    axes[2, 1].grid(True, alpha=0.3)
                    axes[2, 1].axvline(brake_not_engaged_right_eff.mean(), color='red', linestyle='--',
                                      label=f'Mean: {brake_not_engaged_right_eff.mean():.1f}%')
                    axes[2, 1].legend()
                else:
                    axes[2, 1].text(0.5, 0.5, 'No valid right efficiency data\n(Brake Not Engaged)',
                                   ha='center', va='center', transform=axes[2, 1].transAxes)
                    axes[2, 1].set_title('Right Side Efficiency (Brake Not Engaged)', fontweight='bold')

                # 7. Left side kite vs motor force scatter plot (brake not engaged)
                if len(brake_not_engaged) > 0:
                    axes[3, 0].scatter(brake_not_engaged['motor_force_left_kgf'], brake_not_engaged['kite_force_left_kgf'],
                                      alpha=0.6, s=10, color='green')
                    axes[3, 0].set_title('Left: Kite Force vs Motor Force (Brake Not Engaged)', fontweight='bold')
                    axes[3, 0].set_xlabel('Motor Force Left (kgf)')
                    axes[3, 0].set_ylabel('Kite Force Left (kgf)')
                    axes[3, 0].grid(True, alpha=0.3)

                    # Add ideal efficiency lines for left side
                    max_force_left = max(brake_not_engaged['motor_force_left_kgf'].max(),
                                        brake_not_engaged['kite_force_left_kgf'].max())
                    x_line_left = np.linspace(0, max_force_left, 100)
                    axes[3, 0].plot(x_line_left, x_line_left, 'r--', label='100% Efficiency', alpha=0.7)
                    axes[3, 0].plot(x_line_left, 0.8*x_line_left, 'orange', linestyle='--', label='80% Efficiency', alpha=0.7)
                    axes[3, 0].plot(x_line_left, 0.6*x_line_left, 'purple', linestyle='--', label='60% Efficiency', alpha=0.7)
                    axes[3, 0].legend()
                else:
                    axes[3, 0].text(0.5, 0.5, 'No brake not engaged data\nfor left side scatter plot',
                                   ha='center', va='center', transform=axes[3, 0].transAxes)
                    axes[3, 0].set_title('Left: Kite Force vs Motor Force (Brake Not Engaged)', fontweight='bold')

                # 8. Right side kite vs motor force scatter plot (brake not engaged)
                if len(brake_not_engaged) > 0:
                    axes[3, 1].scatter(brake_not_engaged['motor_force_right_kgf'], brake_not_engaged['kite_force_right_kgf'],
                                      alpha=0.6, s=10, color='orange')
                    axes[3, 1].set_title('Right: Kite Force vs Motor Force (Brake Not Engaged)', fontweight='bold')
                    axes[3, 1].set_xlabel('Motor Force Right (kgf)')
                    axes[3, 1].set_ylabel('Kite Force Right (kgf)')
                    axes[3, 1].grid(True, alpha=0.3)

                    # Add ideal efficiency lines for right side
                    max_force_right = max(brake_not_engaged['motor_force_right_kgf'].max(),
                                         brake_not_engaged['kite_force_right_kgf'].max())
                    x_line_right = np.linspace(0, max_force_right, 100)
                    axes[3, 1].plot(x_line_right, x_line_right, 'r--', label='100% Efficiency', alpha=0.7)
                    axes[3, 1].plot(x_line_right, 0.8*x_line_right, 'orange', linestyle='--', label='80% Efficiency', alpha=0.7)
                    axes[3, 1].plot(x_line_right, 0.6*x_line_right, 'purple', linestyle='--', label='60% Efficiency', alpha=0.7)
                    axes[3, 1].legend()
                else:
                    axes[3, 1].text(0.5, 0.5, 'No brake not engaged data\nfor right side scatter plot',
                                   ha='center', va='center', transform=axes[3, 1].transAxes)
                    axes[3, 1].set_title('Right: Kite Force vs Motor Force (Brake Not Engaged)', fontweight='bold')

            plt.tight_layout()

            # Save plot
            plot_filename = f"{output_dir}/{flight_name}_efficiency_analysis.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved efficiency plot: {plot_filename}")
            print(f"  ✓ Average efficiency: {valid_efficiency.mean():.2f}%")

        except Exception as e:
            print(f"  ✗ Error processing {flight_dir}: {str(e)}")

    # Create summary report
    if all_efficiency_results:
        summary_df = pd.DataFrame(all_efficiency_results)

        # Save detailed CSV
        csv_filename = f"{output_dir}/pod_efficiency_summary.csv"
        summary_df.to_csv(csv_filename, index=False)
        print(f"\n✓ Saved summary CSV: {csv_filename}")

        # Create summary text report
        report_filename = f"{output_dir}/pod_efficiency_report.txt"
        with open(report_filename, 'w') as f:
            f.write("POD Mechanical Efficiency Analysis Report\n")
            f.write("="*50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Flights Analyzed: {len(all_efficiency_results)}\n\n")

            f.write("Formulas Used:\n")
            f.write("-"*20 + "\n")
            f.write("Data Filtering:\n")
            f.write("  1. Records with |T_left| < 0.7 AND |T_right| < 0.7 are excluded\n")
            f.write("     (At least one side must have torque >= 0.7)\n")
            f.write("  2. Records with motor force < 25 kgf are excluded (noise removal)\n")
            f.write("  3. Records with efficiency > 100% are excluded (noise removal)\n")
            f.write("Brake Engaged:\n")
            f.write("  F_motor = |T_left + T_right| / ((0.032/2) * 9.81)\n")
            f.write("  F_kite = |Backline_Right - Backline_Left| (kgf)\n")
            f.write("  Efficiency = (F_kite / F_motor) * 100%\n")
            f.write("Brake Not Engaged:\n")
            f.write("  F_motor_left = |T_left| / ((0.032/2) * 9.81)\n")
            f.write("  F_motor_right = |T_right| / ((0.032/2) * 9.81)\n")
            f.write("  F_kite_left = Backline_Left (kgf)\n")
            f.write("  F_kite_right = Backline_Right (kgf)\n")
            f.write("  Efficiency_left = (F_kite_left / F_motor_left) * 100%\n")
            f.write("  Efficiency_right = (F_kite_right / F_motor_right) * 100%\n")
            f.write("  Overall_efficiency = (Efficiency_left + Efficiency_right) / 2\n")
            f.write("  (Separate graphs created for left and right efficiencies)\n\n")

            f.write("Overall Statistics:\n")
            f.write("-"*20 + "\n")
            overall_eff = [r['overall_avg_efficiency'] for r in all_efficiency_results if not np.isnan(r['overall_avg_efficiency'])]
            if overall_eff:
                f.write(f"Average Efficiency Across All Flights: {np.mean(overall_eff):.2f}%\n")
                f.write(f"Median Efficiency Across All Flights: {np.median(overall_eff):.2f}%\n")
                f.write(f"Standard Deviation: {np.std(overall_eff):.2f}%\n")
                f.write(f"Min Efficiency: {np.min(overall_eff):.2f}%\n")
                f.write(f"Max Efficiency: {np.max(overall_eff):.2f}%\n\n")

            f.write("Individual Flight Results:\n")
            f.write("-"*30 + "\n")
            for result in all_efficiency_results:
                f.write(f"{result['flight_name']} ({result['date']} {result['time_start']}):\n")
                f.write(f"  Overall Efficiency: {result['overall_avg_efficiency']:.2f}%\n")
                f.write(f"  Valid Data Points: {result['valid_efficiency_points']:,}\n")
                if not np.isnan(result['brake_engaged_avg_efficiency']):
                    f.write(f"  Brake Engaged Efficiency: {result['brake_engaged_avg_efficiency']:.2f}%\n")
                if not np.isnan(result['brake_not_engaged_avg_efficiency']):
                    f.write(f"  Brake Not Engaged Efficiency: {result['brake_not_engaged_avg_efficiency']:.2f}%\n")
                f.write("\n")

        print(f"✓ Saved summary report: {report_filename}")

        # Create overall summary plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('POD Mechanical Efficiency - Overall Summary', fontsize=16, fontweight='bold')

        # 1. Efficiency distribution across all flights
        overall_efficiencies = [r['overall_avg_efficiency'] for r in all_efficiency_results if not np.isnan(r['overall_avg_efficiency'])]
        if overall_efficiencies:
            axes[0, 0].hist(overall_efficiencies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Average Efficiency Distribution Across Flights', fontweight='bold')
            axes[0, 0].set_xlabel('Average Efficiency (%)')
            axes[0, 0].set_ylabel('Number of Flights')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axvline(np.mean(overall_efficiencies), color='red', linestyle='--',
                              label=f'Overall Mean: {np.mean(overall_efficiencies):.1f}%')
            axes[0, 0].legend()

        # 2. Efficiency by flight timeline
        flight_names = [r['flight_name'] for r in all_efficiency_results]
        efficiencies = [r['overall_avg_efficiency'] for r in all_efficiency_results]
        axes[0, 1].plot(range(len(flight_names)), efficiencies, 'bo-', alpha=0.7)
        axes[0, 1].set_title('Efficiency Trend Across Flights', fontweight='bold')
        axes[0, 1].set_xlabel('Flight Number')
        axes[0, 1].set_ylabel('Average Efficiency (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(0, len(flight_names), max(1, len(flight_names)//10)))

        # 3. Brake engaged vs not engaged comparison
        brake_engaged_effs = [r['brake_engaged_avg_efficiency'] for r in all_efficiency_results if not np.isnan(r['brake_engaged_avg_efficiency'])]
        brake_not_engaged_effs = [r['brake_not_engaged_avg_efficiency'] for r in all_efficiency_results if not np.isnan(r['brake_not_engaged_avg_efficiency'])]

        if brake_engaged_effs and brake_not_engaged_effs:
            axes[1, 0].boxplot([brake_engaged_effs, brake_not_engaged_effs],
                              labels=['Brake Engaged', 'Brake Not Engaged'])
            axes[1, 0].set_title('Efficiency Comparison by Brake Status', fontweight='bold')
            axes[1, 0].set_ylabel('Efficiency (%)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for brake status comparison',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Efficiency Comparison by Brake Status', fontweight='bold')

        # 4. Data points statistics
        total_points = [r['total_points'] for r in all_efficiency_results]
        valid_points = [r['valid_efficiency_points'] for r in all_efficiency_results]
        axes[1, 1].bar(range(len(flight_names)), total_points, alpha=0.6, label='Total Points', color='lightblue')
        axes[1, 1].bar(range(len(flight_names)), valid_points, alpha=0.8, label='Valid Efficiency Points', color='darkblue')
        axes[1, 1].set_title('Data Points per Flight', fontweight='bold')
        axes[1, 1].set_xlabel('Flight Number')
        axes[1, 1].set_ylabel('Number of Data Points')
        axes[1, 1].legend()
        axes[1, 1].set_xticks(range(0, len(flight_names), max(1, len(flight_names)//10)))

        plt.tight_layout()

        # Save summary plot
        summary_plot_filename = f"{output_dir}/pod_efficiency_summary.png"
        plt.savefig(summary_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved summary plot: {summary_plot_filename}")

    print(f"\n{'='*80}")
    print("POD mechanical efficiency analysis complete!")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print("  - Individual flight efficiency plots")
    print("  - pod_efficiency_summary.csv (detailed results)")
    print("  - pod_efficiency_report.txt (summary report)")
    print("  - pod_efficiency_summary.png (overall summary plot)")

if __name__ == "__main__":
    calculate_pod_efficiency()
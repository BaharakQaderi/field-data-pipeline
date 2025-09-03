#!/usr/bin/env python3
"""
Comprehensive Calabria Flights Analysis

Creates both individual flight analyses and combined group analyses.
Properly calculates total forces by summing ALL force columns from PLC data.

Individual plots (similar to July17_Flight_*_analysis_*.png):
- Moving average analysis plots for each flight

Combined plots:  
- Graph 2: t_width vs Left/Right back forces (Fl/Fr)
- Graph 3: t_width vs abs(Fl-Fr)
- Graph 4: t_width vs Fl+Fr (properly calculated as sum of ALL forces)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from pathlib import Path
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Time windows for moving average analysis (same as Vaie analysis)
TIME_WINDOWS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 60, 90, 120, 180]

def load_and_process_flight_data(flight_path):
    """Load and process a single Calabria flight's data"""
    plc_file = os.path.join(flight_path, 'PLC_processed_data.csv')
    
    if not os.path.exists(plc_file):
        print(f"Warning: PLC file not found for {os.path.basename(flight_path)}")
        return None
    
    try:
        # Load PLC data
        plc_data = pd.read_csv(plc_file)
        plc_data['_time'] = pd.to_datetime(plc_data['_time'])
        
        # Calculate TOTAL force by summing ALL force columns
        force_columns = [col for col in plc_data.columns if 'force_' in col and col != '_time']
        print(f"  Force columns found: {force_columns}")
        
        # Calculate total force (sum of all force components)
        plc_data['total_force'] = plc_data[force_columns].sum(axis=1)
        
        # Individual force components for analysis
        plc_data['Fl'] = plc_data['force_back_left']  # Left back force
        plc_data['Fr'] = plc_data['force_back_right']  # Right back force
        
        # Back force analysis
        plc_data['back_force_sum'] = plc_data['Fl'] + plc_data['Fr']
        plc_data['back_force_diff'] = abs(plc_data['Fl'] - plc_data['Fr'])
        
        # Calculate flight duration
        duration = (plc_data['_time'].max() - plc_data['_time'].min()).total_seconds()
        
        # Extract flight info from path
        flight_name = os.path.basename(flight_path)
        
        # Parse date and metadata
        if '2025_01_21' in flight_name:
            date_str = "Jan 21, 2025"
            location = "Calabria"
        elif '2025_05_27' in flight_name:
            date_str = "May 27, 2025"
            location = "Calabria"
        elif '2025_05_29' in flight_name:
            date_str = "May 29, 2025"
            location = "Calabria"
        else:
            date_str = "2025"
            location = "Calabria"
        
        # Extract payload info if available
        payload_info = ""
        if "kg" in flight_name:
            parts = flight_name.split('_')
            for part in parts:
                if 'kg' in part:
                    payload_info = f" ({part.replace('kg', 'kg payload')})"
        
        return {
            'name': flight_name,
            'display_name': flight_name.split('_12mq_RRD')[0].replace('2025_', '').replace('_', '/'),
            'data': plc_data,
            'duration': duration,
            'location': location,
            'date': date_str,
            'payload_info': payload_info,
            'n_points': len(plc_data),
            'force_columns': force_columns
        }
    
    except Exception as e:
        print(f"Error processing {flight_path}: {e}")
        return None

def compute_moving_averages_and_max(series, t_width_list, sampling_rate_hz=10):
    """
    Compute moving averages for different time windows and return maximum values
    """
    av_values = []
    
    for window_sec in t_width_list:
        if window_sec == 0:
            # No averaging - use original data
            max_val = series.max()
        else:
            # Convert seconds to number of samples
            window_samples = max(1, int(window_sec * sampling_rate_hz))
            
            # Compute rolling average
            averaged_series = series.rolling(window=window_samples, center=True, min_periods=1).mean()
            
            # Get maximum of the averaged series
            max_val = averaged_series.max()
        
        av_values.append(max_val)
    
    return av_values

def analyze_single_flight_moving_averages(flight_data):
    """Analyze a single flight with moving averages across time windows"""
    
    data = flight_data['data'].copy()
    
    # Calculate metrics for each time window using moving averages
    results = {
        't_width': TIME_WINDOWS,
        'Fl_max': compute_moving_averages_and_max(abs(data['Fl']), TIME_WINDOWS),
        'Fr_max': compute_moving_averages_and_max(abs(data['Fr']), TIME_WINDOWS),
        'force_diff_max': compute_moving_averages_and_max(data['back_force_diff'], TIME_WINDOWS),
        'total_force_max': compute_moving_averages_and_max(data['total_force'], TIME_WINDOWS)
    }
    
    return results

def create_individual_flight_analysis(flight_data, output_dir):
    """Create individual flight analysis plots (like July17_Flight_*_analysis_*.png)"""
    
    # Analyze with moving averages
    analysis = analyze_single_flight_moving_averages(flight_data)
    
    flight_name = flight_data['display_name']
    duration_str = f"{flight_data['duration']/60:.1f}min"
    
    # Create plots for both linear and log scale
    for scale_type in ['linear', 'log']:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        title = f"Calabria Flight Analysis: {flight_name} | {flight_data['date']} | Duration: {duration_str}{flight_data['payload_info']}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 1. Left vs Right Back Forces
        ax1 = axes[0, 0]
        ax1.plot(analysis['t_width'], analysis['Fl_max'], 'o-', label='Left Back Force (Fl)', linewidth=2, markersize=6)
        ax1.plot(analysis['t_width'], analysis['Fr_max'], 'o-', label='Right Back Force (Fr)', linewidth=2, markersize=6)
        ax1.set_xlabel('Time Window (t_width) [s]')
        ax1.set_ylabel('Maximum Force [N]')
        ax1.set_title('Left vs Right Back Forces')
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        if scale_type == 'log':
            ax1.set_yscale('log')
            ax1.set_xscale('log')
            ax1.set_xlim(1, max(TIME_WINDOWS))
        else:
            ax1.set_ylim(0, 20)
        
        # 2. Force Difference
        ax2 = axes[0, 1]
        ax2.plot(analysis['t_width'], analysis['force_diff_max'], 'o-', color='purple', linewidth=2, markersize=6)
        ax2.set_xlabel('Time Window (t_width) [s]')
        ax2.set_ylabel('Maximum |Fl - Fr| [N]')
        ax2.set_title('Force Difference Analysis')
        ax2.grid(True, alpha=0.4)
        if scale_type == 'log':
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.set_xlim(1, max(TIME_WINDOWS))
        else:
            ax2.set_ylim(0, 20)
        
        # 3. Total Force
        ax3 = axes[1, 0]
        ax3.plot(analysis['t_width'], analysis['total_force_max'], 'o-', color='red', linewidth=2, markersize=6)
        ax3.set_xlabel('Time Window (t_width) [s]')
        ax3.set_ylabel('Maximum Total Force [N]')
        ax3.set_title('Total Force Analysis (Sum of All Forces)')
        ax3.grid(True, alpha=0.4)
        if scale_type == 'log':
            ax3.set_yscale('log')
            ax3.set_xscale('log')
            ax3.set_xlim(1, max(TIME_WINDOWS))
        else:
            ax3.set_ylim(0, 50)  # Higher limit for total force
        
        # 4. Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate summary stats
        data = flight_data['data']
        stats_text = f"""
Flight Summary:
Duration: {flight_data['duration']/60:.1f} minutes
Data Points: {flight_data['n_points']:,}
Sampling Rate: ~{flight_data['n_points']/(flight_data['duration']/60)/60:.1f} Hz

Force Analysis:
Max Total Force: {data['total_force'].max():.2f} N
Mean Total Force: {data['total_force'].mean():.2f} N
Max Left Back Force: {data['Fl'].max():.2f} N
Max Right Back Force: {data['Fr'].max():.2f} N
Max Force Difference: {data['back_force_diff'].max():.2f} N

Force Components Used:
{', '.join(flight_data['force_columns'])}
        """
        
        ax4.text(0.05, 0.95, stats_text.strip(), transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        # Style all axes
        for ax in [ax1, ax2, ax3]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(output_dir, f"Calabria_{flight_name.replace('/', '_')}_analysis_{scale_type}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {os.path.basename(output_file)}")
        
        plt.close()

def create_combined_group_plots(all_flights_data, flights_group_name, output_dir):
    """Create combined plots for all flights in a group"""
    
    # Prepare data for plotting
    plot_data = {
        't_width': TIME_WINDOWS,
        'flights': {}
    }
    
    for flight in all_flights_data:
        if flight is None:
            continue
            
        flight_label = flight['display_name']
        duration_str = f"{flight['duration']/60:.1f}min"
        
        analysis = analyze_single_flight_moving_averages(flight)
        
        plot_data['flights'][flight_label] = {
            'Fl_max': analysis['Fl_max'],
            'Fr_max': analysis['Fr_max'],
            'force_diff_max': analysis['force_diff_max'],
            'total_force_max': analysis['total_force_max'],
            'duration': duration_str,
            'payload_info': flight['payload_info']
        }
    
    # Create the three combined plots
    plot_configs = [
        {
            'title': f'{flights_group_name} - Back Forces Analysis',
            'filename': f'calabria_{flights_group_name.lower().replace(" ", "_")}_back_force_analysis_CORRECTED',
            'metrics': ['Fl_max', 'Fr_max'],
            'ylabel': 'Maximum Force [N]',
            'legend_labels': ['Left Back Force (Fl)', 'Right Back Force (Fr)']
        },
        {
            'title': f'{flights_group_name} - Force Difference Analysis', 
            'filename': f'calabria_{flights_group_name.lower().replace(" ", "_")}_force_diff_analysis_CORRECTED',
            'metrics': ['force_diff_max'],
            'ylabel': 'Maximum |Fl - Fr| [N]',
            'legend_labels': ['Force Difference |Fl-Fr|']
        },
        {
            'title': f'{flights_group_name} - Total Force Analysis (All Forces)', 
            'filename': f'calabria_{flights_group_name.lower().replace(" ", "_")}_total_force_analysis_CORRECTED',
            'metrics': ['total_force_max'],
            'ylabel': 'Maximum Total Force [N]',
            'legend_labels': ['Total Force (Sum of All)']
        }
    ]
    
    for config in plot_configs:
        # Create both linear and log scale plots
        for scale_type in ['linear', 'log']:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(plot_data['flights'])))
            
            for i, (flight_label, flight_data) in enumerate(plot_data['flights'].items()):
                for j, metric in enumerate(config['metrics']):
                    line_style = '-' if j == 0 else '--'
                    label = f"{flight_label}{flight_data['payload_info']} - {config['legend_labels'][j]} ({flight_data['duration']})"
                    
                    if len(config['metrics']) == 1:
                        label = f"{flight_label}{flight_data['payload_info']} ({flight_data['duration']})"
                    
                    ax.plot(plot_data['t_width'], flight_data[metric], 
                           marker='o', linestyle=line_style, linewidth=2, markersize=6,
                           color=colors[i] if len(config['metrics']) == 1 else None,
                           label=label)
            
            ax.set_xlabel('Time Window (t_width) [s]', fontweight='bold')
            ax.set_ylabel(config['ylabel'], fontweight='bold')
            ax.set_title(f"{config['title']} ({'Log Scale' if scale_type == 'log' else 'Linear Scale'})", 
                        fontweight='bold', pad=20)
            
            if scale_type == 'log':
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_xlim(1, max(TIME_WINDOWS))
            else:
                # Fixed y-axis range for better comparison
                if 'total_force' in config['filename']:
                    ax.set_ylim(0, 50)  # Higher limit for total force
                else:
                    ax.set_ylim(0, 20)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.4)
            
            # Style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            
            # Save plot
            output_file = f"{config['filename']}_{scale_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  Saved: {output_file}")
            
            plt.close()

def main():
    """Main analysis function"""
    
    # Define flight groups
    flight_groups = {
        'Jan_2025': [
            '2025_01_21_12_23_00_12mq_RRD',
            '2025_01_21_13_38_05_12mq_RRD', 
            '2025_01_21_13_52_00_12mq_RRD'
        ],
        'May_2025': [
            '2025_05_27_12_13_22_12mq_RRD_00kg_00m',
            '2025_05_27_13_22_16_12mq_RRD_06kg_15m',
            '2025_05_29_06_04_38_12mq_RRD_12kg_35m',
            '2025_05_29_06_48_38_12mq_RRD_15kg_35m'
        ]
    }
    
    base_path = '/Users/baharakqaderi/field-data-pipeline/flight_analysis_catanzaro'
    
    print("Starting Comprehensive Calabria Flights Analysis...")
    print("="*70)
    print("Key corrections:")
    print("- Total force = SUM OF ALL force columns from PLC data")
    print("- Individual flight analysis plots for each flight")
    print("- Combined group analysis plots")
    print("="*70)
    
    # Process each flight group
    for group_name, flight_names in flight_groups.items():
        print(f"\\nProcessing {group_name} flights...")
        
        all_flights_data = []
        
        for flight_name in flight_names:
            flight_path = os.path.join(base_path, flight_name)
            print(f"\\n  Loading {flight_name}...")
            
            flight_data = load_and_process_flight_data(flight_path)
            
            if flight_data:
                print(f"    Duration: {flight_data['duration']/60:.1f} minutes")
                print(f"    Data points: {flight_data['n_points']:,}")
                print(f"    Total force range: {flight_data['data']['total_force'].min():.2f} - {flight_data['data']['total_force'].max():.2f} N")
                
                # Create individual flight analysis
                print(f"    Creating individual analysis plots...")
                create_individual_flight_analysis(flight_data, '.')
                
                all_flights_data.append(flight_data)
            else:
                all_flights_data.append(None)
        
        # Create combined plots for this group
        if any(f is not None for f in all_flights_data):
            print(f"\\n  Creating combined plots for {group_name}...")
            create_combined_group_plots(all_flights_data, group_name.replace('_', ' '), '.')
        else:
            print(f"  No valid data found for {group_name}")
    
    print("\\n" + "="*70)
    print("Comprehensive Calabria flights analysis complete!")
    print("\\nGenerated files:")
    print("\\nIndividual flight analyses:")
    print("- Calabria_*_analysis_linear.png (individual flight plots)")
    print("- Calabria_*_analysis_log.png (individual flight plots)")
    print("\\nCombined group analyses:")
    print("- *_back_force_analysis_CORRECTED_*.png (Fl vs Fr)")
    print("- *_force_diff_analysis_CORRECTED_*.png (|Fl-Fr|)")
    print("- *_total_force_analysis_CORRECTED_*.png (Sum of ALL forces)")
    print("\\nAll plots available in both linear and log scale")
    print("Total force now CORRECTLY calculated as sum of all force columns!")

if __name__ == "__main__":
    main()
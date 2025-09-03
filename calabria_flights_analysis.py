#!/usr/bin/env python3
"""
Calabria Flights Force Analysis

Similar to Vaie analysis but for Calabria flights from Jan 2025 and May 2025.
Creates graphs 2, 3, and 4 (no torque data available for these flights):

Graph 2: t_width vs Left/Right back forces (Fl/Fr)
Graph 3: t_width vs abs(Fl-Fr) 
Graph 4: t_width vs Fl+Fr

Flights analyzed:
- Jan 2025: 2025_01_21_12_23_00, 2025_01_21_13_38_05, 2025_01_21_13_52_00
- May 2025: 2025_05_27_12_13_22, 2025_05_27_13_22_16, 2025_05_29_06_04_38, 2025_05_29_06_48_38
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from pathlib import Path

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

# Define time windows for moving average analysis (same as Vaie analysis)
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
        
        # Calculate flight duration
        duration = (plc_data['_time'].max() - plc_data['_time'].min()).total_seconds()
        
        # Extract flight info from path
        flight_name = os.path.basename(flight_path)
        
        # Parse date and time
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
        
        return {
            'name': flight_name,
            'data': plc_data,
            'duration': duration,
            'location': location,
            'date': date_str,
            'n_points': len(plc_data)
        }
    
    except Exception as e:
        print(f"Error processing {flight_path}: {e}")
        return None

def analyze_single_flight(flight_data):
    """Analyze a single flight with moving averages across time windows"""
    
    data = flight_data['data'].copy()
    
    # Calculate metrics for each time window
    results = {}
    
    for t_width in TIME_WINDOWS:
        if t_width == 0:
            # No averaging - use instantaneous values
            fl_values = data['force_back_left'].values
            fr_values = data['force_back_right'].values
        else:
            # Apply moving average
            window_size = max(1, int(t_width * 10))  # Assuming ~10Hz sampling
            fl_values = data['force_back_left'].rolling(window=window_size, center=True, min_periods=1).mean().values
            fr_values = data['force_back_right'].rolling(window=window_size, center=True, min_periods=1).mean().values
        
        # Calculate metrics (using maximum values as in original Vaie analysis)
        fl_max = np.max(np.abs(fl_values))
        fr_max = np.max(np.abs(fr_values))
        force_diff_max = np.max(np.abs(fl_values - fr_values))
        total_force_max = np.max(fl_values + fr_values)
        
        results[t_width] = {
            'Fl_max': fl_max,
            'Fr_max': fr_max, 
            'force_diff_max': force_diff_max,
            'total_force_max': total_force_max
        }
    
    return results

def create_combined_plots(all_flights_data, flights_group_name, output_dir):
    """Create combined plots for all flights in a group"""
    
    # Prepare data for plotting
    plot_data = {
        't_width': TIME_WINDOWS,
        'flights': {}
    }
    
    for flight in all_flights_data:
        if flight is None:
            continue
            
        flight_label = flight['name'].split('_12mq_RRD')[0].replace('2025_', '').replace('_', '/')
        
        analysis = analyze_single_flight(flight)
        
        plot_data['flights'][flight_label] = {
            'Fl_max': [analysis[t]['Fl_max'] for t in TIME_WINDOWS],
            'Fr_max': [analysis[t]['Fr_max'] for t in TIME_WINDOWS], 
            'force_diff_max': [analysis[t]['force_diff_max'] for t in TIME_WINDOWS],
            'total_force_max': [analysis[t]['total_force_max'] for t in TIME_WINDOWS],
            'duration': f"{flight['duration']/60:.1f}min"
        }
    
    # Create the three plots
    plot_configs = [
        {
            'title': f'{flights_group_name} - Back Forces Analysis',
            'filename': f'calabria_{flights_group_name.lower().replace(" ", "_")}_back_force_analysis',
            'metrics': ['Fl_max', 'Fr_max'],
            'ylabel': 'Maximum Force [N]',
            'legend_labels': ['Left Back Force (Fl)', 'Right Back Force (Fr)']
        },
        {
            'title': f'{flights_group_name} - Force Difference Analysis', 
            'filename': f'calabria_{flights_group_name.lower().replace(" ", "_")}_force_diff_analysis',
            'metrics': ['force_diff_max'],
            'ylabel': 'Maximum |Fl - Fr| [N]',
            'legend_labels': ['Force Difference |Fl-Fr|']
        },
        {
            'title': f'{flights_group_name} - Total Force Analysis',
            'filename': f'calabria_{flights_group_name.lower().replace(" ", "_")}_total_force_analysis', 
            'metrics': ['total_force_max'],
            'ylabel': 'Maximum Fl + Fr [N]',
            'legend_labels': ['Total Force Fl+Fr']
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
                    label = f"{flight_label} - {config['legend_labels'][j]} ({flight_data['duration']})"
                    
                    if len(config['metrics']) == 1:
                        label = f"{flight_label} ({flight_data['duration']})"
                    
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
            print(f"Saved: {output_file}")
            
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
    
    print("Starting Calabria Flights Force Analysis...")
    print("="*60)
    
    # Process each flight group
    for group_name, flight_names in flight_groups.items():
        print(f"\\nProcessing {group_name} flights...")
        
        all_flights_data = []
        
        for flight_name in flight_names:
            flight_path = os.path.join(base_path, flight_name)
            print(f"  Loading {flight_name}...")
            
            flight_data = load_and_process_flight_data(flight_path)
            all_flights_data.append(flight_data)
            
            if flight_data:
                print(f"    Duration: {flight_data['duration']/60:.1f} minutes")
                print(f"    Data points: {flight_data['n_points']:,}")
        
        # Create combined plots for this group
        if any(f is not None for f in all_flights_data):
            print(f"\\nCreating combined plots for {group_name}...")
            create_combined_plots(all_flights_data, group_name.replace('_', ' '), '.')
        else:
            print(f"No valid data found for {group_name}")
    
    print("\\n" + "="*60)
    print("Calabria flights analysis complete!")
    print("\\nGenerated plots:")
    print("- Back force analysis (Fl vs Fr) for both groups")
    print("- Force difference analysis (|Fl-Fr|) for both groups") 
    print("- Total force analysis (Fl+Fr) for both groups")
    print("- Each plot available in both linear and log scale")

if __name__ == "__main__":
    main()
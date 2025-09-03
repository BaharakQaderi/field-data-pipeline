#!/usr/bin/env python3
"""
Fixed Calabria Flights Analysis

Key fixes:
- Forces are already in kg - NO conversion
- Total force = simple sum of all force columns
- No complex processing - direct values
- Fix missing values in back force plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from pathlib import Path
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

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

# Time windows for moving average analysis
TIME_WINDOWS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 60, 90, 120, 180]

def load_and_process_flight_data(flight_path):
    """Load and process a single Calabria flight's data - SIMPLE VERSION"""
    plc_file = os.path.join(flight_path, 'PLC_processed_data.csv')
    
    if not os.path.exists(plc_file):
        print(f"Warning: PLC file not found for {os.path.basename(flight_path)}")
        return None
    
    try:
        # Load PLC data
        plc_data = pd.read_csv(plc_file)
        plc_data['_time'] = pd.to_datetime(plc_data['_time'])
        
        # SIMPLE: Just sum all force columns directly (no conversion - already in kg)
        force_columns = [col for col in plc_data.columns if 'force_' in col]
        print(f"  Force columns: {force_columns}")
        
        # Simple sum - forces already in kg
        plc_data['total_force'] = plc_data[force_columns].sum(axis=1)
        
        # Individual components
        plc_data['Fl'] = plc_data['force_back_left']
        plc_data['Fr'] = plc_data['force_back_right']
        plc_data['force_diff'] = abs(plc_data['Fl'] - plc_data['Fr'])
        plc_data['back_force_sum'] = plc_data['Fl'] + plc_data['Fr']  # Total back force (Fr + Fl)
        
        # Calculate duration
        duration = (plc_data['_time'].max() - plc_data['_time'].min()).total_seconds()
        
        flight_name = os.path.basename(flight_path)
        
        # Extract date info
        if '2025_01_21' in flight_name:
            date_str = "Jan 21, 2025"
        elif '2025_05_27' in flight_name:
            date_str = "May 27, 2025" 
        elif '2025_05_29' in flight_name:
            date_str = "May 29, 2025"
        else:
            date_str = "2025"
        
        return {
            'name': flight_name,
            'display_name': flight_name.split('_12mq_RRD')[0].replace('2025_', '').replace('_', '/'),
            'data': plc_data,
            'duration': duration,
            'date': date_str,
            'n_points': len(plc_data),
            'force_columns': force_columns
        }
    
    except Exception as e:
        print(f"Error processing {flight_path}: {e}")
        return None

def analyze_single_flight_simple(flight_data):
    """Simple analysis without complex moving averages"""
    
    data = flight_data['data']
    
    results = {}
    
    for t_width in TIME_WINDOWS:
        if t_width == 0:
            # No averaging - use direct max values
            fl_max = data['Fl'].max()
            fr_max = data['Fr'].max()
            force_diff_max = data['force_diff'].max()
            total_force_max = data['total_force'].max()
            back_force_sum_max = data['back_force_sum'].max()
        else:
            # Simple rolling window (assuming ~10Hz sampling)
            window_size = max(1, int(t_width * 10))
            
            fl_avg = data['Fl'].rolling(window=window_size, center=True, min_periods=1).mean()
            fr_avg = data['Fr'].rolling(window=window_size, center=True, min_periods=1).mean()
            total_avg = data['total_force'].rolling(window=window_size, center=True, min_periods=1).mean()
            back_sum_avg = data['back_force_sum'].rolling(window=window_size, center=True, min_periods=1).mean()
            
            fl_max = fl_avg.max()
            fr_max = fr_avg.max()
            force_diff_max = abs(fl_avg - fr_avg).max()
            total_force_max = total_avg.max()
            back_force_sum_max = back_sum_avg.max()
        
        results[t_width] = {
            'Fl_max': fl_max,
            'Fr_max': fr_max,
            'force_diff_max': force_diff_max,
            'total_force_max': total_force_max,
            'back_force_sum_max': back_force_sum_max
        }
    
    return results

def create_individual_flight_analysis(flight_data, output_dir):
    """Create individual flight analysis plots"""
    
    analysis = analyze_single_flight_simple(flight_data)
    
    flight_name = flight_data['display_name']
    duration_str = f"{flight_data['duration']/60:.1f}min"
    
    # Extract payload info
    payload_info = ""
    if "kg" in flight_data['name'] and "m" in flight_data['name']:
        parts = flight_data['name'].split('_')
        kg_part = [p for p in parts if 'kg' in p]
        m_part = [p for p in parts if 'm' in p and p != '12mq']
        if kg_part and m_part:
            payload_info = f" ({kg_part[0]}, {m_part[0]})"
    
    # Get the data for statistics
    data = flight_data['data']
    
    # Create plots for both linear and log scale
    for scale_type in ['linear', 'log']:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        title = f"Calabria Flight: {flight_name} | {flight_data['date']} | {duration_str}{payload_info}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Extract data for plotting
        t_widths = TIME_WINDOWS
        fl_values = [analysis[t]['Fl_max'] for t in t_widths]
        fr_values = [analysis[t]['Fr_max'] for t in t_widths]
        diff_values = [analysis[t]['force_diff_max'] for t in t_widths]
        total_values = [analysis[t]['total_force_max'] for t in t_widths]
        back_sum_values = [analysis[t]['back_force_sum_max'] for t in t_widths]
        
        # 1. Left vs Right Back Forces
        ax1 = axes[0, 0]
        
        if scale_type == 'log':
            # Filter out zero or negative values
            valid_t_widths = [t for t in t_widths if t > 0]
            valid_fl_indices = [i for i, t in enumerate(t_widths) if t > 0]
            valid_fr_indices = [i for i, t in enumerate(t_widths) if t > 0]
            
            valid_fl_values = [fl_values[i] for i in valid_fl_indices]
            valid_fr_values = [fr_values[i] for i in valid_fr_indices]
            
            # Plot with logarithmic x-axis
            ax1.plot(valid_t_widths, valid_fl_values, 'o-', label='Left Back Force (Fl)', 
                    linewidth=2, markersize=6, color='blue')
            ax1.plot(valid_t_widths, valid_fr_values, 'o-', label='Right Back Force (Fr)', 
                    linewidth=2, markersize=6, color='red')
            
            # Linear scale with appropriate tick spacing
            ax1.set_xscale('log')  # Keep x-axis logarithmic
            ax1.set_xlim(1, max(TIME_WINDOWS))
            
            # Keep y-axis linear for clear labeling
            ax1.set_ylim(0, max(max(fl_values), max(fr_values)) * 1.1)
        else:
            # Standard linear plot
            ax1.plot(t_widths, fl_values, 'o-', label='Left Back Force (Fl)', 
                    linewidth=2, markersize=6, color='blue')
            ax1.plot(t_widths, fr_values, 'o-', label='Right Back Force (Fr)', 
                    linewidth=2, markersize=6, color='red')
            ax1.set_ylim(0, max(max(fl_values), max(fr_values)) * 1.1)
            
        ax1.set_xlabel('Time Window [s]')
        ax1.set_ylabel('Maximum Force [kg]')
        ax1.set_title('Left vs Right Back Forces')
        ax1.legend()
        
        # 2. Force Difference
        ax2 = axes[0, 1]
        
        if scale_type == 'log':
            # Filter out zero or negative values for log scale
            valid_t_widths = [t for t in t_widths if t > 0]
            valid_diff_indices = [i for i, t in enumerate(t_widths) if t > 0]
            valid_diff_values = [diff_values[i] for i in valid_diff_indices]
            
            # Plot with logarithmic x-axis
            ax2.plot(valid_t_widths, valid_diff_values, 'o-', color='purple', 
                    linewidth=2, markersize=6)
            
            # Set x-axis to log scale
            ax2.set_xscale('log')
            ax2.set_xlim(1, max(TIME_WINDOWS))
            
            # Keep y-axis linear for clear labeling
            ax2.set_ylim(0, max(diff_values) * 1.1)
        else:
            # Standard linear plot
            ax2.plot(t_widths, diff_values, 'o-', color='purple', 
                    linewidth=2, markersize=6)
            ax2.set_ylim(0, max(diff_values) * 1.1)
        
        ax2.set_xlabel('Time Window [s]')
        ax2.set_ylabel('Maximum |Fl - Fr| [kg]')
        ax2.set_title('Force Difference Analysis')
        
        # 3. Total Force
        ax3 = axes[1, 0]
        
        if scale_type == 'log':
            # Filter out zero or negative values for log scale
            valid_t_widths = [t for t in t_widths if t > 0]
            valid_total_indices = [i for i, t in enumerate(t_widths) if t > 0]
            valid_total_values = [total_values[i] for i in valid_total_indices]
            
            # Plot with logarithmic x-axis
            ax3.plot(valid_t_widths, valid_total_values, 'o-', color='green', 
                    linewidth=2, markersize=6)
            
            # Set x-axis to log scale
            ax3.set_xscale('log')
            ax3.set_xlim(1, max(TIME_WINDOWS))
            
            # Keep y-axis linear for clear labeling
            ax3.set_ylim(0, max(total_values) * 1.1)
        else:
            # Standard linear plot
            ax3.plot(t_widths, total_values, 'o-', color='green', 
                    linewidth=2, markersize=6)
            ax3.set_ylim(0, max(total_values) * 1.1)
        
        ax3.set_xlabel('Time Window [s]')
        ax3.set_ylabel('Maximum Total Force [kg]')
        ax3.set_title('Total Force (Sum of All Components)')
        
        # 4. Total Back Force (Fr + Fl)
        ax4 = axes[1, 1]
        
        if scale_type == 'log':
            # Filter out zero or negative values for log scale
            valid_t_widths = [t for t in t_widths if t > 0]
            valid_back_indices = [i for i, t in enumerate(t_widths) if t > 0]
            valid_back_values = [back_sum_values[i] for i in valid_back_indices]
            
            # Plot with logarithmic x-axis
            ax4.plot(valid_t_widths, valid_back_values, 'o-', color='orange', 
                    linewidth=2, markersize=6)
            
            # Set x-axis to log scale
            ax4.set_xscale('log')
            ax4.set_xlim(1, max(TIME_WINDOWS))
            
            # Keep y-axis linear for clear labeling
            ax4.set_ylim(0, max(back_sum_values) * 1.1)
        else:
            # Standard linear plot
            ax4.plot(t_widths, back_sum_values, 'o-', color='orange', 
                    linewidth=2, markersize=6)
            ax4.set_ylim(0, max(back_sum_values) * 1.1)
        
        ax4.set_xlabel('Time Window [s]')
        ax4.set_ylabel('Maximum Back Force Sum [kg]')
        ax4.set_title('Total Back Force (Fr + Fl)')
        
        # 5. Data Summary
        ax5 = axes[1, 2]
        ax5.axis('off')
        
        # Raw statistics
        stats_text = f"""
Flight Summary:
Duration: {flight_data['duration']/60:.1f} minutes
Data Points: {flight_data['n_points']:,}

Force Statistics (kg):
Left Back (Fl):    {data['Fl'].min():.2f} - {data['Fl'].max():.2f} (avg: {data['Fl'].mean():.2f})
Right Back (Fr):   {data['Fr'].min():.2f} - {data['Fr'].max():.2f} (avg: {data['Fr'].mean():.2f})
Back Sum (Fl+Fr):  {data['back_force_sum'].min():.2f} - {data['back_force_sum'].max():.2f} (avg: {data['back_force_sum'].mean():.2f})
Force Diff:        {data['force_diff'].min():.2f} - {data['force_diff'].max():.2f} (avg: {data['force_diff'].mean():.2f})
Total Force:       {data['total_force'].min():.2f} - {data['total_force'].max():.2f} (avg: {data['total_force'].mean():.2f})

Components:
{', '.join([col.replace('force_', '') for col in flight_data['force_columns']])}

Scale: {scale_type.upper()}
        """
        
        ax5.text(0.05, 0.95, stats_text.strip(), transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.7))
        
        # Style and apply grid AFTER all other styling
        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Make grid more visible with explicit colors and higher alpha
            ax.grid(True, alpha=0.7, linewidth=1.0, color='gray')
            ax.minorticks_on()
            ax.grid(which='minor', alpha=0.4, linewidth=0.6, color='lightgray')
        
        plt.tight_layout()
        
        # Save
        safe_name = flight_name.replace('/', '_')
        output_file = f"Calabria_FIXED_{safe_name}_analysis_{scale_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {output_file}")
        
        plt.close()

def create_combined_plots(all_flights_data, group_name):
    """Create combined plots for a group of flights"""
    
    valid_flights = [f for f in all_flights_data if f is not None]
    if not valid_flights:
        return
    
    # Prepare combined data
    plot_data = {
        't_width': TIME_WINDOWS,
        'flights': {}
    }
    
    for flight in valid_flights:
        analysis = analyze_single_flight_simple(flight)
        flight_label = flight['display_name']
        duration_str = f"{flight['duration']/60:.1f}min"
        
        # Add payload info if available
        payload_info = ""
        if "kg" in flight['name'] and "m" in flight['name']:
            parts = flight['name'].split('_')
            kg_part = [p for p in parts if 'kg' in p]
            if kg_part:
                payload_info = f" ({kg_part[0]})"
        
        plot_data['flights'][flight_label] = {
            'Fl_max': [analysis[t]['Fl_max'] for t in TIME_WINDOWS],
            'Fr_max': [analysis[t]['Fr_max'] for t in TIME_WINDOWS],
            'force_diff_max': [analysis[t]['force_diff_max'] for t in TIME_WINDOWS],
            'total_force_max': [analysis[t]['total_force_max'] for t in TIME_WINDOWS],
            'back_force_sum_max': [analysis[t]['back_force_sum_max'] for t in TIME_WINDOWS],
            'duration': duration_str,
            'payload': payload_info
        }
    
    # Create the four graph types
    plot_configs = [
        {
            'title': f'Calabria {group_name} - Back Forces (Fl vs Fr)',
            'filename': f'calabria_{group_name.lower()}_back_forces_FIXED',
            'data_keys': ['Fl_max', 'Fr_max'],
            'ylabel': 'Maximum Force [kg]',
            'colors': ['blue', 'red'],
            'labels': ['Left Back (Fl)', 'Right Back (Fr)']
        },
        {
            'title': f'Calabria {group_name} - Force Difference |Fl-Fr|',
            'filename': f'calabria_{group_name.lower()}_force_diff_FIXED',
            'data_keys': ['force_diff_max'],
            'ylabel': 'Maximum |Fl-Fr| [kg]',
            'colors': ['purple'],
            'labels': ['Force Difference']
        },
        {
            'title': f'Calabria {group_name} - Total Force (Sum of All)',
            'filename': f'calabria_{group_name.lower()}_total_force_FIXED',
            'data_keys': ['total_force_max'],
            'ylabel': 'Maximum Total Force [kg]',
            'colors': ['green'],
            'labels': ['Total Force']
        },
        {
            'title': f'Calabria {group_name} - Total Back Force (Fr + Fl)',
            'filename': f'calabria_{group_name.lower()}_back_force_sum_FIXED',
            'data_keys': ['back_force_sum_max'],
            'ylabel': 'Maximum Back Force Sum [kg]',
            'colors': ['orange'],
            'labels': ['Back Force Sum']
        }
    ]
    
    for config in plot_configs:
        for scale_type in ['linear', 'log']:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Plot each flight
            flight_colors = plt.cm.Set1(np.linspace(0, 1, len(plot_data['flights'])))
            
            for i, (flight_name, flight_data) in enumerate(plot_data['flights'].items()):
                for j, data_key in enumerate(config['data_keys']):
                    if len(config['data_keys']) == 1:
                        # Single metric - use different color per flight
                        color = flight_colors[i]
                        label = f"{flight_name}{flight_data['payload']} ({flight_data['duration']})"
                    else:
                        # Multiple metrics - use config colors
                        color = config['colors'][j]
                        label = f"{flight_name} - {config['labels'][j]} ({flight_data['duration']})"
                    
                    ax.plot(plot_data['t_width'], flight_data[data_key], 
                           'o-', color=color, linewidth=2, markersize=6, 
                           label=label, alpha=0.8)
            
            ax.set_xlabel('Time Window [s]', fontweight='bold')
            ax.set_ylabel(config['ylabel'], fontweight='bold')
            ax.set_title(f"{config['title']} ({'Log Scale' if scale_type == 'log' else 'Linear Scale'})", 
                        fontweight='bold', pad=20)
            
            if scale_type == 'log':
                # Keep x-axis logarithmic
                ax.set_xscale('log')
                ax.set_xlim(1, max(TIME_WINDOWS))
                
                # Filter plots to remove zero values from logarithmic plot
                for line in ax.get_lines():
                    xdata, ydata = line.get_data()
                    # Filter out x=0 points
                    valid_indices = [i for i, x in enumerate(xdata) if x > 0]
                    valid_x = [xdata[i] for i in valid_indices]
                    valid_y = [ydata[i] for i in valid_indices]
                    line.set_data(valid_x, valid_y)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Apply grid AFTER all styling to ensure visibility
            ax.grid(True, alpha=0.7, linewidth=1.0, color='gray')
            ax.minorticks_on()
            ax.grid(which='minor', alpha=0.4, linewidth=0.6, color='lightgray')
            
            plt.tight_layout()
            
            output_file = f"{config['filename']}_{scale_type}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  Saved: {output_file}")
            
            plt.close()

def main():
    """Main function"""
    
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
    
    print("FIXED Calabria Flights Analysis")
    print("="*50)
    print("✅ Forces already in kg - no conversion")
    print("✅ Simple sum for total forces")
    print("✅ Direct max values - no weird processing")
    print("="*50)
    
    for group_name, flight_names in flight_groups.items():
        print(f"\\nProcessing {group_name}...")
        
        all_flights_data = []
        
        for flight_name in flight_names:
            flight_path = os.path.join(base_path, flight_name)
            print(f"\\n  {flight_name}:")
            
            flight_data = load_and_process_flight_data(flight_path)
            
            if flight_data:
                data = flight_data['data']
                print(f"    Duration: {flight_data['duration']/60:.1f} min")
                print(f"    Left back force: {data['Fl'].min():.2f} - {data['Fl'].max():.2f} kg")
                print(f"    Right back force: {data['Fr'].min():.2f} - {data['Fr'].max():.2f} kg")
                print(f"    Total force: {data['total_force'].min():.2f} - {data['total_force'].max():.2f} kg")
                
                create_individual_flight_analysis(flight_data, '.')
                all_flights_data.append(flight_data)
            else:
                all_flights_data.append(None)
        
        print(f"\\n  Creating combined plots for {group_name}...")
        create_combined_plots(all_flights_data, group_name)
    
    print("\\n" + "="*50)
    print("✅ FIXED Analysis Complete!")
    print("\\nGenerated:")
    print("- Individual flight plots: Calabria_FIXED_*_analysis_*.png")  
    print("- Combined group plots: calabria_*_FIXED_*.png")
    print("\\nAll values now show correctly in kg!")

if __name__ == "__main__":
    main()
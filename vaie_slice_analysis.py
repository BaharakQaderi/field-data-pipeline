#!/usr/bin/env python3
"""
Vaie Slice Analysis - Understanding Roll vs Position Relationship

Creates detailed slice visualizations to understand how roll percentage commands
relate to actual position changes over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import math

# System parameters
PULLEY_DIAMETER = 31.0  # mm
PULLEY_RADIUS = PULLEY_DIAMETER / 2  # 15.5 mm
CONVERSION_FACTOR = math.pi * PULLEY_RADIUS / 180  # mm per degree

def degrees_to_mm(degrees):
    """Convert servo position from degrees to millimeters"""
    radians = degrees * math.pi / 180
    mm = radians * PULLEY_RADIUS
    return mm

def load_and_prepare_data():
    """Load and prepare flight data"""
    flight_path = "/Users/baharakqaderi/field-data-pipeline/flight_analysis_july17/2025_07_17_11_30/flight_data.csv"
    
    df = pd.read_csv(flight_path)
    
    # Convert time
    if df['_time'].dtype == 'object':
        df['_time'] = pd.to_datetime(df['_time'], format='mixed')
    df['time_seconds'] = (df['_time'] - df['_time'].min()).dt.total_seconds()
    
    # Convert positions to mm
    df['left_position_mm'] = degrees_to_mm(df['FLIGHT_SEGMENT_left_servo_position'])
    df['right_position_mm'] = degrees_to_mm(df['FLIGHT_SEGMENT_right_servo_position'])
    
    return df

def create_slice_visualizations(df, slice_duration=30, num_slices=6):
    """Create detailed slice visualizations"""
    
    print(f"🔍 CREATING SLICE VISUALIZATIONS")
    print("=" * 40)
    
    total_duration = df['time_seconds'].max()
    slice_starts = np.linspace(0, total_duration - slice_duration, num_slices)
    
    fig, axes = plt.subplots(num_slices, 1, figsize=(20, 4 * num_slices))
    if num_slices == 1:
        axes = [axes]
    
    for i, start_time in enumerate(slice_starts):
        end_time = start_time + slice_duration
        
        # Extract slice data
        slice_mask = (df['time_seconds'] >= start_time) & (df['time_seconds'] <= end_time)
        slice_data = df[slice_mask].copy()
        
        if len(slice_data) == 0:
            continue
        
        ax = axes[i]
        
        # Create twin axes for different scales
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        
        # Offset the third y-axis
        ax3.spines['right'].set_position(('outward', 60))
        
        # Plot roll percentage (primary axis)
        line1 = ax.plot(slice_data['time_seconds'], slice_data['FLIGHT_SEGMENT_roll_percentage'], 
                       'g-', linewidth=3, alpha=0.8, label='Roll %', marker='o', markersize=3)
        ax.set_ylabel('Roll Percentage [%]', color='green', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='green')
        
        # Plot left servo position (secondary axis)
        line2 = ax2.plot(slice_data['time_seconds'], slice_data['left_position_mm'], 
                        'b-', linewidth=2, alpha=0.7, label='Left Position', marker='s', markersize=2)
        ax2.set_ylabel('Left Position [mm]', color='blue', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Plot right servo position (tertiary axis)
        line3 = ax3.plot(slice_data['time_seconds'], slice_data['right_position_mm'], 
                        'r-', linewidth=2, alpha=0.7, label='Right Position', marker='^', markersize=2)
        ax3.set_ylabel('Right Position [mm]', color='red', fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='red')
        
        # Set labels and title
        ax.set_xlabel('Time [seconds]')
        ax.set_title(f'Slice {i+1}: {start_time:.1f}s - {end_time:.1f}s | Points: {len(slice_data)}', 
                    fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        # Add statistics box
        roll_range = slice_data['FLIGHT_SEGMENT_roll_percentage'].max() - slice_data['FLIGHT_SEGMENT_roll_percentage'].min()
        left_range = slice_data['left_position_mm'].max() - slice_data['left_position_mm'].min()
        right_range = slice_data['right_position_mm'].max() - slice_data['right_position_mm'].min()
        
        stats_text = f'Roll range: {roll_range:.1f}%\nLeft range: {left_range:.1f}mm\nRight range: {right_range:.1f}mm'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        print(f"Slice {i+1}: {start_time:.1f}s-{end_time:.1f}s, {len(slice_data)} points, Roll range: {roll_range:.2f}%")
    
    plt.tight_layout()
    plt.suptitle('Roll Percentage vs Servo Positions - Detailed Slices\n'
                 'Understanding the relationship between commands and actual movements', 
                 fontsize=16, y=0.99)
    
    output_path = '/Users/baharakqaderi/field-data-pipeline/vaie_slice_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Created: vaie_slice_analysis.png")
    return output_path

def create_change_correlation_analysis(df):
    """Analyze correlation between roll changes and position changes"""
    
    print(f"\n🔍 ANALYZING CHANGE CORRELATIONS")
    print("=" * 35)
    
    # Calculate changes
    df['roll_change'] = df['FLIGHT_SEGMENT_roll_percentage'].diff()
    df['left_change'] = df['left_position_mm'].diff()
    df['right_change'] = df['right_position_mm'].diff()
    df['time_diff'] = df['time_seconds'].diff()
    
    # Filter out very small changes and time differences
    significant_changes = df[
        (abs(df['roll_change']) > 0.1) &  # Roll changes > 0.1%
        (df['time_diff'] > 0.001)  # Time diff > 1ms
    ].copy()
    
    print(f"Total data points: {len(df)}")
    print(f"Significant change points: {len(significant_changes)}")
    
    # Create change correlation visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Roll change vs Left position change
    ax1.scatter(significant_changes['roll_change'], significant_changes['left_change'], 
               alpha=0.6, s=20, color='blue')
    ax1.set_xlabel('Roll Change [%]')
    ax1.set_ylabel('Left Position Change [mm]')
    ax1.set_title('Roll Change vs Left Position Change')
    ax1.grid(True, alpha=0.6)
    
    # Calculate and show correlation
    corr1 = significant_changes['roll_change'].corr(significant_changes['left_change'])
    ax1.text(0.05, 0.95, f'Correlation: {corr1:.3f}', transform=ax1.transAxes, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Roll change vs Right position change
    ax2.scatter(significant_changes['roll_change'], significant_changes['right_change'], 
               alpha=0.6, s=20, color='red')
    ax2.set_xlabel('Roll Change [%]')
    ax2.set_ylabel('Right Position Change [mm]')
    ax2.set_title('Roll Change vs Right Position Change')
    ax2.grid(True, alpha=0.6)
    
    corr2 = significant_changes['roll_change'].corr(significant_changes['right_change'])
    ax2.text(0.05, 0.95, f'Correlation: {corr2:.3f}', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Plot 3: Time difference vs Roll change magnitude
    ax3.scatter(significant_changes['time_diff'] * 1000, abs(significant_changes['roll_change']), 
               alpha=0.6, s=20, color='green')
    ax3.set_xlabel('Time Difference [ms]')
    ax3.set_ylabel('|Roll Change| [%]')
    ax3.set_title('Time Interval vs Roll Change Magnitude')
    ax3.grid(True, alpha=0.6)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    
    # Plot 4: Position changes over time
    time_window = significant_changes['time_seconds'].iloc[:1000] if len(significant_changes) > 1000 else significant_changes['time_seconds']
    roll_window = significant_changes['roll_change'].iloc[:1000] if len(significant_changes) > 1000 else significant_changes['roll_change']
    left_window = significant_changes['left_change'].iloc[:1000] if len(significant_changes) > 1000 else significant_changes['left_change']
    right_window = significant_changes['right_change'].iloc[:1000] if len(significant_changes) > 1000 else significant_changes['right_change']
    
    ax4.plot(time_window, roll_window, 'g-', alpha=0.7, label='Roll Change', linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(time_window, left_window, 'b-', alpha=0.6, label='Left Change', linewidth=1)
    ax4_twin.plot(time_window, right_window, 'r-', alpha=0.6, label='Right Change', linewidth=1)
    
    ax4.set_xlabel('Time [seconds]')
    ax4.set_ylabel('Roll Change [%]', color='green')
    ax4_twin.set_ylabel('Position Change [mm]', color='black')
    ax4.set_title('Changes Over Time (First 1000 significant changes)')
    ax4.grid(True, alpha=0.6)
    
    # Legends
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.suptitle('Roll vs Position Change Correlation Analysis', fontsize=16, y=0.98)
    
    correlation_output = '/Users/baharakqaderi/field-data-pipeline/vaie_change_correlation_analysis.png'
    plt.savefig(correlation_output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Created: vaie_change_correlation_analysis.png")
    print(f"Roll-Left correlation: {corr1:.3f}")
    print(f"Roll-Right correlation: {corr2:.3f}")
    
    return correlation_output, significant_changes

def create_step_detection_analysis(df, significant_changes):
    """Try to detect actual movement steps"""
    
    print(f"\n🎯 DETECTING ACTUAL MOVEMENT STEPS")
    print("=" * 35)
    
    # Try different approaches to detect real movement steps
    
    # Approach 1: Group consecutive changes within time windows
    time_window = 0.1  # 100ms window
    step_groups = []
    current_group = []
    current_time = None
    
    for idx, row in significant_changes.iterrows():
        if current_time is None or (row['time_seconds'] - current_time) > time_window:
            # Start new group
            if current_group:
                step_groups.append(current_group)
            current_group = [row]
            current_time = row['time_seconds']
        else:
            # Add to current group
            current_group.append(row)
            current_time = row['time_seconds']
    
    if current_group:
        step_groups.append(current_group)
    
    print(f"Detected {len(step_groups)} potential movement groups using {time_window*1000}ms window")
    
    # Analyze the groups
    group_analysis = []
    for i, group in enumerate(step_groups[:10]):  # Analyze first 10 groups
        group_df = pd.DataFrame(group)
        
        total_roll_change = group_df['roll_change'].sum()
        total_left_change = group_df['left_change'].sum()
        total_right_change = group_df['right_change'].sum()
        group_duration = group_df['time_seconds'].max() - group_df['time_seconds'].min()
        
        group_info = {
            'group_id': i,
            'start_time': group_df['time_seconds'].min(),
            'duration': group_duration,
            'points': len(group),
            'total_roll_change': total_roll_change,
            'total_left_change': total_left_change,
            'total_right_change': total_right_change
        }
        
        group_analysis.append(group_info)
        
        print(f"Group {i+1}: {len(group)} points, {group_duration*1000:.1f}ms, "
              f"Roll: {total_roll_change:+.2f}%, Left: {total_left_change:+.1f}mm, Right: {total_right_change:+.1f}mm")
    
    # Create visualization of detected steps
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15))
    
    # Plot 1: Original data with detected groups highlighted
    sample_data = df.iloc[::10]  # Sample every 10th point for clarity
    
    ax1.plot(sample_data['time_seconds'], sample_data['FLIGHT_SEGMENT_roll_percentage'], 
            'g-', alpha=0.5, label='Roll %')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(sample_data['time_seconds'], sample_data['left_position_mm'], 
                  'b-', alpha=0.5, label='Left Position')
    
    # Highlight detected step groups
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(step_groups))))
    for i, (group, color) in enumerate(zip(step_groups[:10], colors)):
        group_df = pd.DataFrame(group)
        ax1.scatter(group_df['time_seconds'], group_df['FLIGHT_SEGMENT_roll_percentage'], 
                   c=[color], s=50, alpha=0.8, label=f'Group {i+1}')
    
    ax1.set_xlabel('Time [seconds]')
    ax1.set_ylabel('Roll Percentage [%]', color='green')
    ax1_twin.set_ylabel('Position [mm]', color='blue')
    ax1.set_title('Detected Movement Step Groups')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Group analysis
    group_df = pd.DataFrame(group_analysis)
    if not group_df.empty:
        x_pos = range(len(group_df))
        ax2.bar([x - 0.2 for x in x_pos], abs(group_df['total_roll_change']), 0.4, 
               label='|Roll Change|', alpha=0.8, color='green')
        ax2.bar([x + 0.2 for x in x_pos], abs(group_df['total_left_change']), 0.4, 
               label='|Left Change|', alpha=0.8, color='blue')
        
        ax2.set_xlabel('Group Number')
        ax2.set_ylabel('Total Change Magnitude')
        ax2.set_title('Movement Group Magnitudes')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'G{i+1}' for i in x_pos])
        ax2.legend()
        ax2.grid(True, alpha=0.6)
    
    # Plot 3: Group durations and point counts
    if not group_df.empty:
        ax3.scatter(group_df['duration'] * 1000, group_df['points'], 
                   c=abs(group_df['total_roll_change']), s=100, alpha=0.7, cmap='viridis')
        ax3.set_xlabel('Group Duration [ms]')
        ax3.set_ylabel('Points in Group')
        ax3.set_title('Movement Group Characteristics')
        ax3.grid(True, alpha=0.6)
        
        # Add colorbar
        cbar = plt.colorbar(ax3.scatter(group_df['duration'] * 1000, group_df['points'], 
                                       c=abs(group_df['total_roll_change']), s=100, alpha=0.7, cmap='viridis'))
        cbar.set_label('|Total Roll Change| [%]')
    
    plt.tight_layout()
    plt.suptitle('Movement Step Detection Analysis', fontsize=16, y=0.98)
    
    step_detection_output = '/Users/baharakqaderi/field-data-pipeline/vaie_step_detection_analysis.png'
    plt.savefig(step_detection_output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Created: vaie_step_detection_analysis.png")
    
    return step_detection_output, step_groups

def main():
    """Main analysis function"""
    print("🚁 SLICE ANALYSIS - UNDERSTANDING ROLL VS POSITION")
    print("=" * 55)
    
    # Load data
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} data points, duration: {df['time_seconds'].max():.1f}s")
    
    # Create slice visualizations
    slice_viz = create_slice_visualizations(df, slice_duration=20, num_slices=8)
    
    # Analyze change correlations
    corr_viz, significant_changes = create_change_correlation_analysis(df)
    
    # Try to detect actual movement steps
    step_viz, step_groups = create_step_detection_analysis(df, significant_changes)
    
    print(f"\n✨ SLICE ANALYSIS COMPLETE")
    print("=" * 30)
    print(f"🔍 Slice visualization: {slice_viz}")
    print(f"📊 Correlation analysis: {corr_viz}")
    print(f"🎯 Step detection: {step_viz}")
    print(f"\n💡 These visualizations should reveal the true relationship")
    print(f"📈 between roll commands and position changes")

if __name__ == "__main__":
    main()
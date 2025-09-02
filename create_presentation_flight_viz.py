#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import seaborn as sns

# Set style for high-quality presentation plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def calculate_flight_duration(start_time, end_time):
    """Calculate flight duration in minutes and seconds"""
    start_dt = datetime.strptime(start_time, "%H:%M:%S")
    end_dt = datetime.strptime(end_time, "%H:%M:%S")
    duration = end_dt - start_dt
    
    total_seconds = duration.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    return f"{minutes}m {seconds}s"

def create_presentation_visualization():
    """Create high-quality presentation visualization for flight 2025_07_17_11_30"""
    
    # Flight details for 2025_07_17_11_30 (from the flights list, this corresponds to flight 4: 11:30:35 - 11:50:12)
    target_flight = "2025_07_17_11_30"
    start_time = "11:30:35"
    end_time = "11:50:12"
    
    # Calculate flight duration
    flight_duration = calculate_flight_duration(start_time, end_time)
    
    # Load flight data
    base_dir = '/Users/baharakqaderi/field-data-pipeline/flight_analysis_july17'
    data_file = os.path.join(base_dir, target_flight, 'flight_data.csv')
    ground_data_file = os.path.join(base_dir, target_flight, 'ground_segment_data.csv')
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return
    
    # Load flight data
    flight_data = pd.read_csv(data_file)
    flight_data['_time'] = pd.to_datetime(flight_data['_time'], format='mixed')
    
    # Load ground segment data for brake status
    brake_engaged_percentage = 0.0
    if os.path.exists(ground_data_file):
        ground_data = pd.read_csv(ground_data_file)
        if 'GROUND_SEGMENT_brake_command' in ground_data.columns:
            brake_engaged_percentage = (ground_data['GROUND_SEGMENT_brake_command'] == 1).mean() * 100
    
    # Calculate total force
    flight_data['total_force'] = (
        flight_data['Backline_Left_kg'] + 
        flight_data['Backline_Right_kg'] + 
        flight_data['5th_line_kg'] + 
        flight_data['Frontline_kg']
    )
    
    # Prepare torque data
    torque_left = flight_data['FLIGHT_SEGMENT_l_torque'].fillna(0)
    torque_right = flight_data['FLIGHT_SEGMENT_r_torque'].fillna(0)
    
    # Combine all torque values for distribution (excluding zeros)
    all_torque = pd.concat([torque_left, torque_right])
    torque_clean = all_torque[all_torque != 0]
    
    # Create the presentation figure with high resolution
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    enhanced_title = (f'Date: July 17, 2025 | Duration: {flight_duration} '
                     f'| POD Brake Status: {brake_engaged_percentage:.1f}% engaged\n')

    fig.suptitle(enhanced_title, fontsize=20, fontweight='bold', y=0.92)
    
    # 1. Total Force Distribution
    ax1 = axes[0]
    n, bins, patches = ax1.hist(flight_data['total_force'], bins=40, alpha=0.8, color='#2E86AB', 
                               edgecolor='black', linewidth=1.2, density=True, 
                               weights=np.ones(len(flight_data['total_force'])) / len(flight_data['total_force']) * 100)
    
    # Add statistics lines for force
    mean_force = flight_data['total_force'].mean()
    max_force = flight_data['total_force'].max()
    
    ax1.axvline(mean_force, color='#F18F01', linestyle='--', linewidth=3, 
               label=f'Mean: {mean_force:.1f} kgf')
    ax1.axvline(max_force, color='#C73E1D', linestyle='-', linewidth=3, 
               label=f'Max: {max_force:.1f} kgf')
    
    ax1.set_title('Total Force Distribution', fontsize=18, fontweight='bold', pad=30)
    ax1.set_xlabel('Force (kgf)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Probability (%)', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=14, loc='upper right')
    ax1.grid(True, alpha=0.4, linewidth=0.8)
    
    # Improve force plot aesthetics
    ax1.set_facecolor('#f8f9fa')
    for patch in patches:
        patch.set_alpha(0.8)
    
    # 2. Torque Distribution (Left and Right separately like original)
    ax2 = axes[1]
    torque_left_clean = torque_left[torque_left != 0]
    torque_right_clean = torque_right[torque_right != 0]
    
    if len(torque_left_clean) > 0:
        ax2.hist(torque_left_clean, bins=25, alpha=0.6, label='Left Torque', color='#FF6B35', 
                edgecolor='black', linewidth=0.8, density=True, 
                weights=np.ones(len(torque_left_clean)) / len(torque_left_clean) * 100)
    
    if len(torque_right_clean) > 0:
        ax2.hist(torque_right_clean, bins=25, alpha=0.6, label='Right Torque', color='#004E89', 
                edgecolor='black', linewidth=0.8, density=True, 
                weights=np.ones(len(torque_right_clean)) / len(torque_right_clean) * 100)
    
    if len(torque_left_clean) == 0 and len(torque_right_clean) == 0:
        ax2.text(0.5, 0.5, 'No Torque Data Available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=16)
    
    ax2.set_title('Torque Distribution', fontsize=18, fontweight='bold', pad=30)
    ax2.set_xlabel('Torque (Nm)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Probability (%)', fontsize=16, fontweight='bold')
    if len(torque_left_clean) > 0 or len(torque_right_clean) > 0:
        ax2.legend(fontsize=14, loc='upper right')
    ax2.grid(True, alpha=0.4, linewidth=0.8)
    ax2.set_facecolor('#f8f9fa')
    
    # Adjust layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)  # Make room for the enhanced title and more space between title and plots
    
    # Save with high resolution for presentation
    output_file = os.path.join(base_dir, target_flight, 'presentation_force_torque_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.3)
    
    print(f"Created high-quality presentation visualization:")
    print(f"  File: {output_file}")
    print(f"  Flight: {target_flight}")
    print(f"  Duration: {flight_duration}")
    print(f"  POD Brake Status: {brake_engaged_percentage:.1f}% engaged")
    print(f"  Total Force Range: {flight_data['total_force'].min():.1f} - {flight_data['total_force'].max():.1f} kgf")
    if len(torque_clean) > 0:
        print(f"  Torque Range: {torque_clean.min():.2f} - {torque_clean.max():.2f} Nm")
    print(f"  Data Points: {len(flight_data)}")
    
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    create_presentation_visualization()
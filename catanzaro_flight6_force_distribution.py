#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Enhanced plotting parameters for presentation quality
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def create_force_distribution_presentation():
    """Create presentation-quality total force distribution for Catanzaro Flight 6"""
    
    # Load the OPC data
    data_file = '/Users/baharakqaderi/field-data-pipeline/flight_analysis_catanzaro/2025_07_29_09_11_Flight_6/opc_data_enhanced.csv'
    
    print("Loading OPC data...")
    opc_data = pd.read_csv(data_file)
    
    # Check if the force column exists
    if 'OPC_DsLoadCells.MeasureFloat_SUM' not in opc_data.columns:
        print("Error: OPC_DsLoadCells.MeasureFloat_SUM column not found!")
        return
    
    # Get the total force data (already calculated, no conversion needed)
    total_force_kgf = opc_data['OPC_DsLoadCells.MeasureFloat_SUM'].dropna()
    
    print(f"Force range: {total_force_kgf.min():.2f} to {total_force_kgf.max():.2f} kgf")
    print(f"Mean force: {total_force_kgf.mean():.2f} kgf")
    
    # Create the distribution plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create histogram
    n_bins = 50
    counts, bins, patches = ax.hist(total_force_kgf, bins=n_bins, alpha=0.7, 
                                   color='#2E86AB', edgecolor='black', linewidth=0.5,
                                   density=True)
    
    # Calculate statistics
    mean_force = total_force_kgf.mean()
    median_force = total_force_kgf.median()
    max_force = total_force_kgf.max()
    std_force = total_force_kgf.std()
    
    # Add statistical lines
    ax.axvline(mean_force, color='#F18F01', linestyle='--', linewidth=3, 
              label=f'Mean: {mean_force:.1f} kgf')
    ax.axvline(median_force, color='#A23B72', linestyle='-.', linewidth=3, 
              label=f'Median: {median_force:.1f} kgf')
    ax.axvline(max_force, color='#C73E1D', linestyle='-', linewidth=3, 
              label=f'Maximum: {max_force:.1f} kgf')
    
    # Enhance the plot
    ax.set_xlabel('Total Force (kgf)', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.set_title('Catanzaro Flight 6 - Total Force Distribution | '
                'Date: July 29, 2025 | Time: 09:11:09 - 09:18:14 | Duration: 7m 5s', 
                fontweight='bold', pad=20)
    
    # Statistics text box removed to avoid overlap with legend
    
    # Style the plot
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.4, linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '/Users/baharakqaderi/field-data-pipeline/flight_analysis_catanzaro/2025_07_29_09_11_Flight_6/Flight_6_total_force_distribution_OPC_presentation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved presentation plot: {output_file}")
    
    plt.close()
    print("Force distribution presentation plot created successfully!")

if __name__ == "__main__":
    create_force_distribution_presentation()
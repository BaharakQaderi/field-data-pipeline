#!/usr/bin/env python3
"""
Energy Generation Time Series Analysis for Catanzaro Flight 7 Only

Shows moments when PoBatt > 0 indicating energy generation from regenerative braking.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta

# Set style for presentation
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

def create_flight7_energy_plot():
    """Create presentation-quality time series of Flight 7 energy generation"""
    
    # Load Flight 7 data
    print("Loading Flight 7 data...")
    df7 = pd.read_csv('/Users/baharakqaderi/field-data-pipeline/flight_analysis_catanzaro/2025_07_29_09_45_Flight_7/opc_data_enhanced.csv')
    df7['_time'] = pd.to_datetime(df7['_time'])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Plot all PoBatt data
    ax.plot(df7['_time'], df7['PoBatt'], alpha=0.6, linewidth=1, color='#2E86AB', label='PoBatt (All Data)')
    
    # Highlight positive values (energy generation)
    positive_mask7 = df7['PoBatt'] > 0
    positive_data7 = df7[positive_mask7]
    
    if len(positive_data7) > 0:
        ax.scatter(positive_data7['_time'], positive_data7['PoBatt'], 
                   color='#F18F01', s=30, alpha=0.8, zorder=5,
                   label=f'Energy Generation ({len(positive_data7)} points)')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Zero Line')
    
    # Statistics for Flight 7
    duration_total7 = (df7['_time'].max() - df7['_time'].min()).total_seconds()
    if len(positive_data7) > 0:
        duration_positive7 = (positive_data7['_time'].max() - positive_data7['_time'].min()).total_seconds()
        max_power7 = positive_data7['PoBatt'].max()
        avg_power7 = positive_data7['PoBatt'].mean()
    else:
        duration_positive7 = 0
        max_power7 = 0
        avg_power7 = 0
    
    ax.set_title(f'Catanzaro Flight 7 - Energy Generation Moments | '
                 f'Date: July 29, 2025 | Duration: {duration_total7/60:.1f} min | '
                 f'Generation Events: {len(positive_data7)/len(df7)*100:.1f}% of time', 
                 fontweight='bold', pad=20)
    ax.set_ylabel('Battery Power (PoBatt) [kW]', fontweight='bold')
    ax.set_xlabel('Time', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.4)
    
    # Add statistics text box for Flight 7
    stats_text7 = f'Max Generation: {max_power7:.2f} kW\nAvg Generation: {avg_power7:.2f} kW\nGeneration Duration: {duration_positive7:.0f}s'
    ax.text(0.02, 0.98, stats_text7, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '/Users/baharakqaderi/field-data-pipeline/catanzaro_flight7_energy_generation_presentation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved Flight 7 energy generation plot: {output_file}")
    
    plt.close()
    print("Flight 7 energy generation plot created successfully!")

if __name__ == "__main__":
    create_flight7_energy_plot()
#!/usr/bin/env python3
"""
Energy Generation Time Series Analysis for Catanzaro Flights 6 & 7

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

def analyze_energy_generation():
    """Create presentation-quality time series of energy generation moments"""
    
    # Load Flight 6 data
    print("Loading Flight 6 data...")
    df6 = pd.read_csv('/Users/baharakqaderi/field-data-pipeline/flight_analysis_catanzaro/2025_07_29_09_11_Flight_6/opc_data_enhanced.csv')
    df6['_time'] = pd.to_datetime(df6['_time'])
    
    # Load Flight 7 data
    print("Loading Flight 7 data...")
    df7 = pd.read_csv('/Users/baharakqaderi/field-data-pipeline/flight_analysis_catanzaro/2025_07_29_09_45_Flight_7/opc_data_enhanced.csv')
    df7['_time'] = pd.to_datetime(df7['_time'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Flight 6 Analysis
    ax1 = axes[0]
    
    # Plot all PoBatt data
    ax1.plot(df6['_time'], df6['PoBatt'], alpha=0.6, linewidth=1, color='#2E86AB', label='PoBatt (All Data)')
    
    # Highlight positive values (energy generation)
    positive_mask6 = df6['PoBatt'] > 0
    positive_data6 = df6[positive_mask6]
    
    if len(positive_data6) > 0:
        ax1.scatter(positive_data6['_time'], positive_data6['PoBatt'], 
                   color='#F18F01', s=30, alpha=0.8, zorder=5,
                   label=f'Energy Generation ({len(positive_data6)} points)')
    
    # Add horizontal line at zero
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Zero Line')
    
    # Statistics for Flight 6
    duration_total6 = (df6['_time'].max() - df6['_time'].min()).total_seconds()
    if len(positive_data6) > 0:
        duration_positive6 = (positive_data6['_time'].max() - positive_data6['_time'].min()).total_seconds()
        max_power6 = positive_data6['PoBatt'].max()
        avg_power6 = positive_data6['PoBatt'].mean()
    else:
        duration_positive6 = 0
        max_power6 = 0
        avg_power6 = 0
    
    ax1.set_title(f'Catanzaro Flight 6 - Energy Generation Moments\\n'
                 f'Date: July 29, 2025 | Duration: {duration_total6/60:.1f} min | '
                 f'Generation Events: {len(positive_data6)/len(df6)*100:.1f}% of time', 
                 fontweight='bold', pad=20)
    ax1.set_ylabel('Battery Power (PoBatt) [kW]', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.4)
    
    # Add statistics text box for Flight 6
    stats_text6 = f'Max Generation: {max_power6:.2f} kW\\nAvg Generation: {avg_power6:.2f} kW\\nGeneration Duration: {duration_positive6:.0f}s'
    ax1.text(0.02, 0.98, stats_text6, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Flight 7 Analysis
    ax2 = axes[1]
    
    # Plot all PoBatt data
    ax2.plot(df7['_time'], df7['PoBatt'], alpha=0.6, linewidth=1, color='#2E86AB', label='PoBatt (All Data)')
    
    # Highlight positive values (energy generation)
    positive_mask7 = df7['PoBatt'] > 0
    positive_data7 = df7[positive_mask7]
    
    if len(positive_data7) > 0:
        ax2.scatter(positive_data7['_time'], positive_data7['PoBatt'], 
                   color='#F18F01', s=30, alpha=0.8, zorder=5,
                   label=f'Energy Generation ({len(positive_data7)} points)')
    
    # Add horizontal line at zero
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Zero Line')
    
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
    
    ax2.set_title(f'Catanzaro Flight 7 - Energy Generation Moments\\n'
                 f'Date: July 29, 2025 | Duration: {duration_total7/60:.1f} min | '
                 f'Generation Events: {len(positive_data7)/len(df7)*100:.1f}% of time', 
                 fontweight='bold', pad=20)
    ax2.set_ylabel('Battery Power (PoBatt) [kW]', fontweight='bold')
    ax2.set_xlabel('Time', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.4)
    
    # Add statistics text box for Flight 7
    stats_text7 = f'Max Generation: {max_power7:.2f} kW\\nAvg Generation: {avg_power7:.2f} kW\\nGeneration Duration: {duration_positive7:.0f}s'
    ax2.text(0.02, 0.98, stats_text7, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Style both plots
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # Save the plots
    output_file = '/Users/baharakqaderi/field-data-pipeline/catanzaro_flights_energy_generation_timeseries.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved energy generation time series: {output_file}")
    
    plt.close()
    
    # Create summary analysis
    print("\\n" + "="*60)
    print("ENERGY GENERATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"\\nFlight 6:")
    print(f"  Total duration: {duration_total6/60:.1f} minutes")
    print(f"  Energy generation events: {len(positive_data6)} points ({len(positive_data6)/len(df6)*100:.1f}% of flight)")
    print(f"  Generation duration: {duration_positive6:.1f} seconds ({duration_positive6/60:.1f} minutes)")
    print(f"  Max generation power: {max_power6:.3f} kW")
    print(f"  Average generation power: {avg_power6:.3f} kW")
    
    print(f"\\nFlight 7:")
    print(f"  Total duration: {duration_total7/60:.1f} minutes")
    print(f"  Energy generation events: {len(positive_data7)} points ({len(positive_data7)/len(df7)*100:.1f}% of flight)")
    print(f"  Generation duration: {duration_positive7:.1f} seconds ({duration_positive7/60:.1f} minutes)")
    print(f"  Max generation power: {max_power7:.3f} kW")
    print(f"  Average generation power: {avg_power7:.3f} kW")
    
    # Find the most significant generation periods (>= 30 seconds continuous)
    print(f"\\n" + "="*40)
    print("SIGNIFICANT GENERATION PERIODS")
    print("="*40)
    
    for flight_num, (df, positive_data) in enumerate([(df6, positive_data6), (df7, positive_data7)], 6):
        print(f"\\nFlight {flight_num}:")
        if len(positive_data) > 0:
            # Find continuous periods
            time_diffs = positive_data['_time'].diff()
            gaps = time_diffs > pd.Timedelta(seconds=10)  # 10 second gap threshold
            periods = []
            
            if len(positive_data) > 0:
                start_idx = 0
                for i, is_gap in enumerate(gaps):
                    if is_gap and i > start_idx:
                        period_data = positive_data.iloc[start_idx:i]
                        duration = (period_data['_time'].max() - period_data['_time'].min()).total_seconds()
                        if duration >= 30:  # Only show periods >= 30 seconds
                            periods.append({
                                'start': period_data['_time'].min(),
                                'end': period_data['_time'].max(),
                                'duration': duration,
                                'avg_power': period_data['PoBatt'].mean(),
                                'max_power': period_data['PoBatt'].max()
                            })
                        start_idx = i
                
                # Add the final period
                if start_idx < len(positive_data):
                    period_data = positive_data.iloc[start_idx:]
                    duration = (period_data['_time'].max() - period_data['_time'].min()).total_seconds()
                    if duration >= 30:
                        periods.append({
                            'start': period_data['_time'].min(),
                            'end': period_data['_time'].max(),
                            'duration': duration,
                            'avg_power': period_data['PoBatt'].mean(),
                            'max_power': period_data['PoBatt'].max()
                        })
            
            if periods:
                for i, period in enumerate(periods, 1):
                    print(f"  Period {i}: {period['start'].strftime('%H:%M:%S')} - {period['end'].strftime('%H:%M:%S')}")
                    print(f"    Duration: {period['duration']:.0f}s, Avg: {period['avg_power']:.3f}kW, Max: {period['max_power']:.3f}kW")
            else:
                print("  No significant periods (>=30s) found")
        else:
            print("  No energy generation detected")

    print(f"\\nEnergy generation time series plot created successfully!")

if __name__ == "__main__":
    analyze_energy_generation()
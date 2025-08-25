#!/usr/bin/env python3
"""
Load Cell Data Visualizer

Quick visualization script to plot load cell readings from multiple CSV files
to identify important data and non-working sensors.

Usage:
    python loadcell_visualizer.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

def analyze_loadcell_file(file_path):
    """
    Analyze a single load cell CSV file
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert datetime column to pandas datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Calculate basic statistics
        stats = {
            'file': os.path.basename(file_path),
            'duration_minutes': (df['datetime'].max() - df['datetime'].min()).total_seconds() / 60,
            'num_readings': len(df),
            'loadcell_1_range': df['Loadcell_1_kg'].max() - df['Loadcell_1_kg'].min(),
            'loadcell_2_range': df['Loadcell_2_kg'].max() - df['Loadcell_2_kg'].min(),
            'loadcell_1_std': df['Loadcell_1_kg'].std(),
            'loadcell_2_std': df['Loadcell_2_kg'].std(),
            'loadcell_1_mean': df['Loadcell_1_kg'].mean(),
            'loadcell_2_mean': df['Loadcell_2_kg'].mean(),
            'total_force_max': df['total_force_kg'].max(),
            'total_force_min': df['total_force_kg'].min(),
        }
        
        return df, stats
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def create_loadcell_plot(df, stats, output_path):
    """
    Create a plot for a single load cell file
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Load Cell Analysis: {stats['file']}", fontsize=14, fontweight='bold')
    
    # Plot 1: Individual load cells over time
    axes[0, 0].plot(df['datetime'], df['Loadcell_1_kg'], label='Load Cell 1', alpha=0.7, linewidth=0.8)
    axes[0, 0].plot(df['datetime'], df['Loadcell_2_kg'], label='Load Cell 2', alpha=0.7, linewidth=0.8)
    axes[0, 0].set_title('Individual Load Cell Readings')
    axes[0, 0].set_ylabel('Force (kg)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Total force over time
    axes[0, 1].plot(df['datetime'], df['total_force_kg'], color='red', alpha=0.8, linewidth=1)
    axes[0, 1].set_title('Total Force')
    axes[0, 1].set_ylabel('Total Force (kg)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Load cell comparison scatter
    axes[1, 0].scatter(df['Loadcell_1_kg'], df['Loadcell_2_kg'], alpha=0.5, s=1)
    axes[1, 0].set_xlabel('Load Cell 1 (kg)')
    axes[1, 0].set_ylabel('Load Cell 2 (kg)')
    axes[1, 0].set_title('Load Cell 1 vs Load Cell 2')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics text
    axes[1, 1].axis('off')
    stats_text = f"""
    Duration: {stats['duration_minutes']:.1f} minutes
    Readings: {stats['num_readings']:,}
    
    Load Cell 1:
      Range: {stats['loadcell_1_range']:.4f} kg
      Mean: {stats['loadcell_1_mean']:.4f} kg
      Std Dev: {stats['loadcell_1_std']:.4f} kg
    
    Load Cell 2:
      Range: {stats['loadcell_2_range']:.4f} kg
      Mean: {stats['loadcell_2_mean']:.4f} kg
      Std Dev: {stats['loadcell_2_std']:.4f} kg
    
    Total Force:
      Max: {stats['total_force_max']:.4f} kg
      Min: {stats['total_force_min']:.4f} kg
    """
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to process all load cell files
    """
    data_dir = "../data/log_raspi2"
    output_dir = "../outputs/loadcell_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    
    print(f"Found {len(csv_files)} load cell CSV files")
    print("Processing files...")
    
    all_stats = []
    
    for i, filename in enumerate(csv_files):
        print(f"Processing {i+1}/{len(csv_files)}: {filename}")
        
        file_path = os.path.join(data_dir, filename)
        df, stats = analyze_loadcell_file(file_path)
        
        if df is not None and stats is not None:
            # Create individual plot
            plot_filename = filename.replace('.csv', '_plot.png')
            plot_path = os.path.join(output_dir, plot_filename)
            create_loadcell_plot(df, stats, plot_path)
            
            all_stats.append(stats)
        else:
            print(f"  Skipped {filename} due to errors")
    
    # Create summary analysis
    if all_stats:
        create_summary_analysis(all_stats, output_dir)
    
    print(f"\nProcessing complete!")
    print(f"Individual plots saved to: {output_dir}")
    print(f"Summary analysis saved to: {output_dir}/summary_analysis.txt")

def create_summary_analysis(all_stats, output_dir):
    """
    Create a summary analysis of all files
    """
    summary_path = os.path.join(output_dir, "summary_analysis.txt")
    
    with open(summary_path, 'w') as f:
        f.write("LOAD CELL DATA SUMMARY ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        # Sort files by different criteria for analysis
        f.write("FILES BY DURATION (longest first):\n")
        f.write("-" * 30 + "\n")
        sorted_by_duration = sorted(all_stats, key=lambda x: x['duration_minutes'], reverse=True)
        for stats in sorted_by_duration[:10]:  # Top 10
            f.write(f"{stats['file']:<40} {stats['duration_minutes']:>8.1f} min\n")
        
        f.write(f"\nFILES BY ACTIVITY (highest range first):\n")
        f.write("-" * 30 + "\n")
        for stats in all_stats:
            stats['max_range'] = max(stats['loadcell_1_range'], stats['loadcell_2_range'])
        
        sorted_by_activity = sorted(all_stats, key=lambda x: x['max_range'], reverse=True)
        for stats in sorted_by_activity[:10]:  # Top 10
            f.write(f"{stats['file']:<40} {stats['max_range']:>8.4f} kg range\n")
        
        f.write(f"\nPOTENTIAL SENSOR ISSUES:\n")
        f.write("-" * 30 + "\n")
        
        for stats in all_stats:
            issues = []
            
            # Check for very low activity (might indicate broken sensor)
            if stats['loadcell_1_range'] < 0.001:
                issues.append("LC1: Very low range")
            if stats['loadcell_2_range'] < 0.001:
                issues.append("LC2: Very low range")
            
            # Check for very high standard deviation (might indicate noise)
            if stats['loadcell_1_std'] > 0.01:
                issues.append("LC1: High noise")
            if stats['loadcell_2_std'] > 0.01:
                issues.append("LC2: High noise")
            
            # Check for extreme bias
            if abs(stats['loadcell_1_mean']) > 0.05:
                issues.append("LC1: High bias")
            if abs(stats['loadcell_2_mean']) > 0.05:
                issues.append("LC2: High bias")
            
            if issues:
                f.write(f"{stats['file']:<40} {', '.join(issues)}\n")
        
        f.write(f"\nOVERALL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total files processed: {len(all_stats)}\n")
        f.write(f"Total duration: {sum(s['duration_minutes'] for s in all_stats):.1f} minutes\n")
        f.write(f"Total readings: {sum(s['num_readings'] for s in all_stats):,}\n")
        
        # Find the most active periods
        f.write(f"\nMOST ACTIVE FILES (likely important):\n")
        f.write("-" * 30 + "\n")
        for stats in sorted_by_activity[:5]:
            f.write(f"{stats['file']}\n")
            f.write(f"  Duration: {stats['duration_minutes']:.1f} min, Max range: {stats['max_range']:.4f} kg\n")
            f.write(f"  Max total force: {stats['total_force_max']:.4f} kg\n\n")

if __name__ == "__main__":
    main()
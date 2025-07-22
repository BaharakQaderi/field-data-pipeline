"""
Force Validation Plots

This module creates time series visualizations to validate the force data alignment
between InfluxDB data and FORCES data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

# Setup plotting style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class ForceValidator:
    """
    Creates validation plots for force data alignment.
    """
    
    def __init__(self, data_dir: Path = Path("data"), output_dir: Path = Path("outputs")):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
    def load_original_influx(self, date: str) -> pd.DataFrame:
        """Load original InfluxDB data."""
        filename = f"FLIGHT_SEGMENT_data_{date}.csv"
        filepath = self.data_dir / "INFLUX" / filename
        
        print(f"Loading original InfluxDB data from {filename}")
        df = pd.read_csv(filepath)
        df['_time'] = pd.to_datetime(df['_time'], format='mixed')
        df = df.dropna(subset=['FLIGHT_SEGMENT_loadcells_force'])
        
        print(f"Loaded {len(df)} InfluxDB records with force data")
        return df
        
    def load_original_forces(self, date: str) -> pd.DataFrame:
        """Load original FORCES data for the date."""
        # Get the appropriate files based on date
        forces_files = sorted(list((self.data_dir / "FORCES").glob("loadcell_readings_*.csv")))
        
        if date == "2025-07-16":
            selected_files = forces_files[:3]  # First 3 files
        elif date == "2025-07-17":
            selected_files = forces_files[3:5]  # Last 2 files
        else:
            raise ValueError(f"Unsupported date: {date}")
            
        print(f"Loading original FORCES data from {len(selected_files)} files")
        
        all_forces = []
        for file_path in selected_files:
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
            df = df[df['total_force_kg'] > 0.001]  # Filter meaningful data
            df['source_file'] = file_path.name
            all_forces.append(df)
            
        combined_df = pd.concat(all_forces, ignore_index=True)
        print(f"Loaded {len(combined_df)} FORCES records")
        return combined_df
        
    def load_merged_data(self, date: str) -> pd.DataFrame:
        """Load the merged data."""
        filename = f"merged_flight_data_{date}.csv"
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Merged data file not found: {filepath}")
            
        print(f"Loading merged data from {filename}")
        df = pd.read_csv(filepath)
        df['_time'] = pd.to_datetime(df['_time'], format='mixed')
        
        print(f"Loaded {len(df)} merged records")
        return df
        
    def create_validation_plots(self, date: str, time_window_hours: int = 2):
        """
        Create comprehensive validation plots comparing original and merged data.
        
        Args:
            date: Date to analyze (YYYY-MM-DD)
            time_window_hours: Hours of data to show in detailed plot
        """
        print(f"\n🎯 Creating validation plots for {date}")
        
        # Load all datasets
        influx_orig = self.load_original_influx(date)
        forces_orig = self.load_original_forces(date)
        merged_data = self.load_merged_data(date)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Overview plot - Full day comparison
        ax1 = plt.subplot(4, 2, (1, 2))
        
        # Plot original InfluxDB forces
        ax1.plot(influx_orig['_time'], influx_orig['FLIGHT_SEGMENT_loadcells_force'], 
                'b-', alpha=0.7, linewidth=1, label='InfluxDB Original Force')
        
        # Plot merged data (only where forces were matched)
        matched_data = merged_data[merged_data['forces_matched'] == True]
        ax1.plot(matched_data['_time'], matched_data['FLIGHT_SEGMENT_loadcells_force'], 
                'r.', markersize=2, alpha=0.8, label='InfluxDB (Matched Points)')
        
        ax1.set_title(f'Full Day Force Comparison - {date}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Force (kg)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Individual Force Components (Matched data only)
        ax2 = plt.subplot(4, 2, (3, 4))
        
        force_components = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
        colors = ['red', 'blue', 'green', 'orange']
        
        for component, color in zip(force_components, colors):
            component_data = matched_data.dropna(subset=[component])
            if len(component_data) > 0:
                ax2.plot(component_data['_time'], component_data[component], 
                        color=color, alpha=0.7, linewidth=1, label=component.replace('_kg', ''))
        
        # Also plot the total from components
        components_total = matched_data[force_components].sum(axis=1)
        ax2.plot(matched_data['_time'], components_total, 
                'k--', alpha=0.8, linewidth=2, label='Sum of Components')
        
        ax2.set_title('Individual Force Components Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Force (kg)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Force Matching Quality
        ax3 = plt.subplot(4, 2, 5)
        
        match_rate = matched_data['forces_matched'].sum() / len(merged_data) * 100
        force_diff_stats = matched_data['force_difference'].describe()
        
        # Plot force differences
        ax3.hist(matched_data['force_difference'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(force_diff_stats['mean'], color='red', linestyle='--', 
                   label=f'Mean: {force_diff_stats["mean"]:.4f} kg')
        ax3.axvline(force_diff_stats['50%'], color='orange', linestyle='--', 
                   label=f'Median: {force_diff_stats["50%"]:.4f} kg')
        
        ax3.set_title(f'Force Matching Quality (Match Rate: {match_rate:.1f}%)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Force Difference (kg)')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Correlation Analysis
        ax4 = plt.subplot(4, 2, 6)
        
        # Scatter plot: InfluxDB vs Components Sum
        influx_forces = matched_data['FLIGHT_SEGMENT_loadcells_force']
        components_sum = matched_data[force_components].sum(axis=1)
        
        ax4.scatter(influx_forces, components_sum, alpha=0.5, s=10, color='green')
        
        # Perfect correlation line
        min_val = min(influx_forces.min(), components_sum.min())
        max_val = max(influx_forces.max(), components_sum.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Correlation')
        
        # Calculate correlation
        correlation = np.corrcoef(influx_forces, components_sum)[0, 1]
        ax4.set_title(f'InfluxDB vs Components Sum (r = {correlation:.3f})', fontsize=12, fontweight='bold')
        ax4.set_xlabel('InfluxDB Force (kg)')
        ax4.set_ylabel('Sum of Components (kg)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Detailed Time Window Analysis
        ax5 = plt.subplot(4, 2, 7)
        
        # Find a time window with good data
        start_time = matched_data['_time'].iloc[0]
        end_time = start_time + pd.Timedelta(hours=time_window_hours)
        
        window_data = matched_data[
            (matched_data['_time'] >= start_time) & 
            (matched_data['_time'] <= end_time)
        ]
        
        if len(window_data) > 0:
            ax5.plot(window_data['_time'], window_data['FLIGHT_SEGMENT_loadcells_force'], 
                    'b-', linewidth=2, label='InfluxDB Total', marker='o', markersize=3)
            
            for component, color in zip(force_components[:2], colors[:2]):  # Show first 2 components
                component_data = window_data.dropna(subset=[component])
                if len(component_data) > 0:
                    ax5.plot(component_data['_time'], component_data[component], 
                            color=color, linewidth=1.5, label=component.replace('_kg', ''), 
                            marker='s', markersize=2)
        
        ax5.set_title(f'Detailed View - First {time_window_hours} Hours', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Force (kg)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        ax6 = plt.subplot(4, 2, 8)
        ax6.axis('off')  # Turn off axis for text display
        
        # Create summary text
        summary_text = f"""
        📊 VALIDATION SUMMARY for {date}
        
        📈 Data Coverage:
        • Total InfluxDB records: {len(influx_orig):,}
        • Successfully matched: {len(matched_data):,}
        • Match rate: {match_rate:.1f}%
        
        🎯 Force Alignment Quality:
        • Mean difference: {force_diff_stats['mean']:.4f} kg
        • Median difference: {force_diff_stats['50%']:.4f} kg
        • Max difference: {force_diff_stats['max']:.4f} kg
        • Std deviation: {force_diff_stats['std']:.4f} kg
        
        🔗 Correlation:
        • InfluxDB vs Components: {correlation:.3f}
        
        ⚡ Force Components Available:
        • Backline Left: {matched_data['Backline_Left_kg'].notna().sum():,} records
        • Backline Right: {matched_data['Backline_Right_kg'].notna().sum():,} records
        • 5th Line: {matched_data['5th_line_kg'].notna().sum():,} records
        • Front Line: {matched_data['Frontline_kg'].notna().sum():,} records
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"force_validation_{date}.png"
        plot_path = self.output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"✅ Validation plot saved to: {plot_path}")
        
        # Show the plot
        plt.show()
        
        # Print summary to console
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"  Match rate: {match_rate:.1f}%")
        print(f"  Correlation (InfluxDB vs Components): {correlation:.3f}")
        print(f"  Mean force difference: {force_diff_stats['mean']:.4f} kg")
        
        return {
            'match_rate': match_rate,
            'correlation': correlation,
            'force_diff_stats': force_diff_stats,
            'plot_path': plot_path
        }


def main():
    """Main function to create validation plots."""
    validator = ForceValidator()
    
    # Create validation plots for July 16
    results = validator.create_validation_plots("2025-07-16")
    
    print(f"\n🎯 Validation complete!")
    print(f"   Plot saved to: {results['plot_path']}")


if __name__ == "__main__":
    main() 
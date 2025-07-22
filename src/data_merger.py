"""
Data Merger for Field Data Pipeline

This module handles the alignment and merging of FORCES data with InfluxDB data
using total force values as matching keys to overcome timestamp synchronization issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from datetime import datetime
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataMerger:
    """
    Handles merging of FORCES data with InfluxDB data using force values as alignment keys.
    """
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = Path(data_dir)
        self.influx_dir = self.data_dir / "INFLUX"
        self.forces_dir = self.data_dir / "FORCES"
        
    def load_influx_data(self, date: str) -> pd.DataFrame:
        """
        Load InfluxDB data for a specific date.
        
        Args:
            date: Date string in format 'YYYY-MM-DD'
            
        Returns:
            DataFrame with InfluxDB data
        """
        filename = f"FLIGHT_SEGMENT_data_{date}.csv"
        filepath = self.influx_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"InfluxDB file not found: {filepath}")
            
        logger.info(f"Loading InfluxDB data from {filename}")
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime (handle mixed formats)
        df['_time'] = pd.to_datetime(df['_time'], format='mixed')
        
        # Filter out rows with missing force data
        df = df.dropna(subset=['FLIGHT_SEGMENT_loadcells_force'])
        
        logger.info(f"Loaded {len(df)} rows with force data")
        return df
        
    def load_forces_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load FORCES data from a CSV file.
        
        Args:
            filepath: Path to FORCES CSV file
            
        Returns:
            DataFrame with FORCES data
        """
        logger.info(f"Loading FORCES data from {filepath.name}")
        df = pd.read_csv(filepath)
        
        # Convert datetime column to pandas datetime (even though unreliable)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        
        # Filter out rows with zero total force (likely no meaningful data)
        df = df[df['total_force_kg'] > 0.001]  # Small threshold to avoid noise
        
        logger.info(f"Loaded {len(df)} rows with meaningful force data")
        return df
        
    def get_forces_files_for_date(self, date: str) -> List[Path]:
        """
        Get FORCES files corresponding to a specific date.
        
        Args:
            date: Date string in format 'YYYY-MM-DD'
            
        Returns:
            List of Path objects for FORCES files
        """
        # Based on your analysis:
        # July 16 (2025-07-16) -> first 3 files
        # July 17 (2025-07-17) -> last 2 files
        
        forces_files = sorted(list(self.forces_dir.glob("loadcell_readings_*.csv")))
        
        if date == "2025-07-16":
            return forces_files[:3]  # First 3 files
        elif date == "2025-07-17":
            return forces_files[3:5]  # Last 2 files (files 4 and 5, 0-indexed: 3 and 4)
        else:
            raise ValueError(f"Unsupported date: {date}. Only 2025-07-16 and 2025-07-17 are supported.")
            
    def align_forces_with_influx(self, influx_df: pd.DataFrame, forces_df: pd.DataFrame, 
                                tolerance: float = 0.1) -> pd.DataFrame:
        """
        Align FORCES data with InfluxDB data using total force values.
        
        Args:
            influx_df: InfluxDB DataFrame
            forces_df: FORCES DataFrame  
            tolerance: Tolerance for force matching (kg)
            
        Returns:
            DataFrame with merged data
        """
        logger.info(f"Aligning FORCES data with InfluxDB data (tolerance: {tolerance} kg)")
        
        # Prepare the result DataFrame
        result_df = influx_df.copy()
        
        # Initialize columns for individual force components
        force_columns = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
        for col in force_columns:
            result_df[col] = np.nan
            
        result_df['forces_matched'] = False
        result_df['force_difference'] = np.nan
        
        matched_count = 0
        
        # For each InfluxDB record, find the best matching FORCES record
        for idx, influx_row in influx_df.iterrows():
            influx_force = influx_row['FLIGHT_SEGMENT_loadcells_force']
            
            if pd.isna(influx_force):
                continue
                
            # Find FORCES records within tolerance
            force_diff = np.abs(forces_df['total_force_kg'] - influx_force)
            matches = force_diff <= tolerance
            
            if matches.any():
                # Get the closest match
                best_match_idx = force_diff.idxmin()
                best_match = forces_df.loc[best_match_idx]
                
                # Add force components to result
                for col in force_columns:
                    result_df.loc[idx, col] = best_match[col]
                    
                result_df.loc[idx, 'forces_matched'] = True
                result_df.loc[idx, 'force_difference'] = force_diff.loc[best_match_idx]
                
                matched_count += 1
                
        match_rate = matched_count / len(influx_df) * 100
        logger.info(f"Successfully matched {matched_count}/{len(influx_df)} records ({match_rate:.1f}%)")
        
        return result_df
        
    def merge_day_data(self, date: str, tolerance: float = 0.1) -> pd.DataFrame:
        """
        Merge all data for a specific day.
        
        Args:
            date: Date string in format 'YYYY-MM-DD'
            tolerance: Tolerance for force matching (kg)
            
        Returns:
            DataFrame with merged data for the day
        """
        logger.info(f"Starting data merge for {date}")
        
        # Load InfluxDB data
        influx_df = self.load_influx_data(date)
        
        # Get FORCES files for this date
        forces_files = self.get_forces_files_for_date(date)
        
        # Load and combine all FORCES data for the day
        all_forces_data = []
        for forces_file in forces_files:
            forces_df = self.load_forces_data(forces_file)
            forces_df['source_file'] = forces_file.name
            all_forces_data.append(forces_df)
            
        combined_forces_df = pd.concat(all_forces_data, ignore_index=True)
        logger.info(f"Combined {len(combined_forces_df)} FORCES records from {len(forces_files)} files")
        
        # Align and merge the data
        merged_df = self.align_forces_with_influx(influx_df, combined_forces_df, tolerance)
        
        return merged_df
        
    def generate_summary_stats(self, merged_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the merged data.
        
        Args:
            merged_df: Merged DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_records': len(merged_df),
            'matched_records': merged_df['forces_matched'].sum(),
            'match_rate': merged_df['forces_matched'].mean() * 100,
            'avg_force_difference': merged_df[merged_df['forces_matched']]['force_difference'].mean(),
            'max_force_difference': merged_df[merged_df['forces_matched']]['force_difference'].max(),
            'force_component_coverage': {
                col: merged_df[col].notna().sum() for col in 
                ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
            }
        }
        
        return stats
        
    def save_merged_data(self, merged_df: pd.DataFrame, date: str, output_dir: Path = None):
        """
        Save merged data to CSV file.
        
        Args:
            merged_df: Merged DataFrame
            date: Date string
            output_dir: Output directory (defaults to outputs/)
        """
        if output_dir is None:
            output_dir = Path("outputs")
            
        output_dir.mkdir(exist_ok=True)
        
        filename = f"merged_flight_data_{date}.csv"
        filepath = output_dir / filename
        
        merged_df.to_csv(filepath, index=False)
        logger.info(f"Saved merged data to {filepath}")
        
        return filepath


def main():
    """
    Main function to demonstrate the data merger functionality.
    """
    merger = DataMerger()
    
    dates = ["2025-07-16", "2025-07-17"]
    
    for date in dates:
        try:
            logger.info(f"=== Processing {date} ===")
            
            # Merge data for the day
            merged_df = merger.merge_day_data(date)
            
            # Generate and display summary statistics
            stats = merger.generate_summary_stats(merged_df)
            
            print(f"\n📊 Summary for {date}:")
            print(f"  Total records: {stats['total_records']}")
            print(f"  Matched records: {stats['matched_records']}")
            print(f"  Match rate: {stats['match_rate']:.1f}%")
            print(f"  Avg force difference: {stats['avg_force_difference']:.4f} kg")
            print(f"  Max force difference: {stats['max_force_difference']:.4f} kg")
            
            print(f"\n🔧 Force component coverage:")
            for component, count in stats['force_component_coverage'].items():
                print(f"  {component}: {count} records")
            
            # Save the merged data
            output_file = merger.save_merged_data(merged_df, date)
            print(f"✅ Saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {date}: {e}")
            
    logger.info("Data merging complete!")


if __name__ == "__main__":
    main() 
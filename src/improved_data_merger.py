"""
Improved Data Merger for Field Data Pipeline

This module handles temporal-first alignment and merging of FORCES data with InfluxDB data,
preserving temporal relationships and avoiding artificial repetition patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime, timezone
from scipy import interpolate
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedDataMerger:
    """
    Handles temporal-first merging of FORCES data with InfluxDB data.
    Preserves temporal relationships and handles missing data gracefully.
    """

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = Path(data_dir)
        self.influx_dir = self.data_dir / "INFLUX"
        self.forces_dir = self.data_dir / "FORCES"

    def load_influx_data(self, date: str) -> pd.DataFrame:
        """Load InfluxDB data for a specific date."""
        filename = f"FLIGHT_SEGMENT_data_{date}.csv"
        filepath = self.influx_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"InfluxDB file not found: {filepath}")

        logger.info(f"Loading InfluxDB data from {filename}")
        df = pd.read_csv(filepath)

        # Convert timestamp to datetime with timezone awareness
        df['_time'] = pd.to_datetime(df['_time'], format='mixed', utc=True)

        # Filter out rows with missing force data
        df = df.dropna(subset=['FLIGHT_SEGMENT_loadcells_force'])
        df = df.sort_values('_time')

        logger.info(f"Loaded {len(df)} InfluxDB records with force data")
        return df

    def load_forces_data(self, filepath: Path) -> pd.DataFrame:
        """Load FORCES data from a CSV file."""
        logger.info(f"Loading FORCES data from {filepath.name}")
        df = pd.read_csv(filepath)

        # Convert datetime column - assume local time, make timezone-aware for comparison
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        # Make timezone naive datetime timezone-aware (assume local timezone)
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')

        # Filter out rows with zero total force
        df = df[df['total_force_kg'] > 0.001]
        df = df.sort_values('datetime')

        logger.info(f"Loaded {len(df)} FORCES records")
        return df

    def detect_timezone_offset(self, influx_df: pd.DataFrame, forces_df: pd.DataFrame) -> float:
        """
        Detect timezone offset between InfluxDB (UTC) and FORCES (local) data
        by analyzing force value patterns.
        """
        logger.info("Detecting timezone offset between datasets...")

        # Extract force distributions for comparison
        influx_forces = influx_df['FLIGHT_SEGMENT_loadcells_force'].values
        forces_values = forces_df['total_force_kg'].values

        # Try different timezone offsets (in hours) and find best correlation
        best_offset = 0
        best_score = 0

        for offset_hours in range(-12, 13):  # Try -12 to +12 hours
            # Adjust FORCES time to UTC equivalent
            forces_adjusted = forces_df.copy()
            forces_adjusted['datetime_utc'] = forces_adjusted['datetime'] + pd.Timedelta(hours=offset_hours)

            # Check if there's temporal overlap
            influx_start = influx_df['_time'].min()
            influx_end = influx_df['_time'].max()
            forces_start = forces_adjusted['datetime_utc'].min()
            forces_end = forces_adjusted['datetime_utc'].max()

            # Calculate overlap
            overlap_start = max(influx_start, forces_start)
            overlap_end = min(influx_end, forces_end)

            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                total_duration = max((influx_end - influx_start).total_seconds(),
                                   (forces_end - forces_start).total_seconds())
                overlap_ratio = overlap_duration / total_duration

                # Score based on overlap ratio and force distribution similarity
                influx_hist, bins = np.histogram(influx_forces, bins=50, density=True)
                forces_hist, _ = np.histogram(forces_values, bins=bins, density=True)

                # Correlation between histograms
                hist_corr = np.corrcoef(influx_hist, forces_hist)[0, 1]
                if np.isnan(hist_corr):
                    hist_corr = 0

                score = overlap_ratio * 0.7 + abs(hist_corr) * 0.3

                if score > best_score:
                    best_score = score
                    best_offset = offset_hours

        logger.info(f"Best timezone offset: {best_offset} hours (score: {best_score:.3f})")
        return best_offset

    def temporal_interpolation_merge(self, influx_df: pd.DataFrame, forces_df: pd.DataFrame,
                                   timezone_offset_hours: float = 0) -> pd.DataFrame:
        """
        Merge data using temporal interpolation to preserve time relationships.
        """
        logger.info("Starting temporal interpolation merge...")

        # Adjust FORCES timestamps to UTC
        forces_df_adj = forces_df.copy()
        forces_df_adj['datetime_utc'] = forces_df_adj['datetime'] + pd.Timedelta(hours=timezone_offset_hours)

        # Find temporal overlap
        influx_start = influx_df['_time'].min()
        influx_end = influx_df['_time'].max()
        forces_start = forces_df_adj['datetime_utc'].min()
        forces_end = forces_df_adj['datetime_utc'].max()

        overlap_start = max(influx_start, forces_start)
        overlap_end = min(influx_end, forces_end)

        if overlap_start >= overlap_end:
            logger.warning("No temporal overlap found between datasets!")
            logger.info(f"InfluxDB: {influx_start} to {influx_end}")
            logger.info(f"FORCES (UTC): {forces_start} to {forces_end}")

            # Return InfluxDB data with NaN force components
            result_df = influx_df.copy()
            force_columns = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
            for col in force_columns:
                result_df[col] = np.nan
            result_df['forces_source'] = 'no_temporal_overlap'
            return result_df

        # Filter to overlap period for both datasets
        influx_overlap = influx_df[
            (influx_df['_time'] >= overlap_start) &
            (influx_df['_time'] <= overlap_end)
        ].copy()

        forces_overlap = forces_df_adj[
            (forces_df_adj['datetime_utc'] >= overlap_start) &
            (forces_df_adj['datetime_utc'] <= overlap_end)
        ].copy()

        logger.info(f"Temporal overlap: {overlap_start} to {overlap_end}")
        logger.info(f"Overlap duration: {(overlap_end - overlap_start).total_seconds()/3600:.2f} hours")
        logger.info(f"InfluxDB records in overlap: {len(influx_overlap)}")
        logger.info(f"FORCES records in overlap: {len(forces_overlap)}")

        if len(forces_overlap) < 2:
            logger.warning("Insufficient FORCES data in overlap period for interpolation")
            result_df = influx_overlap.copy()
            force_columns = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
            for col in force_columns:
                result_df[col] = np.nan
            result_df['forces_source'] = 'insufficient_forces_data'
            return result_df

        # Convert timestamps to numeric for interpolation
        forces_times = forces_overlap['datetime_utc'].values.astype('datetime64[ns]').astype('int64')
        influx_times = influx_overlap['_time'].values.astype('datetime64[ns]').astype('int64')

        # Prepare result DataFrame
        result_df = influx_overlap.copy()
        force_columns = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']

        # Interpolate each force component
        for col in force_columns:
            forces_values = forces_overlap[col].values

            # Use linear interpolation
            interpolator = interpolate.interp1d(
                forces_times,
                forces_values,
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )

            # Interpolate values at InfluxDB timestamps
            interpolated_values = interpolator(influx_times)
            result_df[col] = interpolated_values

        # Add metadata about interpolation
        result_df['forces_source'] = 'temporal_interpolation'
        result_df['timezone_offset_hours'] = timezone_offset_hours

        # Calculate how many values were successfully interpolated
        interpolated_count = result_df[force_columns].notna().all(axis=1).sum()
        interpolation_rate = interpolated_count / len(result_df) * 100

        logger.info(f"Successfully interpolated force components for {interpolated_count}/{len(result_df)} records ({interpolation_rate:.1f}%)")

        return result_df

    def merge_day_data_temporal(self, date: str, auto_detect_timezone: bool = True,
                              timezone_offset_hours: Optional[float] = None) -> pd.DataFrame:
        """
        Merge data for a day using temporal alignment approach.
        """
        logger.info(f"Starting temporal merge for {date}")

        # Load InfluxDB data
        influx_df = self.load_influx_data(date)

        # Get FORCES files for this date (using existing mapping logic)
        forces_files = self.get_forces_files_for_date(date)

        # Load and combine all FORCES data
        all_forces_data = []
        for forces_file in forces_files:
            forces_df = self.load_forces_data(forces_file)
            forces_df['source_file'] = forces_file.name
            all_forces_data.append(forces_df)

        combined_forces_df = pd.concat(all_forces_data, ignore_index=True)
        combined_forces_df = combined_forces_df.sort_values('datetime')
        logger.info(f"Combined {len(combined_forces_df)} FORCES records from {len(forces_files)} files")

        # Detect or use provided timezone offset
        if auto_detect_timezone and timezone_offset_hours is None:
            timezone_offset_hours = self.detect_timezone_offset(influx_df, combined_forces_df)
        elif timezone_offset_hours is None:
            timezone_offset_hours = 0

        # Perform temporal interpolation merge
        merged_df = self.temporal_interpolation_merge(
            influx_df, combined_forces_df, timezone_offset_hours
        )

        return merged_df

    def get_forces_files_for_date(self, date: str) -> List[Path]:
        """Get FORCES files for a date (using existing mapping logic)."""
        forces_files = sorted(list(self.forces_dir.glob("loadcell_readings_*.csv")))

        if date == "2025-07-16":
            return forces_files[:3]  # First 3 files
        elif date == "2025-07-17":
            return forces_files[3:5]  # Last 2 files
        else:
            raise ValueError(f"Unsupported date: {date}")

    def generate_temporal_summary_stats(self, merged_df: pd.DataFrame) -> Dict:
        """Generate summary statistics for temporally merged data."""
        force_columns = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']

        # Count how many records have interpolated force data
        has_force_data = merged_df[force_columns].notna().all(axis=1)
        interpolated_count = has_force_data.sum()

        stats = {
            'total_records': len(merged_df),
            'interpolated_records': interpolated_count,
            'interpolation_rate': interpolated_count / len(merged_df) * 100,
            'timezone_offset': merged_df['timezone_offset_hours'].iloc[0] if len(merged_df) > 0 else None,
            'force_component_coverage': {
                col: merged_df[col].notna().sum() for col in force_columns
            },
            'temporal_span': {
                'start': merged_df['_time'].min(),
                'end': merged_df['_time'].max(),
                'duration_hours': (merged_df['_time'].max() - merged_df['_time'].min()).total_seconds() / 3600
            }
        }

        return stats


def main():
    """Demonstrate the improved temporal merger."""
    merger = ImprovedDataMerger()

    dates = ["2025-07-16", "2025-07-17"]

    for date in dates:
        try:
            logger.info(f"=== Processing {date} with Temporal Approach ===")

            # Merge data temporally
            merged_df = merger.merge_day_data_temporal(date)

            # Generate and display summary statistics
            stats = merger.generate_temporal_summary_stats(merged_df)

            print(f"\n📊 Temporal Merge Summary for {date}:")
            print(f"  Total records: {stats['total_records']}")
            print(f"  Records with interpolated forces: {stats['interpolated_records']}")
            print(f"  Interpolation rate: {stats['interpolation_rate']:.1f}%")
            print(f"  Detected timezone offset: {stats['timezone_offset']} hours")
            print(f"  Temporal span: {stats['temporal_span']['duration_hours']:.1f} hours")

            # Save results
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            output_file = output_dir / f"temporal_merged_{date}.csv"
            merged_df.to_csv(output_file, index=False)
            print(f"✅ Saved temporal merge to: {output_file}")

        except Exception as e:
            logger.error(f"Error processing {date}: {e}")


if __name__ == "__main__":
    main()
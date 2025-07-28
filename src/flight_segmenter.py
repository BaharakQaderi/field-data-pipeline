"""
Flight Segmenter

This module segments the processed flight data into individual flights based on 
manually recorded time ranges from the pilot.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FlightSegmenter:
    """
    Segments processed flight data into individual flights based on time ranges.
    """
    
    def __init__(self, data_dir: Path = Path("outputs"), flight_data_dir: Path = None):
        self.data_dir = Path(data_dir)
        self.flight_data_dir = Path(flight_data_dir) if flight_data_dir else self.data_dir / "flights"
        self.flight_data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        (self.flight_data_dir / "individual_flights").mkdir(exist_ok=True)
        (self.flight_data_dir / "flight_summaries").mkdir(exist_ok=True)
        (self.flight_data_dir / "flight_metadata").mkdir(exist_ok=True)
        
    def create_flight_definition_template(self) -> str:
        """
        Create a template CSV file for flight definitions.
        
        Returns:
            Path to the created template file
        """
        template_data = {
            'flight_id': ['FLIGHT_001', 'FLIGHT_002', 'FLIGHT_003'],
            'date': ['2025-07-16', '2025-07-16', '2025-07-17'],
            'start_time': ['09:30:00', '14:15:00', '10:45:00'],
            'end_time': ['10:45:00', '15:30:00', '12:00:00'],
            'pilot_notes': ['Morning test flight', 'Afternoon session', 'Validation flight'],
            'weather_conditions': ['Clear', 'Light wind', 'Calm'],
            'flight_type': ['Test', 'Training', 'Validation']
        }
        
        template_df = pd.DataFrame(template_data)
        template_path = self.flight_data_dir / "flight_definitions_template.csv"
        
        template_df.to_csv(template_path, index=False)
        logger.info(f"Created flight definitions template: {template_path}")
        
        return str(template_path)
        
    def load_flight_definitions(self, definitions_file: str = None) -> pd.DataFrame:
        """
        Load flight definitions from CSV file.
        
        Args:
            definitions_file: Path to flight definitions CSV
            
        Returns:
            DataFrame with flight definitions
        """
        if definitions_file is None:
            # Look for flight definitions file
            possible_files = [
                self.flight_data_dir / "flight_definitions.csv",
                self.flight_data_dir / "flights.csv",
                "flight_definitions.csv",
                "flights.csv"
            ]
            
            definitions_file = None
            for file_path in possible_files:
                if Path(file_path).exists():
                    definitions_file = file_path
                    break
                    
            if definitions_file is None:
                raise FileNotFoundError(
                    f"Flight definitions file not found. Please create one using the template. "
                    f"Run create_flight_definition_template() first."
                )
        
        logger.info(f"Loading flight definitions from: {definitions_file}")
        flights_df = pd.read_csv(definitions_file)
        
        # Validate required columns
        required_cols = ['flight_id', 'date', 'start_time', 'end_time']
        missing_cols = [col for col in required_cols if col not in flights_df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in flight definitions: {missing_cols}")
            
        # Convert to datetime
        flights_df['start_datetime'] = pd.to_datetime(
            flights_df['date'] + ' ' + flights_df['start_time']
        )
        flights_df['end_datetime'] = pd.to_datetime(
            flights_df['date'] + ' ' + flights_df['end_time']
        )
        
        # Calculate duration
        flights_df['duration_minutes'] = (
            flights_df['end_datetime'] - flights_df['start_datetime']
        ).dt.total_seconds() / 60
        
        logger.info(f"Loaded {len(flights_df)} flight definitions")
        return flights_df
        
    def load_processed_data(self, date: str = None) -> pd.DataFrame:
        """Load processed flight data."""
        if date:
            files = [self.data_dir / f"processed_merged_flight_data_{date}.csv"]
        else:
            files = list(self.data_dir.glob("processed_merged_flight_data_*.csv"))
            
        if not files:
            raise FileNotFoundError("No processed flight data files found")
            
        all_data = []
        for file in files:
            if file.exists():
                df = pd.read_csv(file)
                df['_time'] = pd.to_datetime(df['_time'], format='mixed')
                df = df[df['forces_matched'] == True]  # Only matched data
                all_data.append(df)
                logger.info(f"Loaded {len(df)} records from {file.name}")
                
        combined_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        logger.info(f"Total processed data: {len(combined_data)} records")
        
        return combined_data
        
    def segment_flight_data(self, flight_def: pd.Series, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract data for a specific flight based on time range.
        
        Args:
            flight_def: Flight definition row
            processed_data: Full processed dataset
            
        Returns:
            DataFrame with data for this specific flight
        """
        start_time = flight_def['start_datetime']
        end_time = flight_def['end_datetime']
        
        # Add buffer around flight times (2 minutes before/after)
        buffer = timedelta(minutes=2)
        start_with_buffer = start_time - buffer
        end_with_buffer = end_time + buffer
        
        # Filter data for this flight
        flight_mask = (
            (processed_data['_time'] >= start_with_buffer) &
            (processed_data['_time'] <= end_with_buffer)
        )
        
        flight_data = processed_data[flight_mask].copy()
        
        if len(flight_data) == 0:
            logger.warning(f"No data found for flight {flight_def['flight_id']} "
                          f"({start_time} to {end_time})")
            return pd.DataFrame()
            
        # Add flight metadata to the data
        flight_data['flight_id'] = flight_def['flight_id']
        flight_data['flight_start'] = start_time
        flight_data['flight_end'] = end_time
        flight_data['time_from_flight_start'] = (
            flight_data['_time'] - start_time
        ).dt.total_seconds()
        
        logger.info(f"Flight {flight_def['flight_id']}: {len(flight_data)} records "
                   f"({flight_data['_time'].min()} to {flight_data['_time'].max()})")
        
        return flight_data
        
    def calculate_flight_summary(self, flight_data: pd.DataFrame, flight_def: pd.Series) -> Dict:
        """
        Calculate summary statistics for a flight.
        
        Args:
            flight_data: Flight-specific data
            flight_def: Flight definition
            
        Returns:
            Dictionary with flight summary statistics
        """
        if flight_data.empty:
            return {
                'flight_id': flight_def['flight_id'],
                'status': 'NO_DATA',
                'records_count': 0
            }
            
        force_components = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
        
        summary = {
            'flight_id': flight_def['flight_id'],
            'date': flight_def['date'],
            'planned_start': flight_def['start_datetime'],
            'planned_end': flight_def['end_datetime'],
            'actual_start': flight_data['_time'].min(),
            'actual_end': flight_data['_time'].max(),
            'planned_duration_min': flight_def['duration_minutes'],
            'actual_duration_min': (
                flight_data['_time'].max() - flight_data['_time'].min()
            ).total_seconds() / 60,
            'records_count': len(flight_data),
            'data_quality': {
                'match_rate': (flight_data['forces_matched'].sum() / len(flight_data)) * 100,
                'sampling_rate_hz': len(flight_data) / (flight_def['duration_minutes'] * 60)
            }
        }
        
        # Force statistics
        summary['force_statistics'] = {}
        
        # Total force statistics
        if 'FLIGHT_SEGMENT_loadcells_force' in flight_data.columns:
            total_force = flight_data['FLIGHT_SEGMENT_loadcells_force']
            summary['force_statistics']['total_force'] = {
                'mean': float(total_force.mean()),
                'std': float(total_force.std()),
                'min': float(total_force.min()),
                'max': float(total_force.max()),
                'median': float(total_force.median())
            }
        
        # Individual component statistics
        for component in force_components:
            if component in flight_data.columns:
                comp_data = flight_data[component].dropna()
                if len(comp_data) > 0:
                    summary['force_statistics'][component] = {
                        'mean': float(comp_data.mean()),
                        'std': float(comp_data.std()),
                        'min': float(comp_data.min()),
                        'max': float(comp_data.max()),
                        'median': float(comp_data.median()),
                        'records': len(comp_data)
                    }
        
        # Add pilot notes and metadata
        for col in ['pilot_notes', 'weather_conditions', 'flight_type']:
            if col in flight_def:
                summary[col] = flight_def[col]
                
        summary['status'] = 'SUCCESS'
        return summary
        
    def process_all_flights(self, flight_definitions_file: str = None) -> Dict:
        """
        Process all flights and create individual flight files.
        
        Args:
            flight_definitions_file: Path to flight definitions CSV
            
        Returns:
            Dictionary with processing results
        """
        logger.info("🚁 Starting flight segmentation process...")
        
        # Load flight definitions
        try:
            flights_df = self.load_flight_definitions(flight_definitions_file)
        except FileNotFoundError as e:
            logger.error(f"Flight definitions not found: {e}")
            # Create template for user
            template_path = self.create_flight_definition_template()
            logger.info(f"Created template at: {template_path}")
            logger.info("Please fill in the template with your flight data and run again.")
            return {'status': 'TEMPLATE_CREATED', 'template_path': template_path}
        
        # Load processed data
        processed_data = self.load_processed_data()
        
        if processed_data.empty:
            logger.error("No processed data available")
            return {'status': 'NO_DATA'}
        
        # Process each flight
        results = {
            'status': 'SUCCESS',
            'flights_processed': 0,
            'flights_with_data': 0,
            'total_records_segmented': 0,
            'flight_summaries': []
        }
        
        for idx, flight_def in flights_df.iterrows():
            logger.info(f"Processing flight: {flight_def['flight_id']}")
            
            # Segment flight data
            flight_data = self.segment_flight_data(flight_def, processed_data)
            
            # Calculate summary
            flight_summary = self.calculate_flight_summary(flight_data, flight_def)
            results['flight_summaries'].append(flight_summary)
            
            if not flight_data.empty:
                # Save individual flight data
                flight_filename = f"{flight_def['flight_id']}_data.csv"
                flight_path = self.flight_data_dir / "individual_flights" / flight_filename
                flight_data.to_csv(flight_path, index=False)
                
                # Save flight summary
                summary_filename = f"{flight_def['flight_id']}_summary.json"
                summary_path = self.flight_data_dir / "flight_summaries" / summary_filename
                with open(summary_path, 'w') as f:
                    json.dump(flight_summary, f, indent=2, default=str)
                
                results['flights_with_data'] += 1
                results['total_records_segmented'] += len(flight_data)
                
                logger.info(f"✅ {flight_def['flight_id']}: {len(flight_data)} records saved")
            else:
                logger.warning(f"⚠️  {flight_def['flight_id']}: No data found")
            
            results['flights_processed'] += 1
        
        # Save overall summary
        overall_summary_path = self.flight_data_dir / "flight_metadata" / "all_flights_summary.json"
        with open(overall_summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"🎯 Flight segmentation complete!")
        logger.info(f"   Processed: {results['flights_processed']} flights")
        logger.info(f"   With data: {results['flights_with_data']} flights")
        logger.info(f"   Total records: {results['total_records_segmented']:,}")
        
        return results
        
    def create_flight_overview(self) -> pd.DataFrame:
        """
        Create an overview table of all processed flights.
        
        Returns:
            DataFrame with flight overview
        """
        summary_files = list((self.flight_data_dir / "flight_summaries").glob("*_summary.json"))
        
        if not summary_files:
            logger.warning("No flight summaries found")
            return pd.DataFrame()
        
        flight_overviews = []
        
        for summary_file in summary_files:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                
            overview = {
                'flight_id': summary['flight_id'],
                'date': summary.get('date', 'Unknown'),
                'planned_duration_min': summary.get('planned_duration_min', 0),
                'actual_duration_min': summary.get('actual_duration_min', 0),
                'records_count': summary.get('records_count', 0),
                'data_quality_pct': summary.get('data_quality', {}).get('match_rate', 0),
                'sampling_rate_hz': summary.get('data_quality', {}).get('sampling_rate_hz', 0),
                'avg_total_force_kg': summary.get('force_statistics', {}).get('total_force', {}).get('mean', 0),
                'max_total_force_kg': summary.get('force_statistics', {}).get('total_force', {}).get('max', 0),
                'flight_type': summary.get('flight_type', 'Unknown'),
                'weather': summary.get('weather_conditions', 'Unknown'),
                'status': summary.get('status', 'Unknown')
            }
            
            flight_overviews.append(overview)
        
        overview_df = pd.DataFrame(flight_overviews)
        overview_df = overview_df.sort_values(['date', 'flight_id'])
        
        # Save overview
        overview_path = self.flight_data_dir / "flight_metadata" / "flights_overview.csv"
        overview_df.to_csv(overview_path, index=False)
        
        logger.info(f"Created flight overview with {len(overview_df)} flights")
        return overview_df


def main():
    """
    Main function to demonstrate flight segmentation.
    """
    segmenter = FlightSegmenter()
    
    logger.info("🚁 Flight Data Segmentation Tool")
    logger.info("This tool will segment your processed flight data into individual flights.")
    logger.info("")
    
    # Check if flight definitions exist, create template if not
    flight_defs_path = segmenter.flight_data_dir / "flight_definitions.csv"
    
    if not flight_defs_path.exists():
        logger.info("Flight definitions file not found. Creating template...")
        template_path = segmenter.create_flight_definition_template()
        logger.info("")
        logger.info("📝 NEXT STEPS:")
        logger.info(f"1. Edit the template file: {template_path}")
        logger.info("2. Fill in your flight time ranges from the pilot's notes")
        logger.info("3. Save as 'flight_definitions.csv' in the same directory")
        logger.info("4. Re-run this script")
        logger.info("")
        logger.info("Template columns:")
        logger.info("  - flight_id: Unique identifier (e.g., FLIGHT_001)")
        logger.info("  - date: Flight date (YYYY-MM-DD)")
        logger.info("  - start_time: Flight start time (HH:MM:SS)")
        logger.info("  - end_time: Flight end time (HH:MM:SS)")
        logger.info("  - pilot_notes: Any notes from the pilot")
        logger.info("  - weather_conditions: Weather during flight")
        logger.info("  - flight_type: Type of flight (Test, Training, etc.)")
        return
    
    # Process all flights
    results = segmenter.process_all_flights()
    
    if results['status'] == 'SUCCESS':
        # Create overview
        overview_df = segmenter.create_flight_overview()
        
        logger.info("")
        logger.info("📊 FLIGHT SEGMENTATION SUMMARY:")
        logger.info(f"  Total flights processed: {results['flights_processed']}")
        logger.info(f"  Flights with data: {results['flights_with_data']}")
        logger.info(f"  Total records segmented: {results['total_records_segmented']:,}")
        logger.info("")
        logger.info("📁 OUTPUT STRUCTURE:")
        logger.info(f"  Individual flight data: outputs/flights/individual_flights/")
        logger.info(f"  Flight summaries: outputs/flights/flight_summaries/")
        logger.info(f"  Flight metadata: outputs/flights/flight_metadata/")
        logger.info("")
        logger.info("🎯 Next steps:")
        logger.info("  - Review flight summaries in flight_summaries/")
        logger.info("  - Analyze individual flights in individual_flights/")
        logger.info("  - Use flights_overview.csv for comparative analysis")


if __name__ == "__main__":
    main() 
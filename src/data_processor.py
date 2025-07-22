"""
Data Processor - Hardware Corrections

This module applies hardware-specific corrections to the merged data files.
Currently applies 5th_line_kg division by 2 due to hardware setup.
"""

import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Applies hardware corrections to merged data files.
    """
    
    def __init__(self, input_dir: Path = Path("outputs"), output_dir: Path = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        
    def apply_hardware_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply hardware-specific corrections to the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with corrections applied
        """
        # Make a copy to avoid modifying the original
        corrected_df = df.copy()
        
        # Define force component columns
        force_components = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
        
        # Apply absolute value to all force components (to match InfluxDB calculation)
        abs_corrections = 0
        for component in force_components:
            if component in corrected_df.columns:
                mask = corrected_df[component].notna()
                component_count = mask.sum()
                
                if component_count > 0:
                    # Apply absolute value to make all forces positive
                    corrected_df.loc[mask, component] = corrected_df.loc[mask, component].abs()
                    abs_corrections += component_count
                    logger.info(f"Applied abs() to {component}: {component_count:,} records")
        
        if abs_corrections > 0:
            logger.info(f"Total absolute value corrections: {abs_corrections:,} values")
        
        # Apply 5th_line_kg correction (divide by 2 due to hardware setup)
        if '5th_line_kg' in corrected_df.columns:
            # Only apply correction where data exists (not NaN)
            mask = corrected_df['5th_line_kg'].notna()
            original_count = mask.sum()
            
            if original_count > 0:
                corrected_df.loc[mask, '5th_line_kg'] = corrected_df.loc[mask, '5th_line_kg'] / 2
                logger.info(f"Applied 5th_line_kg correction to {original_count:,} records (divided by 2 after abs)")
            else:
                logger.warning("No 5th_line_kg data found to correct")
        else:
            logger.warning("5th_line_kg column not found in data")
            
        return corrected_df
        
    def process_merged_file(self, input_filename: str) -> str:
        """
        Process a single merged file with hardware corrections.
        
        Args:
            input_filename: Name of input file (e.g., "merged_flight_data_2025-07-16.csv")
            
        Returns:
            Output filename
        """
        input_path = self.input_dir / input_filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        logger.info(f"Processing {input_filename}")
        
        # Load the merged data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df):,} records from {input_filename}")
        
        # Apply hardware corrections
        corrected_df = self.apply_hardware_corrections(df)
        
        # Create output filename with "processed_" prefix
        output_filename = f"processed_{input_filename}"
        output_path = self.output_dir / output_filename
        
        # Save the processed data
        corrected_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_filename}")
        
        # Generate summary
        if '5th_line_kg' in corrected_df.columns:
            corrected_records = corrected_df['5th_line_kg'].notna().sum()
            logger.info(f"Summary: {corrected_records:,} records with corrected 5th_line_kg values")
            
        return output_filename
        
    def process_all_merged_files(self):
        """
        Process all merged files in the input directory.
        """
        merged_files = list(self.input_dir.glob("merged_flight_data_*.csv"))
        
        if not merged_files:
            logger.warning(f"No merged files found in {self.input_dir}")
            return []
            
        logger.info(f"Found {len(merged_files)} merged files to process")
        
        processed_files = []
        for merged_file in merged_files:
            try:
                output_filename = self.process_merged_file(merged_file.name)
                processed_files.append(output_filename)
                logger.info(f"✅ Successfully processed {merged_file.name}")
            except Exception as e:
                logger.error(f"❌ Error processing {merged_file.name}: {e}")
                
        return processed_files
        
    def validate_corrections(self, original_filename: str, processed_filename: str):
        """
        Validate that corrections were applied correctly.
        
        Args:
            original_filename: Original merged file
            processed_filename: Processed file with corrections
        """
        logger.info(f"Validating corrections between {original_filename} and {processed_filename}")
        
        # Load both files
        original_df = pd.read_csv(self.input_dir / original_filename)
        processed_df = pd.read_csv(self.output_dir / processed_filename)
        
        # Check that files have same shape
        if original_df.shape != processed_df.shape:
            logger.error(f"Shape mismatch: Original {original_df.shape}, Processed {processed_df.shape}")
            return False
            
        # Check 5th_line_kg correction
        if '5th_line_kg' in original_df.columns and '5th_line_kg' in processed_df.columns:
            # Get rows where both have data
            mask = original_df['5th_line_kg'].notna() & processed_df['5th_line_kg'].notna()
            
            if mask.any():
                original_values = original_df.loc[mask, '5th_line_kg']
                processed_values = processed_df.loc[mask, '5th_line_kg']
                # Expected values: apply abs() first, then divide by 2
                expected_values = original_values.abs() / 2
                
                # Check if processed values match expected (within small tolerance for floating point)
                differences = abs(processed_values - expected_values)
                max_diff = differences.max()
                
                if max_diff < 1e-10:  # Very small tolerance for floating point precision
                    logger.info(f"✅ 5th_line_kg correction validated successfully")
                    logger.info(f"   Corrected {len(original_values):,} records")
                    logger.info(f"   Max difference: {max_diff:.2e}")
                    
                    # Additional validation: check that all force values are positive
                    force_components = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
                    negative_counts = {}
                    
                    for component in force_components:
                        if component in processed_df.columns:
                            component_data = processed_df[component].dropna()
                            negative_count = (component_data < 0).sum()
                            negative_counts[component] = negative_count
                            
                    total_negatives = sum(negative_counts.values())
                    if total_negatives == 0:
                        logger.info(f"✅ All force components are positive (abs correction validated)")
                    else:
                        logger.warning(f"⚠️  Found {total_negatives} negative values across components: {negative_counts}")
                    
                    return True
                else:
                    logger.error(f"❌ 5th_line_kg correction validation failed. Max difference: {max_diff}")
                    return False
            else:
                logger.warning("No 5th_line_kg data to validate")
                return True
        else:
            logger.warning("5th_line_kg column not found for validation")
            return True


def main():
    """
    Main function to process all merged files with hardware corrections.
    """
    processor = DataProcessor()
    
    logger.info("🔧 Starting hardware corrections processing...")
    
    # Process all merged files
    processed_files = processor.process_all_merged_files()
    
    if processed_files:
        logger.info(f"✅ Successfully processed {len(processed_files)} files:")
        for filename in processed_files:
            logger.info(f"   📄 {filename}")
            
        # Validate corrections for each file
        logger.info("\n🔍 Validating corrections...")
        merged_files = list(Path("outputs").glob("merged_flight_data_*.csv"))
        
        for merged_file in merged_files:
            processed_filename = f"processed_{merged_file.name}"
            if processed_filename in processed_files:
                processor.validate_corrections(merged_file.name, processed_filename)
                
    else:
        logger.error("❌ No files were processed successfully")
        
    logger.info("🎯 Hardware corrections complete!")


if __name__ == "__main__":
    main() 
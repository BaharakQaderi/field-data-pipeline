#!/usr/bin/env python3
"""
Field Data Pipeline - Main Entry Point

This script provides multiple ways to interact with your field data:
1. Run data merger to align FORCES with InfluxDB data
2. Start interactive exploration
3. Generate analysis reports
"""

import sys
import argparse
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append('src')

from src.data_merger import DataMerger, main as merger_main


def main():
    """Main entry point for the field data pipeline."""
    
    parser = argparse.ArgumentParser(description='Field Data Pipeline')
    parser.add_argument('--merge', action='store_true', 
                       help='Run data merger to align FORCES with InfluxDB data')
    parser.add_argument('--date', type=str, 
                       help='Process specific date (YYYY-MM-DD)')
    parser.add_argument('--notebook', action='store_true',
                       help='Start Jupyter notebook for interactive analysis')
    parser.add_argument('--dashboard', action='store_true',
                       help='Start interactive Streamlit dashboard')
    
    args = parser.parse_args()
    
    if args.merge:
        print("🚀 Starting data merger...")
        if args.date:
            # Merge specific date
            merger = DataMerger()
            try:
                merged_df = merger.merge_day_data(args.date)
                stats = merger.generate_summary_stats(merged_df)
                merger.save_merged_data(merged_df, args.date)
                print(f"✅ Successfully processed {args.date}")
                print(f"   Match rate: {stats['match_rate']:.1f}%")
            except Exception as e:
                print(f"❌ Error processing {args.date}: {e}")
        else:
            # Merge all available dates
            merger_main()
            
    elif args.notebook:
        import subprocess
        print("📓 Starting Jupyter notebook...")
        subprocess.run(["uv", "run", "jupyter", "notebook"])
        
    elif args.dashboard:
        import subprocess
        print("🚀 Starting interactive dashboard...")
        subprocess.run(["uv", "run", "streamlit", "run", "src/interactive_dashboard.py"])
        
    else:
        print("🔬 Field Data Pipeline")
        print("\nWelcome to your field data analysis pipeline!")
        print("\nAvailable commands:")
        print("  python main.py --merge           # Merge all data")
        print("  python main.py --merge --date YYYY-MM-DD  # Merge specific date")
        print("  python main.py --notebook        # Start Jupyter")
        print("  python main.py --dashboard       # Start interactive dashboard")
        print("  uv run jupyter notebook          # Direct Jupyter start")
        print("\nOr run the data merger directly:")
        print("  uv run python src/data_merger.py")
        print("\nData files detected:")
        
        # Show available data
        data_dir = Path("data")
        if (data_dir / "INFLUX").exists():
            influx_files = list((data_dir / "INFLUX").glob("FLIGHT_SEGMENT_*.csv"))
            print(f"  📊 InfluxDB files: {len(influx_files)}")
            
        if (data_dir / "FORCES").exists():
            forces_files = list((data_dir / "FORCES").glob("loadcell_*.csv"))
            print(f"  ⚡ FORCES files: {len(forces_files)}")


if __name__ == "__main__":
    main()

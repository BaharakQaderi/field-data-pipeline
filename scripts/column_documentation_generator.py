#!/usr/bin/env python3
"""
Column Documentation Generator

This script extracts column information from large CSV files and creates
comprehensive documentation templates to help teams understand field meanings.

Usage:
    python column_documentation_generator.py path/to/your/file.csv
"""

import pandas as pd
import sys
import os
from collections import defaultdict
from pathlib import Path


def smart_categorize_columns(columns):
    """
    Automatically categorize columns based on naming patterns
    """
    categories = {
        'Time/Timestamp': [],
        'Temperature': [],
        'Voltage/Current/Power': [],
        'Position/Movement': [],
        'Status/Control': [],
        'Brake System': [],
        'Motor/Drive': [],
        'Inverter/Converter': [],
        'Safety/Limits': [],
        'Communication': [],
        'Sensors': [],
        'Flight/Navigation': [],
        'Unknown/Other': []
    }
    
    for col in columns:
        col_lower = col.lower()
        
        if any(keyword in col_lower for keyword in ['time', 'epoch', 'iso', 'start', 'stop']):
            categories['Time/Timestamp'].append(col)
        elif any(keyword in col_lower for keyword in ['temp', 'temperature']):
            categories['Temperature'].append(col)
        elif any(keyword in col_lower for keyword in ['volt', 'current', 'power', 'energy']):
            categories['Voltage/Current/Power'].append(col)
        elif any(keyword in col_lower for keyword in ['position', 'servo', 'moving', 'percentage']):
            categories['Position/Movement'].append(col)
        elif any(keyword in col_lower for keyword in ['brake', 'br_']):
            categories['Brake System'].append(col)
        elif any(keyword in col_lower for keyword in ['motor', 'drive', 'cnt_', 'cable']):
            categories['Motor/Drive'].append(col)
        elif any(keyword in col_lower for keyword in ['inv', 'conv', 'converter', 'inverter']):
            categories['Inverter/Converter'].append(col)
        elif any(keyword in col_lower for keyword in ['fault', 'alarm', 'emergency', 'safety', 'limit', 'overtemp']):
            categories['Safety/Limits'].append(col)
        elif any(keyword in col_lower for keyword in ['status', '_en', '_fbk', 'ctrl', 'inhibit', 'rdy']):
            categories['Status/Control'].append(col)  
        elif any(keyword in col_lower for keyword in ['comm', 'source', 'measurement']):
            categories['Communication'].append(col)
        elif any(keyword in col_lower for keyword in ['gps', 'altitude', 'latitude', 'longitude', 'velocity', 'acceleration', 'pitch', 'roll', 'yaw']):
            categories['Flight/Navigation'].append(col)
        else:
            categories['Unknown/Other'].append(col)
    
    return categories


def analyze_column_groups(columns):
    """
    Group columns by prefix to understand system organization
    """
    column_groups = defaultdict(list)
    
    for col in columns:
        if col.startswith('OPC_'):
            # Extract the prefix (first part after OPC_)
            parts = col.split('_')
            if len(parts) >= 2:
                prefix = '_'.join(parts[:2])  # e.g., OPC_BRAKE, OPC_CONV1
            else:
                prefix = parts[0]
            column_groups[prefix].append(col)
        elif col.startswith('FLIGHT_SEGMENT_'):
            # Group flight segment data
            parts = col.split('_')
            if len(parts) >= 3:
                prefix = '_'.join(parts[:3])  # e.g., FLIGHT_SEGMENT_gps
            else:
                prefix = '_'.join(parts[:2])
            column_groups[prefix].append(col)
        else:
            column_groups['Other'].append(col)
    
    return column_groups


def create_enhanced_documentation_template(df, output_file):
    """
    Create an enhanced documentation template with smart categorization
    """
    print(f"Analyzing {len(df.columns)} columns...")
    
    categories = smart_categorize_columns(df.columns.tolist())
    columns_info = []
    
    for i, col in enumerate(df.columns):
        if i % 100 == 0:
            print(f"Processing column {i+1}/{len(df.columns)}")
            
        # Find which category this column belongs to
        col_category = 'Unknown/Other'
        for category, cols in categories.items():
            if col in cols:
                col_category = category
                break
        
        # Basic column information
        col_info = {
            'column_name': col,
            'auto_category': col_category,
            'data_type': str(df[col].dtype),
            'non_null_count': df[col].count(),
            'total_count': len(df),
            'null_percentage': round((len(df) - df[col].count()) / len(df) * 100, 2),
        }
        
        # Sample values (first few non-null values)
        try:
            sample_values = df[col].dropna().head(3).tolist()
            col_info['sample_values'] = str(sample_values)[:100]  # Truncate long values
        except:
            col_info['sample_values'] = 'Error getting samples'
        
        # Statistics for numeric columns
        if df[col].dtype in ['int64', 'float64']:
            try:
                col_info['min_value'] = df[col].min()
                col_info['max_value'] = df[col].max() 
                col_info['mean_value'] = round(df[col].mean(), 4) if pd.notna(df[col].mean()) else None
                col_info['unique_values'] = df[col].nunique()
            except:
                col_info['min_value'] = None
                col_info['max_value'] = None
                col_info['mean_value'] = None
                col_info['unique_values'] = None
        elif df[col].dtype == 'bool':
            col_info['min_value'] = None
            col_info['max_value'] = None
            col_info['mean_value'] = None
            col_info['unique_values'] = df[col].nunique()
        else:
            col_info['min_value'] = None
            col_info['max_value'] = None  
            col_info['mean_value'] = None
            col_info['unique_values'] = df[col].nunique() if df[col].nunique() < 50 else 'Too many'
            
        # Add empty fields for manual documentation
        col_info['manual_category'] = ''  # Override auto category if needed
        col_info['description'] = ''  # To be filled manually
        col_info['unit'] = ''  # To be filled manually
        col_info['importance'] = ''  # To be filled manually (critical, high, medium, low)
        col_info['related_systems'] = ''  # What systems this relates to
        col_info['notes'] = ''  # To be filled manually
        
        columns_info.append(col_info)
    
    # Create DataFrame and save to CSV
    doc_df = pd.DataFrame(columns_info)
    doc_df = doc_df.sort_values(['auto_category', 'column_name'])  # Sort by category then name
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    doc_df.to_csv(output_file, index=False)
    
    print(f"\nEnhanced documentation template created: {output_file}")
    print(f"Total columns documented: {len(columns_info)}")
    
    # Create summary by category
    summary = doc_df.groupby('auto_category').size().sort_values(ascending=False)
    print(f"\nColumns by category:")
    for cat, count in summary.items():
        print(f"  {cat}: {count} columns")
    
    return doc_df, categories


def print_category_details(categories):
    """
    Print detailed breakdown of columns by category
    """
    print("\n" + "="*60)
    print("DETAILED COLUMN CATEGORIZATION")
    print("="*60)
    
    for category, cols in categories.items():
        if cols:  # Only show categories that have columns
            print(f"\n{category} ({len(cols)} columns):")
            for col in cols[:10]:  # Show first 10 columns
                print(f"  - {col}")
            if len(cols) > 10:
                print(f"  ... and {len(cols) - 10} more")


def main():
    """
    Main function to process CSV file and generate documentation
    """
    if len(sys.argv) != 2:
        print("Usage: python column_documentation_generator.py <csv_file_path>")
        print("Example: python column_documentation_generator.py ../data/INFLUX/OPC_data_2025-07-29.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found!")
        sys.exit(1)
    
    print(f"Loading CSV file: {csv_file}")
    print("This may take a moment for large files...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Generate output filename
        base_name = Path(csv_file).stem
        output_file = f"../outputs/{base_name}_column_documentation.csv"
        
        # Create documentation template
        doc_df, categories = create_enhanced_documentation_template(df, output_file)
        
        # Print detailed categorization
        print_category_details(categories)
        
        # Create prefix analysis
        column_groups = analyze_column_groups(df.columns.tolist())
        
        print("\n" + "="*60)
        print("COLUMN GROUPS BY PREFIX")
        print("="*60)
        
        for prefix, cols in sorted(column_groups.items()):
            if len(cols) > 1:  # Only show groups with multiple columns
                print(f"\n{prefix} ({len(cols)} columns):")
                for col in cols[:8]:  # Show first 8 columns
                    print(f"  - {col}")
                if len(cols) > 8:
                    print(f"  ... and {len(cols) - 8} more")
        
        print(f"\n{'='*60}")
        print("NEXT STEPS:")
        print("="*60)
        print(f"1. Open the documentation file: {output_file}")
        print("2. Share with your team to fill in the manual fields:")
        print("   - description: What this field represents")
        print("   - unit: Measurement unit (V, A, °C, etc.)")
        print("   - importance: Critical/High/Medium/Low")
        print("   - related_systems: Which aircraft systems this relates to")
        print("3. Use the auto_category grouping to assign team members")
        print("4. Focus on 'Critical' and 'High' importance fields first")
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
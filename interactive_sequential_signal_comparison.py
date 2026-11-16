#!/usr/bin/env python3
"""
Interactive Sequential signal comparison - arrange Raspberry Pi files in order
and compare with InfluxDB data using sequential index with interactive plots
and timestamp information for zoom analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path

def load_raspi_files_sequentially(date):
    """Load Raspberry Pi files in sequential order, preserving timestamps."""

    raspi_dir = Path("data/FORCES/log_raspi2")

    if date == "2025-07-16":
        filenames = [
            "loadcell_readings_20250715_161829.csv",
            "loadcell_readings_20250715_183235.csv",
            "loadcell_readings_20250715_211757.csv"
        ]
    elif date == "2025-07-17":
        filenames = [
            "loadcell_readings_20250715_224238.csv",
            "loadcell_readings_20250716_003735.csv"
        ]
    else:
        raise ValueError(f"Unsupported date: {date}")

    print(f"\n🔍 Loading Raspberry Pi files sequentially for {date}:")

    all_data = []
    sequential_index = 0

    for i, filename in enumerate(filenames):
        filepath = raspi_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df = df[df['total_force_kg'] > 0.001]  # Only meaningful data

            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

            # Create sequential index
            df['sequential_index'] = range(sequential_index, sequential_index + len(df))
            df['file_number'] = i + 1
            df['source_file'] = filename

            # Add hover information
            df['hover_info'] = df.apply(lambda row:
                f"File: {row['source_file']}<br>" +
                f"Timestamp: {row['datetime']}<br>" +
                f"Force: {row['total_force_kg']:.3f} kg<br>" +
                f"Sequential Index: {row['sequential_index']}", axis=1)

            all_data.append(df)
            sequential_index += len(df)

            print(f"  File {i+1}: {filename}")
            print(f"    Records: {len(df)}")
            print(f"    Index range: {df['sequential_index'].min()} - {df['sequential_index'].max()}")
            print(f"    Force range: {df['total_force_kg'].min():.1f} - {df['total_force_kg'].max():.1f} kg")
            print(f"    Time range: {df['timestamp'].min()} - {df['timestamp'].max()}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"  📊 Total sequential records: {len(combined_df)}")
        return combined_df
    else:
        return None

def load_influx_sequential(date):
    """Load InfluxDB data and create sequential index, preserving timestamps."""

    filepath = Path(f"data/INFLUX/FLIGHT_SEGMENT_data_{date}.csv")
    df = pd.read_csv(filepath)
    df['_time'] = pd.to_datetime(df['_time'], format='mixed')
    df = df.sort_values('_time')
    df = df[df['FLIGHT_SEGMENT_loadcells_force'] > 0.001]  # Only meaningful data

    # Create sequential index
    df['sequential_index'] = range(len(df))

    # Add hover information
    df['hover_info'] = df.apply(lambda row:
        f"InfluxDB Data<br>" +
        f"Timestamp: {row['_time']}<br>" +
        f"Force: {row['FLIGHT_SEGMENT_loadcells_force']:.3f} kg<br>" +
        f"Sequential Index: {row['sequential_index']}", axis=1)

    print(f"\n🌐 InfluxDB {date}:")
    print(f"  Records: {len(df)}")
    print(f"  Force range: {df['FLIGHT_SEGMENT_loadcells_force'].min():.1f} - {df['FLIGHT_SEGMENT_loadcells_force'].max():.1f} kg")
    print(f"  Time range: {df['_time'].min()} - {df['_time'].max()}")

    return df

def create_interactive_sequential_comparison(date):
    """Create interactive comparison using sequential indices with timestamp info."""

    print(f"\n📊 Creating Interactive Sequential Comparison for {date}")
    print("=" * 50)

    # Load data
    raspi_df = load_raspi_files_sequentially(date)
    influx_df = load_influx_sequential(date)

    if raspi_df is None:
        return

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            'Raspberry Pi - Sequential Force Signal',
            'InfluxDB - Sequential Force Signal',
            'Normalized Overlay Comparison'
        ],
        vertical_spacing=0.08,
        shared_xaxes=False
    )

    # Panel 1: Raspberry Pi sequential signal with file coloring
    colors = ['blue', 'green', 'orange', 'red', 'purple']

    for file_num in sorted(raspi_df['file_number'].unique()):
        file_data = raspi_df[raspi_df['file_number'] == file_num]
        color = colors[(file_num - 1) % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=file_data['sequential_index'],
                y=file_data['total_force_kg'],
                mode='lines',
                name=f'File {file_num} ({file_data["source_file"].iloc[0]})',
                line=dict(color=color, width=1),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=file_data['hover_info'],
                showlegend=True
            ),
            row=1, col=1
        )

    # Panel 2: InfluxDB sequential signal
    fig.add_trace(
        go.Scatter(
            x=influx_df['sequential_index'],
            y=influx_df['FLIGHT_SEGMENT_loadcells_force'],
            mode='lines',
            name=f'InfluxDB ({len(influx_df)} points)',
            line=dict(color='red', width=1),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=influx_df['hover_info'],
            showlegend=True
        ),
        row=2, col=1
    )

    # Panel 3: Normalized overlay comparison
    min_length = min(len(raspi_df), len(influx_df))

    raspi_normalized = np.interp(np.linspace(0, 1, min_length),
                                np.linspace(0, 1, len(raspi_df)),
                                raspi_df['total_force_kg'].values)

    influx_normalized = np.interp(np.linspace(0, 1, min_length),
                                 np.linspace(0, 1, len(influx_df)),
                                 influx_df['FLIGHT_SEGMENT_loadcells_force'].values)

    # Create timestamp arrays for normalized data
    raspi_timestamps_normalized = np.interp(np.linspace(0, 1, min_length),
                                          np.linspace(0, 1, len(raspi_df)),
                                          range(len(raspi_df)))
    raspi_timestamps_normalized = raspi_timestamps_normalized.astype(int)

    influx_timestamps_normalized = np.interp(np.linspace(0, 1, min_length),
                                           np.linspace(0, 1, len(influx_df)),
                                           range(len(influx_df)))
    influx_timestamps_normalized = influx_timestamps_normalized.astype(int)

    # Create hover info for normalized data
    raspi_hover_normalized = []
    influx_hover_normalized = []

    for i, idx in enumerate(raspi_timestamps_normalized):
        if idx < len(raspi_df):
            raspi_hover_normalized.append(
                f"Raspberry Pi (Normalized)<br>" +
                f"Original Timestamp: {raspi_df.iloc[idx]['timestamp']}<br>" +
                f"Original Force: {raspi_df.iloc[idx]['total_force_kg']:.3f} kg<br>" +
                f"Normalized Index: {i}<br>" +
                f"Normalized Force: {raspi_normalized[i]:.3f} kg"
            )
        else:
            raspi_hover_normalized.append("No data")

    for i, idx in enumerate(influx_timestamps_normalized):
        if idx < len(influx_df):
            influx_hover_normalized.append(
                f"InfluxDB (Normalized)<br>" +
                f"Original Timestamp: {influx_df.iloc[idx]['_time']}<br>" +
                f"Original Force: {influx_df.iloc[idx]['FLIGHT_SEGMENT_loadcells_force']:.3f} kg<br>" +
                f"Normalized Index: {i}<br>" +
                f"Normalized Force: {influx_normalized[i]:.3f} kg"
            )
        else:
            influx_hover_normalized.append("No data")

    fig.add_trace(
        go.Scatter(
            x=list(range(min_length)),
            y=raspi_normalized,
            mode='lines',
            name='Raspberry Pi (normalized)',
            line=dict(color='blue', width=2),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=raspi_hover_normalized,
            showlegend=True
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(min_length)),
            y=influx_normalized,
            mode='lines',
            name='InfluxDB (normalized)',
            line=dict(color='red', width=2),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=influx_hover_normalized,
            showlegend=True
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'Interactive Sequential Signal Comparison - {date}<br>Raspberry Pi vs InfluxDB (Sequential Order)',
        height=1000,
        showlegend=True,
        hovermode='closest'
    )

    # Update axes
    fig.update_yaxes(title_text="Force (kg)", row=1, col=1)
    fig.update_yaxes(title_text="Force (kg)", row=2, col=1)
    fig.update_yaxes(title_text="Force (kg)", row=3, col=1)
    fig.update_xaxes(title_text="Sequential Index", row=1, col=1)
    fig.update_xaxes(title_text="Sequential Index", row=2, col=1)
    fig.update_xaxes(title_text="Normalized Sequential Index", row=3, col=1)

    # Save interactive plot
    output_path = f"interactive_sequential_comparison_{date}.html"
    fig.write_html(output_path)
    print(f"💾 Interactive sequential comparison saved to: {output_path}")

    # Show plot
    fig.show()

    # Statistics
    print(f"\n📈 Statistics for {date}:")
    print(f"Raspberry Pi: mean={raspi_df['total_force_kg'].mean():.1f} kg, max={raspi_df['total_force_kg'].max():.1f} kg")
    print(f"InfluxDB: mean={influx_df['FLIGHT_SEGMENT_loadcells_force'].mean():.1f} kg, max={influx_df['FLIGHT_SEGMENT_loadcells_force'].max():.1f} kg")

    # Correlation of normalized signals
    correlation = np.corrcoef(raspi_normalized, influx_normalized)[0, 1]
    print(f"Normalized signal correlation: {correlation:.3f}")

    return fig, raspi_df, influx_df

def create_timestamp_analysis_tool(raspi_df, influx_df, date):
    """Create an additional tool to analyze specific time ranges."""

    print(f"\n🕐 Creating Timestamp Analysis Tool for {date}")

    # Create a separate plot for timestamp analysis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            'Raspberry Pi - Time vs Force',
            'InfluxDB - Time vs Force'
        ],
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    # Raspberry Pi time series
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for file_num in sorted(raspi_df['file_number'].unique()):
        file_data = raspi_df[raspi_df['file_number'] == file_num]
        color = colors[(file_num - 1) % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=file_data['timestamp'],
                y=file_data['total_force_kg'],
                mode='lines',
                name=f'RasPi File {file_num}',
                line=dict(color=color, width=1),
                customdata=np.column_stack((
                    file_data['datetime'],
                    file_data['Backline_Left_kg'],
                    file_data['Backline_Right_kg'],
                    file_data['5th_line_kg'],
                    file_data['Frontline_kg']
                )),
                hovertemplate='<b>Raspberry Pi</b><br>' +
                            'Timestamp: %{x}<br>' +
                            'DateTime: %{customdata[0]}<br>' +
                            'Backline_Left_kg: %{customdata[1]:.3f}<br>' +
                            'Backline_Right_kg: %{customdata[2]:.3f}<br>' +
                            '5th_line_kg: %{customdata[3]:.3f}<br>' +
                            'Frontline_kg: %{customdata[4]:.3f}<br>' +
                            'Total_force_kg: %{y:.3f}<br>' +
                            'File: ' + file_data['source_file'].iloc[0] + '<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )

    # InfluxDB time series
    fig.add_trace(
        go.Scatter(
            x=influx_df['_time'],
            y=influx_df['FLIGHT_SEGMENT_loadcells_force'],
            mode='lines',
            name='InfluxDB',
            line=dict(color='red', width=1),
            hovertemplate='<b>InfluxDB</b><br>' +
                        'Time: %{x}<br>' +
                        'Force: %{y:.3f} kg<extra></extra>',
            showlegend=True
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f'Timestamp Analysis Tool - {date}<br>Click and drag to zoom, hover for exact timestamps',
        height=800,
        hovermode='closest'
    )

    fig.update_yaxes(title_text="Force (kg)", row=1, col=1)
    fig.update_yaxes(title_text="Force (kg)", row=2, col=1)
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)

    # Save timestamp analysis tool
    timestamp_output_path = f"timestamp_analysis_{date}.html"
    fig.write_html(timestamp_output_path)
    print(f"💾 Timestamp analysis tool saved to: {timestamp_output_path}")

    fig.show()

    return fig

def main():
    """Main function."""

    print("🚁 Interactive Sequential Signal Comparison Tool")
    print("=" * 50)

    for date in ["2025-07-16", "2025-07-17"]:
        # Create main interactive comparison
        main_fig, raspi_df, influx_df = create_interactive_sequential_comparison(date)

        # Create timestamp analysis tool
        timestamp_fig = create_timestamp_analysis_tool(raspi_df, influx_df, date)

        print(f"\n✅ Completed analysis for {date}")
        print(f"   - Interactive sequential comparison: interactive_sequential_comparison_{date}.html")
        print(f"   - Timestamp analysis tool: timestamp_analysis_{date}.html")
        print("\n" + "="*50)

    print("\n🎯 Usage Instructions:")
    print("1. Open the interactive_sequential_comparison_*.html files to see the main comparison")
    print("2. Hover over data points to see exact timestamps and force values")
    print("3. Use the timestamp_analysis_*.html files to zoom into specific time periods")
    print("4. Use the zoom and pan tools to analyze specific flight segments")
    print("5. The hover tooltips show original timestamps for correlation with flight logs")

if __name__ == "__main__":
    main()
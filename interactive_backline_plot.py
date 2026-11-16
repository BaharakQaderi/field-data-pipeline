#!/usr/bin/env python3
"""
Create an interactive time series plot of backline left and right forces
that you can click and zoom to explore details.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

def create_interactive_backline_plot(filename="loadcell_readings_20250715_183235.csv"):
    """Create interactive plot of backline forces."""

    print(f"📊 Creating Interactive Backline Plot from: {filename}")
    print("=" * 60)

    # Load the specific file
    filepath = Path("data/FORCES") / filename
    if not filepath.exists():
        # Try the log_raspi2 directory
        filepath = Path("data/FORCES/log_raspi2") / filename

    if not filepath.exists():
        print(f"❌ File not found: {filename}")
        return

    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df = df.sort_values('datetime')

    print(f"Total records: {len(df)}")
    print(f"Time span: {df['datetime'].min()} to {df['datetime'].max()}")

    # Filter meaningful data (but include more range for context)
    meaningful_df = df[df['total_force_kg'] > 0.001].copy()
    print(f"Records with meaningful force: {len(meaningful_df)}")

    if len(meaningful_df) == 0:
        print("❌ No meaningful force data found!")
        return

    # Create interactive plot
    fig = go.Figure()

    # Add Backline Left
    fig.add_trace(go.Scatter(
        x=meaningful_df['datetime'],
        y=meaningful_df['Backline_Left_kg'],
        mode='lines',
        name='Backline Left',
        line=dict(color='blue', width=1),
        hovertemplate='<b>Backline Left</b><br>' +
                      'Time: %{x}<br>' +
                      'Force: %{y:.3f} kg<br>' +
                      '<extra></extra>'
    ))

    # Add Backline Right
    fig.add_trace(go.Scatter(
        x=meaningful_df['datetime'],
        y=meaningful_df['Backline_Right_kg'],
        mode='lines',
        name='Backline Right',
        line=dict(color='red', width=1),
        hovertemplate='<b>Backline Right</b><br>' +
                      'Time: %{x}<br>' +
                      'Force: %{y:.3f} kg<br>' +
                      '<extra></extra>'
    ))

    # Update layout for interactivity
    fig.update_layout(
        title=f'Interactive Backline Forces - {filename}<br><sub>Click and drag to zoom, double-click to reset</sub>',
        xaxis_title='Time',
        yaxis_title='Force (kg)',
        hovermode='x unified',
        showlegend=True,
        width=1400,
        height=700,
        xaxis=dict(
            rangeslider=dict(visible=True),  # Add range slider
            type="date"
        ),
        yaxis=dict(
            fixedrange=False  # Allow y-axis zooming
        ),
        template='plotly_white'
    )

    # Add crossfilter cursor
    fig.update_layout(
        xaxis=dict(
            showspikes=True,
            spikecolor="green",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1
        ),
        yaxis=dict(
            showspikes=True,
            spikecolor="green",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1
        )
    )

    # Save as HTML
    output_path = f"outputs/interactive_backline_{filename.replace('.csv', '.html')}"
    fig.write_html(output_path)
    print(f"💾 Interactive plot saved to: {output_path}")
    print("🖱️  Features:")
    print("   • Click and drag to zoom into any time period")
    print("   • Double-click to reset zoom")
    print("   • Use the range slider at the bottom to navigate")
    print("   • Hover over points to see exact values")
    print("   • Toggle traces on/off by clicking legend")

    # Also show in browser if available
    try:
        fig.show()
        print("📱 Plot should open in your default browser")
    except:
        print("📱 Open the HTML file in your browser to interact with the plot")

    # Print some quick stats
    print(f"\n📊 Quick Stats:")
    print(f"Backline Left:  {meaningful_df['Backline_Left_kg'].min():.3f} to {meaningful_df['Backline_Left_kg'].max():.3f} kg")
    print(f"Backline Right: {meaningful_df['Backline_Right_kg'].min():.3f} to {meaningful_df['Backline_Right_kg'].max():.3f} kg")

def create_multiple_file_interactive():
    """Create interactive plots for multiple files for comparison."""

    files = [
        "loadcell_readings_20250715_161829.csv",
        "loadcell_readings_20250715_183235.csv",
        "loadcell_readings_20250715_211757.csv"
    ]

    print("🚁 Creating Interactive Plots for July 16 Files")
    print("=" * 60)

    for filename in files:
        create_interactive_backline_plot(filename)
        print()

def main():
    """Main function."""

    print("🚁 Interactive Backline Analysis Tool")
    print("=" * 50)

    # Create interactive plot for the specific requested file
    create_interactive_backline_plot("loadcell_readings_20250715_183235.csv")

    # Ask if user wants plots for other files too
    print(f"\n❓ Would you like interactive plots for the other July 16 files too?")
    print("   (Run with --all flag or modify main() to include create_multiple_file_interactive())")

if __name__ == "__main__":
    main()
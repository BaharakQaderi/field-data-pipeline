#!/usr/bin/env python3
"""
Create an interactive two-panel comparison of backline forces:
- Panel 1: Original Raspberry Pi data
- Panel 2: Merged flight analysis data
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def create_dual_panel_comparison():
    """Create interactive two-panel comparison of backline forces."""

    print("📊 Creating Dual Panel Interactive Comparison")
    print("=" * 60)

    # Load original Raspberry Pi data
    raspi_filepath = Path("data/FORCES/log_raspi2/loadcell_readings_20250715_183235.csv")
    if not raspi_filepath.exists():
        raspi_filepath = Path("data/FORCES/loadcell_readings_20250715_183235.csv")

    print(f"Loading original Raspberry Pi data from: {raspi_filepath.name}")
    raspi_df = pd.read_csv(raspi_filepath)
    raspi_df['datetime'] = pd.to_datetime(raspi_df['datetime'], format='mixed')
    raspi_df = raspi_df.sort_values('datetime')
    raspi_meaningful = raspi_df[raspi_df['total_force_kg'] > 0.001].copy()

    print(f"  Original data: {len(raspi_meaningful)} meaningful records")
    print(f"  Time span: {raspi_meaningful['datetime'].min()} to {raspi_meaningful['datetime'].max()}")

    # Load merged flight analysis data
    flight_filepath = Path("flight_analysis_july16/2025_07_16_11_47/flight_data.csv")
    print(f"Loading merged flight analysis data from: {flight_filepath}")

    flight_df = pd.read_csv(flight_filepath)
    flight_df['_time'] = pd.to_datetime(flight_df['_time'], format='mixed')
    flight_df = flight_df.sort_values('_time')

    # Filter to records with meaningful backline data
    flight_meaningful = flight_df[
        (flight_df['forces_matched'] == True) &
        (flight_df['Backline_Left_kg'].notna()) &
        (flight_df['Backline_Right_kg'].notna())
    ].copy()

    print(f"  Merged data: {len(flight_meaningful)} meaningful records")
    print(f"  Time span: {flight_meaningful['_time'].min()} to {flight_meaningful['_time'].max()}")

    # Create subplot with 2 panels - NO shared x-axes
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        subplot_titles=(
            'Original Raspberry Pi Data - loadcell_readings_20250715_183235.csv',
            'Merged Flight Analysis Data - flight_data.csv'
        ),
        vertical_spacing=0.08
    )

    # Panel 1: Original Raspberry Pi data (absolute values)
    fig.add_trace(
        go.Scatter(
            x=raspi_meaningful['datetime'],
            y=raspi_meaningful['Backline_Left_kg'].abs(),
            mode='lines',
            name='Raspberry Pi - Left',
            line=dict(color='blue', width=1),
            hovertemplate='<b>Original - Left</b><br>' +
                          'Time: %{x}<br>' +
                          'Force: %{y:.3f} kg<br>' +
                          '<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=raspi_meaningful['datetime'],
            y=raspi_meaningful['Backline_Right_kg'].abs(),
            mode='lines',
            name='Raspberry Pi - Right',
            line=dict(color='red', width=1),
            hovertemplate='<b>Original - Right</b><br>' +
                          'Time: %{x}<br>' +
                          'Force: %{y:.3f} kg<br>' +
                          '<extra></extra>'
        ),
        row=1, col=1
    )

    # Panel 2: Merged flight analysis data
    fig.add_trace(
        go.Scatter(
            x=flight_meaningful['_time'],
            y=flight_meaningful['Backline_Left_kg'],
            mode='lines',
            name='Merged - Left',
            line=dict(color='darkblue', width=1),
            hovertemplate='<b>Merged - Left</b><br>' +
                          'Time: %{x}<br>' +
                          'Force: %{y:.3f} kg<br>' +
                          'Match: %{customdata}<br>' +
                          '<extra></extra>',
            customdata=flight_meaningful['forces_matched']
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=flight_meaningful['_time'],
            y=flight_meaningful['Backline_Right_kg'],
            mode='lines',
            name='Merged - Right',
            line=dict(color='darkred', width=1),
            hovertemplate='<b>Merged - Right</b><br>' +
                          'Time: %{x}<br>' +
                          'Force: %{y:.3f} kg<br>' +
                          'Match: %{customdata}<br>' +
                          '<extra></extra>',
            customdata=flight_meaningful['forces_matched']
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='Interactive Backline Forces Comparison<br><sub>Original Raspberry Pi vs Merged Flight Data - Click and drag to zoom</sub>',
            x=0.5
        ),
        hovermode='x unified',
        showlegend=True,
        width=1400,
        height=900,
        template='plotly_white'
    )

    # Update x-axes for both panels separately
    fig.update_xaxes(
        showspikes=True,
        spikecolor="green",
        spikesnap="cursor",
        spikemode="across",
        spikethickness=1,
        title_text="Time (Original Raspberry Pi)",
        row=1, col=1
    )

    fig.update_xaxes(
        showspikes=True,
        spikecolor="green",
        spikesnap="cursor",
        spikemode="across",
        spikethickness=1,
        title_text="Time (Merged Flight Data)",
        row=2, col=1
    )

    # Update y-axes
    fig.update_yaxes(
        title_text="Force (kg)",
        showspikes=True,
        spikecolor="green",
        spikesnap="cursor",
        spikemode="across",
        spikethickness=1
    )

    # Add range sliders to both panels (no traces shown)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                bgcolor="white",
                bordercolor="lightgray",
                borderwidth=1
            ),
            type="date"
        ),
        xaxis2=dict(
            rangeslider=dict(
                visible=True,
                bgcolor="white",
                bordercolor="lightgray",
                borderwidth=1
            ),
            type="date"
        )
    )

    # Save as HTML
    output_path = "outputs/interactive_backline_comparison.html"
    fig.write_html(output_path)
    print(f"💾 Interactive comparison saved to: {output_path}")

    print(f"\n🖱️  Interactive Features:")
    print("   • Click and drag to zoom into any time period (each panel zooms independently)")
    print("   • Double-click to reset zoom")
    print("   • Use the range slider on each panel to navigate independently")
    print("   • Hover over points to see exact values")
    print("   • Toggle traces on/off by clicking legend")
    print("   • Compare original timing patterns with merged data patterns")
    print("   • Each panel uses its own timestamp system (no time synchronization)")

    # Show in browser
    try:
        fig.show()
        print("📱 Plot should open in your default browser")
    except:
        print("📱 Open the HTML file in your browser to interact with the plot")

    # Print comparison statistics
    print(f"\n📊 Data Comparison Statistics:")
    print("-" * 50)

    print(f"Original Raspberry Pi Data:")
    print(f"  Records: {len(raspi_meaningful)}")
    print(f"  Left range: {raspi_meaningful['Backline_Left_kg'].min():.3f} to {raspi_meaningful['Backline_Left_kg'].max():.3f} kg")
    print(f"  Right range: {raspi_meaningful['Backline_Right_kg'].min():.3f} to {raspi_meaningful['Backline_Right_kg'].max():.3f} kg")

    print(f"\nMerged Flight Analysis Data:")
    print(f"  Records: {len(flight_meaningful)}")
    print(f"  Left range: {flight_meaningful['Backline_Left_kg'].min():.3f} to {flight_meaningful['Backline_Right_kg'].max():.3f} kg")
    print(f"  Right range: {flight_meaningful['Backline_Right_kg'].min():.3f} to {flight_meaningful['Backline_Right_kg'].max():.3f} kg")

    # Check for timing variation preservation
    print(f"\nTiming Variation Analysis:")
    print("-" * 30)

    raspi_left_diff = raspi_meaningful['Backline_Left_kg'].diff().abs().mean()
    raspi_right_diff = raspi_meaningful['Backline_Right_kg'].diff().abs().mean()

    flight_left_diff = flight_meaningful['Backline_Left_kg'].diff().abs().mean()
    flight_right_diff = flight_meaningful['Backline_Right_kg'].diff().abs().mean()

    print(f"Original - Mean consecutive differences:")
    print(f"  Left: {raspi_left_diff:.6f} kg")
    print(f"  Right: {raspi_right_diff:.6f} kg")

    print(f"Merged - Mean consecutive differences:")
    print(f"  Left: {flight_left_diff:.6f} kg")
    print(f"  Right: {flight_right_diff:.6f} kg")

    if flight_left_diff > 0.001 and flight_right_diff > 0.001:
        print("✅ GOOD: Merged data shows timing variations!")
    else:
        print("⚠️  WARNING: Merged data may have reduced timing variations")

def main():
    """Main function."""

    print("🚁 Interactive Backline Comparison Tool")
    print("=" * 50)

    create_dual_panel_comparison()

if __name__ == "__main__":
    main()
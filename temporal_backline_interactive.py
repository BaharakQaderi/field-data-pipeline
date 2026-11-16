#!/usr/bin/env python3
"""
Interactive visualization of backline forces from temporal merged data
to show how they change over time naturally (without artificial repetition).
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

def create_interactive_backline_timeseries():
    """Create interactive time series of backline forces from temporal merged data."""

    print("🎯 Creating Interactive Backline Force Time Series")
    print("=" * 60)

    # Load the temporal merged data
    temporal_file = Path("outputs/temporal_merged_2025-07-16.csv")

    if not temporal_file.exists():
        print(f"❌ Temporal merged file not found: {temporal_file}")
        print("Run the improved_data_merger.py first to generate temporal merged data.")
        return

    print(f"📊 Loading temporal merged data from: {temporal_file}")
    df = pd.read_csv(temporal_file)

    # Convert timestamp to datetime
    df['_time'] = pd.to_datetime(df['_time'])

    # Filter to records with interpolated force data
    force_cols = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
    has_forces = df[force_cols].notna().all(axis=1)
    df_forces = df[has_forces].copy()

    print(f"Records with interpolated forces: {len(df_forces)}/{len(df)} ({100*len(df_forces)/len(df):.1f}%)")

    if len(df_forces) == 0:
        print("❌ No interpolated force data found!")
        return

    # Calculate total force and additional metrics
    df_forces['Total_Force_kg'] = df_forces[force_cols].sum(axis=1)
    df_forces['Backline_Total_kg'] = df_forces['Backline_Left_kg'] + df_forces['Backline_Right_kg']
    df_forces['Left_Right_Difference'] = df_forces['Backline_Left_kg'] - df_forces['Backline_Right_kg']
    df_forces['Left_Right_Ratio'] = df_forces['Backline_Left_kg'] / (df_forces['Backline_Right_kg'] + 0.001)  # Avoid div by 0

    # Add time-based features
    df_forces['Time_Minutes'] = (df_forces['_time'] - df_forces['_time'].min()).dt.total_seconds() / 60

    print(f"\n📈 Force Data Summary:")
    print(f"  Time span: {df_forces['_time'].min()} to {df_forces['_time'].max()}")
    print(f"  Duration: {df_forces['Time_Minutes'].max():.1f} minutes")
    print(f"  Backline Left range: {df_forces['Backline_Left_kg'].min():.2f} - {df_forces['Backline_Left_kg'].max():.2f} kg")
    print(f"  Backline Right range: {df_forces['Backline_Right_kg'].min():.2f} - {df_forces['Backline_Right_kg'].max():.2f} kg")
    print(f"  Left-Right difference range: {df_forces['Left_Right_Difference'].min():.2f} - {df_forces['Left_Right_Difference'].max():.2f} kg")

    # Create interactive subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            'Individual Backline Forces Over Time',
            'Backline Left vs Right Comparison',
            'Left-Right Force Difference',
            'Force Distribution Analysis'
        ],
        vertical_spacing=0.08,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}]
        ]
    )

    # Plot 1: Individual backline forces over time
    fig.add_trace(
        go.Scatter(
            x=df_forces['_time'],
            y=df_forces['Backline_Left_kg'],
            mode='lines',
            name='Backline Left',
            line=dict(color='red', width=2),
            hovertemplate='<b>Backline Left</b><br>' +
                         'Time: %{x}<br>' +
                         'Force: %{y:.3f} kg<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_forces['_time'],
            y=df_forces['Backline_Right_kg'],
            mode='lines',
            name='Backline Right',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Backline Right</b><br>' +
                         'Time: %{x}<br>' +
                         'Force: %{y:.3f} kg<extra></extra>'
        ),
        row=1, col=1
    )

    # Plot 2: Left vs Right with total force
    fig.add_trace(
        go.Scatter(
            x=df_forces['Backline_Left_kg'],
            y=df_forces['Backline_Right_kg'],
            mode='markers',
            name='Left vs Right',
            marker=dict(
                color=df_forces['Time_Minutes'],
                colorscale='Viridis',
                size=4,
                colorbar=dict(title="Time (minutes)", x=1.02)
            ),
            hovertemplate='<b>Left vs Right Forces</b><br>' +
                         'Left: %{x:.3f} kg<br>' +
                         'Right: %{y:.3f} kg<br>' +
                         'Time: %{marker.color:.1f} min<extra></extra>'
        ),
        row=2, col=1
    )

    # Add diagonal line for reference (equal forces)
    min_force = min(df_forces['Backline_Left_kg'].min(), df_forces['Backline_Right_kg'].min())
    max_force = max(df_forces['Backline_Left_kg'].max(), df_forces['Backline_Right_kg'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_force, max_force],
            y=[min_force, max_force],
            mode='lines',
            name='Equal Forces Line',
            line=dict(color='gray', dash='dash', width=1),
            hoverinfo='skip'
        ),
        row=2, col=1
    )

    # Plot 3: Left-Right difference over time
    fig.add_trace(
        go.Scatter(
            x=df_forces['_time'],
            y=df_forces['Left_Right_Difference'],
            mode='lines',
            name='L-R Difference',
            line=dict(color='purple', width=2),
            hovertemplate='<b>Left-Right Difference</b><br>' +
                         'Time: %{x}<br>' +
                         'Difference: %{y:.3f} kg<extra></extra>'
        ),
        row=3, col=1
    )

    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

    # Plot 4: Force distribution histograms
    fig.add_trace(
        go.Histogram(
            x=df_forces['Backline_Left_kg'],
            name='Left Distribution',
            opacity=0.7,
            marker_color='red',
            nbinsx=30
        ),
        row=4, col=1
    )

    fig.add_trace(
        go.Histogram(
            x=df_forces['Backline_Right_kg'],
            name='Right Distribution',
            opacity=0.7,
            marker_color='blue',
            nbinsx=30
        ),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='Interactive Backline Forces Analysis - Temporal Merged Data<br>' +
                 '<sub>Natural force variation preserved through temporal interpolation</sub>',
            x=0.5,
            font=dict(size=18)
        ),
        height=1600,
        showlegend=True,
        hovermode='closest'
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Force (kg)", row=1, col=1)

    fig.update_xaxes(title_text="Backline Left Force (kg)", row=2, col=1)
    fig.update_yaxes(title_text="Backline Right Force (kg)", row=2, col=1)

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Left - Right Difference (kg)", row=3, col=1)

    fig.update_xaxes(title_text="Force (kg)", row=4, col=1)
    fig.update_yaxes(title_text="Count", row=4, col=1)

    # Save interactive plot
    output_file = "temporal_backline_forces_interactive.html"
    fig.write_html(output_file)
    print(f"\n✅ Interactive plot saved to: {output_file}")

    # Create additional detailed analysis
    create_force_variation_analysis(df_forces)

    # Show the plot
    fig.show()

def create_force_variation_analysis(df_forces):
    """Create additional analysis of force variations."""

    print(f"\n🔍 FORCE VARIATION ANALYSIS")
    print("=" * 40)

    # Calculate variation metrics
    left_std = df_forces['Backline_Left_kg'].std()
    right_std = df_forces['Backline_Right_kg'].std()
    diff_std = df_forces['Left_Right_Difference'].std()

    # Calculate rate of change (smoothness)
    left_changes = df_forces['Backline_Left_kg'].diff().abs()
    right_changes = df_forces['Backline_Right_kg'].diff().abs()

    print(f"Standard Deviations:")
    print(f"  Backline Left: {left_std:.3f} kg")
    print(f"  Backline Right: {right_std:.3f} kg")
    print(f"  Left-Right Difference: {diff_std:.3f} kg")

    print(f"\nRate of Change (Smoothness):")
    print(f"  Backline Left avg change: {left_changes.mean():.4f} kg/step")
    print(f"  Backline Right avg change: {right_changes.mean():.4f} kg/step")
    print(f"  Max Left change: {left_changes.max():.3f} kg/step")
    print(f"  Max Right change: {right_changes.max():.3f} kg/step")

    # Check for artificial repetition patterns
    left_unique = df_forces['Backline_Left_kg'].nunique()
    right_unique = df_forces['Backline_Right_kg'].nunique()
    total_records = len(df_forces)

    print(f"\nUnique Value Analysis:")
    print(f"  Total records: {total_records}")
    print(f"  Unique Left values: {left_unique} ({100*left_unique/total_records:.1f}%)")
    print(f"  Unique Right values: {right_unique} ({100*right_unique/total_records:.1f}%)")

    if left_unique / total_records > 0.8:  # More than 80% unique
        print(f"  ✅ High uniqueness indicates natural temporal variation")
    else:
        print(f"  ⚠️  Low uniqueness may indicate repetition patterns")

    # Check force balance over time
    avg_left = df_forces['Backline_Left_kg'].mean()
    avg_right = df_forces['Backline_Right_kg'].mean()
    balance_ratio = avg_left / avg_right

    print(f"\nForce Balance:")
    print(f"  Average Left: {avg_left:.3f} kg")
    print(f"  Average Right: {avg_right:.3f} kg")
    print(f"  Balance ratio: {balance_ratio:.3f}")

    if 0.8 <= balance_ratio <= 1.2:
        print(f"  ✅ Well-balanced forces")
    else:
        print(f"  ⚠️  Significant force imbalance detected")

if __name__ == "__main__":
    create_interactive_backline_timeseries()
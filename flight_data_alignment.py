import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import interpolate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

class FlightDataAligner:
    """
    Aligns flight data from Flight Segment (FS) and Ground Segment (GS) sources
    with different timestamp systems and interpolates missing force components.
    """

    def __init__(self):
        self.fs_data = None
        self.gs_data = None
        self.aligned_data = None

    def load_fs_data(self, filepath, start_time_str, end_time_str):
        """Load and filter Flight Segment data"""
        self.fs_data = pd.read_csv(filepath)

        # Convert datetime string to datetime objects for filtering
        start_time = pd.to_datetime(start_time_str)
        end_time = pd.to_datetime(end_time_str)

        # Convert datetime column to pandas datetime
        self.fs_data['datetime'] = pd.to_datetime(self.fs_data['datetime'])

        # Filter by time range
        mask = (self.fs_data['datetime'] >= start_time) & (self.fs_data['datetime'] <= end_time)
        self.fs_data = self.fs_data[mask].copy()

        # Create relative time in seconds from start
        self.fs_data['relative_time'] = (self.fs_data['datetime'] - self.fs_data['datetime'].iloc[0]).dt.total_seconds()

        print(f"FS Data loaded: {len(self.fs_data)} records")
        print(f"Time range: {self.fs_data['datetime'].iloc[0]} to {self.fs_data['datetime'].iloc[-1]}")

    def load_gs_data(self, filepath, start_time_str, end_time_str):
        """Load and filter Ground Segment data"""
        self.gs_data = pd.read_csv(filepath)

        # Convert time strings to datetime objects with UTC timezone
        start_time = pd.to_datetime(start_time_str, utc=True)
        end_time = pd.to_datetime(end_time_str, utc=True)

        # Convert _time column to pandas datetime with mixed format handling
        self.gs_data['_time'] = pd.to_datetime(self.gs_data['_time'], format='mixed')

        # Filter by time range
        mask = (self.gs_data['_time'] >= start_time) & (self.gs_data['_time'] <= end_time)
        self.gs_data = self.gs_data[mask].copy()

        # Create relative time in seconds from start
        self.gs_data['relative_time'] = (self.gs_data['_time'] - self.gs_data['_time'].iloc[0]).dt.total_seconds()

        print(f"GS Data loaded: {len(self.gs_data)} records")
        print(f"Time range: {self.gs_data['_time'].iloc[0]} to {self.gs_data['_time'].iloc[-1]}")

    def create_unified_time_vector(self, method='gs_based'):
        """
        Create a unified time reference vector

        Methods:
        - 'gs_based': Use GS timestamps as reference (higher frequency)
        - 'combined': Combine both timestamp sets
        - 'regular': Create regular intervals based on min sampling rate
        """
        if self.fs_data is None or self.gs_data is None:
            raise ValueError("Both FS and GS data must be loaded first")

        if method == 'gs_based':
            # Use GS relative time as the reference
            unified_time = self.gs_data['relative_time'].values

        elif method == 'combined':
            # Combine both time vectors and sort
            fs_times = self.fs_data['relative_time'].values
            gs_times = self.gs_data['relative_time'].values
            unified_time = np.sort(np.unique(np.concatenate([fs_times, gs_times])))

        elif method == 'regular':
            # Create regular intervals based on the minimum sampling period
            fs_dt = np.median(np.diff(self.fs_data['relative_time']))
            gs_dt = np.median(np.diff(self.gs_data['relative_time']))
            min_dt = min(fs_dt, gs_dt)

            start_time = max(self.fs_data['relative_time'].min(), self.gs_data['relative_time'].min())
            end_time = min(self.fs_data['relative_time'].max(), self.gs_data['relative_time'].max())

            unified_time = np.arange(start_time, end_time + min_dt, min_dt)

        print(f"Unified time vector created with {len(unified_time)} points")
        print(f"Sampling rate info:")
        print(f"  FS median Δt: {np.median(np.diff(self.fs_data['relative_time'])):.3f}s")
        print(f"  GS median Δt: {np.median(np.diff(self.gs_data['relative_time'])):.3f}s")

        return unified_time

    def interpolate_force_components(self, unified_time):
        """
        Interpolate FS force components to the unified time vector
        and align with GS total force data
        """
        # Force component columns from FS
        force_cols = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg', 'total_force_kg']

        # Initialize result dataframe
        aligned_data = pd.DataFrame({
            'unified_time': unified_time
        })

        # Interpolate GS timestamp (_time) to unified time with full precision
        if '_time' in self.gs_data.columns:
            # Convert GS _time to nanoseconds since epoch for full precision
            gs_timestamp_ns = self.gs_data['_time'].astype('int64')  # Keep nanosecond precision
            valid_mask = ~self.gs_data['_time'].isna()
            if valid_mask.sum() > 1:
                interp_func = interpolate.interp1d(
                    self.gs_data.loc[valid_mask, 'relative_time'],
                    gs_timestamp_ns.loc[valid_mask].astype('float64'),  # Use float64 for interpolation
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                # Interpolate and convert back to datetime with full precision
                interpolated_timestamps_ns = interp_func(unified_time)
                aligned_data['gs_time'] = pd.to_datetime(interpolated_timestamps_ns.astype('int64'), unit='ns', utc=True)
            else:
                aligned_data['gs_time'] = pd.NaT

        # Interpolate FS force components to unified time
        for col in force_cols:
            # Remove NaN values for interpolation
            valid_mask = ~self.fs_data[col].isna()
            if valid_mask.sum() > 1:  # Need at least 2 points for interpolation
                # Take absolute value of force data
                force_data = np.abs(self.fs_data.loc[valid_mask, col])
                interp_func = interpolate.interp1d(
                    self.fs_data.loc[valid_mask, 'relative_time'],
                    force_data,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                aligned_data[f'fs_{col}'] = interp_func(unified_time)
            else:
                aligned_data[f'fs_{col}'] = np.nan

        # Interpolate GS total force to unified time
        gs_force_col = 'FLIGHT_SEGMENT_loadcells_force'
        if gs_force_col in self.gs_data.columns:
            valid_mask = ~self.gs_data[gs_force_col].isna()
            if valid_mask.sum() > 1:
                # Take absolute value of GS total force
                gs_force_data = np.abs(self.gs_data.loc[valid_mask, gs_force_col])
                interp_func = interpolate.interp1d(
                    self.gs_data.loc[valid_mask, 'relative_time'],
                    gs_force_data,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                aligned_data['gs_total_force'] = interp_func(unified_time)
            else:
                aligned_data['gs_total_force'] = np.nan

        # Interpolate GS roll percentage to unified time
        gs_roll_col = 'FLIGHT_SEGMENT_roll_percentage'
        if gs_roll_col in self.gs_data.columns:
            valid_mask = ~self.gs_data[gs_roll_col].isna()
            if valid_mask.sum() > 1:
                gs_roll_data = self.gs_data.loc[valid_mask, gs_roll_col]
                interp_func = interpolate.interp1d(
                    self.gs_data.loc[valid_mask, 'relative_time'],
                    gs_roll_data,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                aligned_data['gs_roll_percentage'] = interp_func(unified_time)
            else:
                aligned_data['gs_roll_percentage'] = np.nan

        # For GS data, directly interpolate the original FS component data to GS timeline
        # This preserves the exact relationships and patterns from the FS measurements
        for component in ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']:
            gs_component = f'gs_{component}'

            # Find the component in original FS data (not the already interpolated version)
            if component in self.fs_data.columns:
                valid_mask = ~self.fs_data[component].isna()
                if valid_mask.sum() > 1:  # Need at least 2 points for interpolation
                    # Take absolute value and interpolate directly from FS measurements
                    original_fs_data = np.abs(self.fs_data.loc[valid_mask, component])
                    interp_func = interpolate.interp1d(
                        self.fs_data.loc[valid_mask, 'relative_time'],
                        original_fs_data,
                        kind='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                    aligned_data[gs_component] = interp_func(unified_time)
                else:
                    aligned_data[gs_component] = np.nan

        self.aligned_data = aligned_data
        print(f"Interpolation complete. Aligned data shape: {aligned_data.shape}")

        return aligned_data

    def analyze_alignment_quality(self):
        """Analyze the quality of the alignment between FS and GS total forces"""
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first")

        # Compare FS and GS total forces where both exist
        valid_mask = (self.aligned_data['fs_total_force_kg'].notna() &
                     self.aligned_data['gs_total_force'].notna())

        if valid_mask.sum() == 0:
            print("No overlapping valid data points found for comparison")
            return

        fs_total = self.aligned_data.loc[valid_mask, 'fs_total_force_kg']
        gs_total = self.aligned_data.loc[valid_mask, 'gs_total_force']

        # Calculate alignment metrics
        correlation = np.corrcoef(fs_total, gs_total)[0, 1]
        rmse = np.sqrt(np.mean((fs_total - gs_total)**2))
        mean_abs_error = np.mean(np.abs(fs_total - gs_total))

        print("Alignment Quality Analysis:")
        print(f"  Valid comparison points: {valid_mask.sum()}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  RMSE: {rmse:.4f} kg")
        print(f"  Mean Absolute Error: {mean_abs_error:.4f} kg")

        return {
            'correlation': correlation,
            'rmse': rmse,
            'mae': mean_abs_error,
            'valid_points': valid_mask.sum()
        }

    def plot_alignment_comparison(self, save_path=None):
        """Plot comparison of FS and GS data alignment"""
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Total force comparison
        axes[0, 0].plot(self.aligned_data['unified_time'], self.aligned_data['fs_total_force_kg'],
                       'b-', alpha=0.7, label='FS Total Force')
        axes[0, 0].plot(self.aligned_data['unified_time'], self.aligned_data['gs_total_force'],
                       'r-', alpha=0.7, label='GS Total Force')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Force (kg)')
        axes[0, 0].set_title('Total Force Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot 2: Force components from FS
        force_components = ['fs_Backline_Left_kg', 'fs_Backline_Right_kg', 'fs_5th_line_kg', 'fs_Frontline_kg']
        for i, component in enumerate(force_components):
            if component in self.aligned_data.columns:
                axes[0, 1].plot(self.aligned_data['unified_time'], self.aligned_data[component],
                               label=component.replace('fs_', '').replace('_kg', ''))
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Force (kg)')
        axes[0, 1].set_title('FS Force Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot 3: Interpolated GS components
        gs_components = ['gs_Backline_Left_kg', 'gs_Backline_Right_kg', 'gs_5th_line_kg', 'gs_Frontline_kg']
        for i, component in enumerate(gs_components):
            if component in self.aligned_data.columns:
                axes[1, 0].plot(self.aligned_data['unified_time'], self.aligned_data[component],
                               label=component.replace('gs_', '').replace('_kg', ''))
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Force (kg)')
        axes[1, 0].set_title('Interpolated GS Force Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot 4: Scatter plot of FS vs GS total forces
        valid_mask = (self.aligned_data['fs_total_force_kg'].notna() &
                     self.aligned_data['gs_total_force'].notna())
        if valid_mask.sum() > 0:
            axes[1, 1].scatter(self.aligned_data.loc[valid_mask, 'fs_total_force_kg'],
                              self.aligned_data.loc[valid_mask, 'gs_total_force'], alpha=0.6)

            # Add 1:1 line for reference
            min_val = min(self.aligned_data.loc[valid_mask, 'fs_total_force_kg'].min(),
                         self.aligned_data.loc[valid_mask, 'gs_total_force'].min())
            max_val = max(self.aligned_data.loc[valid_mask, 'fs_total_force_kg'].max(),
                         self.aligned_data.loc[valid_mask, 'gs_total_force'].max())
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 line')

        axes[1, 1].set_xlabel('FS Total Force (kg)')
        axes[1, 1].set_ylabel('GS Total Force (kg)')
        axes[1, 1].set_title('FS vs GS Total Force Correlation')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def plot_interactive_backline_comparison(self, save_path=None):
        """Create interactive time series plot comparing FS and interpolated backline forces"""
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('FS Backline Forces (Original)', 'GS Backline Forces (Interpolated)'),
            shared_xaxes=True,
            vertical_spacing=0.08
        )

        # Colors for left and right backlines
        left_color = '#1f77b4'   # Blue for left
        right_color = '#ff7f0e'  # Orange for right

        # Top panel: FS Forces (Left and Right together)
        if 'fs_Backline_Left_kg' in self.aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.aligned_data['unified_time'],
                    y=self.aligned_data['fs_Backline_Left_kg'],
                    mode='lines',
                    name='FS Left',
                    line=dict(color=left_color, width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )

        if 'fs_Backline_Right_kg' in self.aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.aligned_data['unified_time'],
                    y=self.aligned_data['fs_Backline_Right_kg'],
                    mode='lines',
                    name='FS Right',
                    line=dict(color=right_color, width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )

        # Bottom panel: GS Forces (Left and Right together)
        if 'gs_Backline_Left_kg' in self.aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.aligned_data['unified_time'],
                    y=self.aligned_data['gs_Backline_Left_kg'],
                    mode='lines',
                    name='GS Left',
                    line=dict(color=left_color, width=2, dash='dot'),
                    opacity=0.7
                ),
                row=2, col=1
            )

        if 'gs_Backline_Right_kg' in self.aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.aligned_data['unified_time'],
                    y=self.aligned_data['gs_Backline_Right_kg'],
                    mode='lines',
                    name='GS Right',
                    line=dict(color=right_color, width=2, dash='dot'),
                    opacity=0.7
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Interactive Backline Forces Comparison: FS Original vs GS Interpolated',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=800,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Update x-axis labels
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)

        # Update y-axis labels
        fig.update_yaxes(title_text="Force (kg)", row=1, col=1)
        fig.update_yaxes(title_text="Force (kg)", row=2, col=1)

        # Add custom hover information
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                         "Time: %{x:.2f}s<br>" +
                         "Force: %{y:.3f} kg<br>" +
                         "<extra></extra>"
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to: {save_path}")

        # Show the plot
        fig.show()

        return fig

    def plot_side_by_side_comparison(self, save_path=None):
        """Create interactive plot comparing left FS vs left GS in top panel, right FS vs right GS in bottom panel"""
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Left Backline Forces: FS Original vs GS Interpolated', 'Right Backline Forces: FS Original vs GS Interpolated'),
            shared_xaxes=True,
            vertical_spacing=0.08
        )

        # Colors for FS vs GS
        fs_color = '#1f77b4'  # Blue for FS
        gs_color = '#ff7f0e'  # Orange for GS

        # Top panel: Left forces comparison
        if 'fs_Backline_Left_kg' in self.aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.aligned_data['unified_time'],
                    y=self.aligned_data['fs_Backline_Left_kg'],
                    mode='lines',
                    name='FS Left (Original)',
                    line=dict(color=fs_color, width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )

        if 'gs_Backline_Left_kg' in self.aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.aligned_data['unified_time'],
                    y=self.aligned_data['gs_Backline_Left_kg'],
                    mode='lines',
                    name='GS Left (Interpolated)',
                    line=dict(color=gs_color, width=1.5, dash='dot'),
                    opacity=0.7
                ),
                row=1, col=1
            )

        # Bottom panel: Right forces comparison
        if 'fs_Backline_Right_kg' in self.aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.aligned_data['unified_time'],
                    y=self.aligned_data['fs_Backline_Right_kg'],
                    mode='lines',
                    name='FS Right (Original)',
                    line=dict(color=fs_color, width=2),
                    opacity=0.8,
                    showlegend=False  # Don't duplicate legend
                ),
                row=2, col=1
            )

        if 'gs_Backline_Right_kg' in self.aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.aligned_data['unified_time'],
                    y=self.aligned_data['gs_Backline_Right_kg'],
                    mode='lines',
                    name='GS Right (Interpolated)',
                    line=dict(color=gs_color, width=1.5, dash='dot'),
                    opacity=0.7,
                    showlegend=False  # Don't duplicate legend
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Side-by-Side Backline Forces Comparison: Original vs Interpolated',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=800,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Update x-axis labels
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)

        # Update y-axis labels
        fig.update_yaxes(title_text="Force (kg)", row=1, col=1)
        fig.update_yaxes(title_text="Force (kg)", row=2, col=1)

        # Add custom hover information
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                         "Time: %{x:.2f}s<br>" +
                         "Force: %{y:.3f} kg<br>" +
                         "<extra></extra>"
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Side-by-side interactive plot saved to: {save_path}")

        # Show the plot
        fig.show()

        return fig

    def create_detailed_comparison_stats(self):
        """Create detailed statistics comparing FS and interpolated GS backline forces"""
        if self.aligned_data is None:
            raise ValueError("Data must be aligned first")

        stats_dict = {}

        # Analyze each backline component
        for side in ['Left', 'Right']:
            fs_col = f'fs_Backline_{side}_kg'
            gs_col = f'gs_Backline_{side}_kg'

            if fs_col in self.aligned_data.columns and gs_col in self.aligned_data.columns:
                # Get valid comparison points
                valid_mask = (self.aligned_data[fs_col].notna() &
                             self.aligned_data[gs_col].notna())

                if valid_mask.sum() > 0:
                    fs_data = self.aligned_data.loc[valid_mask, fs_col]
                    gs_data = self.aligned_data.loc[valid_mask, gs_col]

                    # Calculate statistics
                    correlation = np.corrcoef(fs_data, gs_data)[0, 1]
                    rmse = np.sqrt(np.mean((fs_data - gs_data)**2))
                    mae = np.mean(np.abs(fs_data - gs_data))

                    # Peak preservation analysis
                    fs_peaks = fs_data.quantile([0.9, 0.95, 0.99])
                    gs_peaks = gs_data.quantile([0.9, 0.95, 0.99])

                    stats_dict[f'Backline_{side}'] = {
                        'correlation': correlation,
                        'rmse': rmse,
                        'mae': mae,
                        'valid_points': valid_mask.sum(),
                        'fs_mean': fs_data.mean(),
                        'gs_mean': gs_data.mean(),
                        'fs_std': fs_data.std(),
                        'gs_std': gs_data.std(),
                        'fs_90th_percentile': fs_peaks[0.9],
                        'gs_90th_percentile': gs_peaks[0.9],
                        'fs_95th_percentile': fs_peaks[0.95],
                        'gs_95th_percentile': gs_peaks[0.95],
                        'fs_99th_percentile': fs_peaks[0.99],
                        'gs_99th_percentile': gs_peaks[0.99]
                    }

        # Print detailed comparison
        print("\n" + "="*60)
        print("DETAILED BACKLINE FORCES COMPARISON")
        print("="*60)

        for side, stats in stats_dict.items():
            print(f"\n{side.upper()}:")
            print(f"  Correlation: {stats['correlation']:.4f}")
            print(f"  RMSE: {stats['rmse']:.4f} kg")
            print(f"  MAE: {stats['mae']:.4f} kg")
            print(f"  Valid points: {stats['valid_points']:,}")
            print(f"  Mean - FS: {stats['fs_mean']:.3f} kg, GS: {stats['gs_mean']:.3f} kg")
            print(f"  Std Dev - FS: {stats['fs_std']:.3f} kg, GS: {stats['gs_std']:.3f} kg")
            print(f"  90th %ile - FS: {stats['fs_90th_percentile']:.3f} kg, GS: {stats['gs_90th_percentile']:.3f} kg")
            print(f"  95th %ile - FS: {stats['fs_95th_percentile']:.3f} kg, GS: {stats['gs_95th_percentile']:.3f} kg")
            print(f"  99th %ile - FS: {stats['fs_99th_percentile']:.3f} kg, GS: {stats['gs_99th_percentile']:.3f} kg")

        return stats_dict

def main():
    """Example usage with provided flight data"""

    # Initialize aligner
    aligner = FlightDataAligner()

    # Load data
    fs_file = '/Users/baharakqaderi/field-data-pipeline/data/FORCES/loadcell_readings_20250715_183235.csv'
    fs_start = '2025-07-15 18:50:00'
    fs_end = '2025-07-15 19:21:00'

    gs_file = '/Users/baharakqaderi/field-data-pipeline/data/INFLUX/FLIGHT_SEGMENT_data_2025-07-16.csv'
    gs_start = '2025-07-16 11:47:46'
    gs_end = '2025-07-16 12:18:40'

    print("Loading Flight Segment data...")
    aligner.load_fs_data(fs_file, fs_start, fs_end)

    print("\nLoading Ground Segment data...")
    aligner.load_gs_data(gs_file, gs_start, gs_end)

    print("\nCreating unified time vector...")
    unified_time = aligner.create_unified_time_vector(method='gs_based')

    print("\nInterpolating force components...")
    aligned_data = aligner.interpolate_force_components(unified_time)

    print("\nAnalyzing alignment quality...")
    quality_metrics = aligner.analyze_alignment_quality()

    print("\nGenerating comparison plots...")
    aligner.plot_alignment_comparison('flight_data_alignment_comparison.png')

    print("\nCreating interactive backline forces comparison...")
    aligner.plot_interactive_backline_comparison('backline_forces_interactive_comparison.html')

    print("\nAnalyzing detailed backline forces statistics...")
    backline_stats = aligner.create_detailed_comparison_stats()

    # Save aligned data
    output_file = 'aligned_flight_data.csv'
    aligned_data.to_csv(output_file, index=False)
    print(f"\nAligned data saved to: {output_file}")

    return aligner, aligned_data, backline_stats

if __name__ == "__main__":
    aligner, aligned_data, backline_stats = main()
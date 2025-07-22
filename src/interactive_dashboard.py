"""
Interactive Force Data Dashboard

This module creates a comprehensive interactive dashboard using Plotly Dash
for exploring force data with time series plots, component analysis, and filtering.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class ForceDashboard:
    """
    Interactive dashboard for force data analysis.
    """
    
    def __init__(self, data_dir: Path = Path("outputs")):
        self.data_dir = Path(data_dir)
        self.force_components = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
    def load_processed_data(self, date: str = None) -> pd.DataFrame:
        """Load processed data for analysis."""
        if date:
            filename = f"processed_merged_flight_data_{date}.csv"
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                st.error(f"File not found: {filename}")
                return pd.DataFrame()
                
            data = pd.read_csv(filepath)
        else:
            # Load all processed files
            files = list(self.data_dir.glob("processed_merged_flight_data_*.csv"))
            if not files:
                st.error("No processed data files found")
                return pd.DataFrame()
                
            all_data = []
            for file in files:
                df = pd.read_csv(file)
                df['source_date'] = file.stem.split('_')[-1]  # Extract date from filename
                all_data.append(df)
                
            data = pd.concat(all_data, ignore_index=True)
            
        # Convert timestamp and filter matched data (handle mixed formats)
        data['_time'] = pd.to_datetime(data['_time'], format='mixed')
        data = data[data['forces_matched'] == True]
        
        return data
        
    def create_time_series_plot(self, data: pd.DataFrame, selected_components: list = None):
        """Create interactive time series plot of force components."""
        if selected_components is None:
            selected_components = self.force_components
            
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Individual Force Components', 'Total Force'),
            row_heights=[0.7, 0.3]
        )
        
        # Plot individual components
        for i, component in enumerate(selected_components):
            if component in data.columns:
                component_data = data.dropna(subset=[component])
                if len(component_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=component_data['_time'],
                            y=component_data[component],
                            name=component.replace('_kg', ''),
                            line=dict(color=self.color_palette[i % len(self.color_palette)]),
                            hovertemplate=f'<b>{component.replace("_kg", "")}</b><br>' +
                                        'Time: %{x}<br>' +
                                        'Force: %{y:.3f} kg<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # Plot total force
        fig.add_trace(
            go.Scatter(
                x=data['_time'],
                y=data['FLIGHT_SEGMENT_loadcells_force'],
                name='Total Force',
                line=dict(color='black', width=2),
                hovertemplate='<b>Total Force</b><br>' +
                            'Time: %{x}<br>' +
                            'Force: %{y:.3f} kg<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Force Components Time Series Analysis",
            height=800,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Force (kg)", row=1, col=1)
        fig.update_yaxes(title_text="Total Force (kg)", row=2, col=1)
        
        return fig
        
    def create_correlation_matrix(self, data: pd.DataFrame):
        """Create correlation matrix heatmap."""
        # Select numerical columns for correlation
        corr_cols = self.force_components + ['FLIGHT_SEGMENT_loadcells_force']
        available_cols = [col for col in corr_cols if col in data.columns]
        
        if len(available_cols) < 2:
            st.warning("Not enough data for correlation analysis")
            return None
            
        corr_data = data[available_cols].corr()
        
        fig = px.imshow(
            corr_data,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Force Components Correlation Matrix'
        )
        
        fig.update_layout(
            width=600,
            height=500,
            title_x=0.5
        )
        
        return fig
        
    def create_force_distribution(self, data: pd.DataFrame, component: str):
        """Create distribution plot for a force component."""
        if component not in data.columns:
            return None
            
        component_data = data.dropna(subset=[component])
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribution', 'Time Series'),
            column_widths=[0.3, 0.7]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                y=component_data[component],
                name='Distribution',
                nbinsy=50,
                marker_color=self.color_palette[0],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Time series
        fig.add_trace(
            go.Scatter(
                x=component_data['_time'],
                y=component_data[component],
                mode='markers',
                name='Time Series',
                marker=dict(
                    color=component_data[component],
                    colorscale='Viridis',
                    colorbar=dict(title=f'{component} (kg)'),
                    size=3,
                    opacity=0.6
                ),
                hovertemplate='Time: %{x}<br>Force: %{y:.3f} kg<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'{component.replace("_kg", "")} Analysis',
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Force (kg)", row=1, col=1)
        fig.update_yaxes(title_text="Force (kg)", row=1, col=2)
        
        return fig
        
    def calculate_statistics(self, data: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        stats = {}
        
        # Overall statistics
        stats['total_records'] = len(data)
        stats['time_range'] = {
            'start': data['_time'].min(),
            'end': data['_time'].max(),
            'duration': data['_time'].max() - data['_time'].min()
        }
        
        # Force component statistics
        stats['components'] = {}
        for component in self.force_components:
            if component in data.columns:
                component_data = data.dropna(subset=[component])
                if len(component_data) > 0:
                    stats['components'][component] = {
                        'count': len(component_data),
                        'mean': component_data[component].mean(),
                        'std': component_data[component].std(),
                        'min': component_data[component].min(),
                        'max': component_data[component].max(),
                        'median': component_data[component].median()
                    }
        
        # Total force statistics
        if 'FLIGHT_SEGMENT_loadcells_force' in data.columns:
            total_force = data['FLIGHT_SEGMENT_loadcells_force']
            stats['total_force'] = {
                'mean': total_force.mean(),
                'std': total_force.std(),
                'min': total_force.min(),
                'max': total_force.max(),
                'median': total_force.median()
            }
        
        return stats


def create_streamlit_dashboard():
    """Create the main Streamlit dashboard."""
    
    st.set_page_config(
        page_title="Force Data Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚡ Interactive Force Data Dashboard")
    st.markdown("### Analyzing Flight Segment Load Cell Force Data")
    
    # Initialize dashboard
    dashboard = ForceDashboard()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Date selection
    available_dates = []
    for file in Path("outputs").glob("processed_merged_flight_data_*.csv"):
        date = file.stem.split('_')[-1]
        available_dates.append(date)
    
    if not available_dates:
        st.error("No processed data files found. Please run the data processor first.")
        st.stop()
        
    selected_date = st.sidebar.selectbox(
        "Select Date",
        ["All Dates"] + sorted(available_dates)
    )
    
    # Load data
    with st.spinner("Loading data..."):
        if selected_date == "All Dates":
            data = dashboard.load_processed_data()
        else:
            data = dashboard.load_processed_data(selected_date)
    
    if data.empty:
        st.error("No data available for the selected date.")
        st.stop()
    
    # Component selection
    available_components = [comp for comp in dashboard.force_components if comp in data.columns]
    selected_components = st.sidebar.multiselect(
        "Select Force Components",
        available_components,
        default=available_components
    )
    
    # Time range filter
    if len(data) > 0:
        min_time = data['_time'].min().to_pydatetime()
        max_time = data['_time'].max().to_pydatetime()
        
        time_range = st.sidebar.slider(
            "Select Time Range",
            min_value=min_time,
            max_value=max_time,
            value=(min_time, max_time),
            format="MM/DD HH:mm"
        )
        
        # Filter data by time range
        data = data[
            (data['_time'] >= pd.Timestamp(time_range[0])) &
            (data['_time'] <= pd.Timestamp(time_range[1]))
        ]
    
    # Main dashboard content
    if len(data) == 0:
        st.warning("No data available for the selected filters.")
        st.stop()
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        duration = data['_time'].max() - data['_time'].min()
        hours = duration.total_seconds() / 3600
        st.metric("Duration", f"{hours:.1f} hours")
    
    with col3:
        if 'FLIGHT_SEGMENT_loadcells_force' in data.columns:
            avg_force = data['FLIGHT_SEGMENT_loadcells_force'].mean()
            st.metric("Avg Total Force", f"{avg_force:.2f} kg")
    
    with col4:
        match_rate = (data['forces_matched'].sum() / len(data)) * 100 if 'forces_matched' in data.columns else 100
        st.metric("Data Quality", f"{match_rate:.1f}%")
    
    # Time series plot
    st.subheader("📈 Force Components Time Series")
    if selected_components:
        fig_timeseries = dashboard.create_time_series_plot(data, selected_components)
        st.plotly_chart(fig_timeseries, use_container_width=True)
    else:
        st.warning("Please select at least one force component to display.")
    
    # Two-column layout for additional analyses
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔗 Correlation Analysis")
        fig_corr = dashboard.create_correlation_matrix(data)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.subheader("📊 Component Distribution")
        selected_component = st.selectbox(
            "Select Component for Distribution Analysis",
            available_components,
            key="dist_component"
        )
        
        if selected_component:
            fig_dist = dashboard.create_force_distribution(data, selected_component)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # Detailed statistics
    st.subheader("📋 Detailed Statistics")
    stats = dashboard.calculate_statistics(data)
    
    # Component statistics table
    if stats['components']:
        st.write("**Force Components Statistics:**")
        
        stats_data = []
        for component, comp_stats in stats['components'].items():
            stats_data.append({
                'Component': component.replace('_kg', ''),
                'Count': f"{comp_stats['count']:,}",
                'Mean': f"{comp_stats['mean']:.3f}",
                'Std Dev': f"{comp_stats['std']:.3f}",
                'Min': f"{comp_stats['min']:.3f}",
                'Max': f"{comp_stats['max']:.3f}",
                'Median': f"{comp_stats['median']:.3f}"
            })
        
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    # Download section
    st.subheader("💾 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download filtered data
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"filtered_force_data_{selected_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download statistics
        stats_df = pd.DataFrame(stats_data) if 'stats_data' in locals() else pd.DataFrame()
        if not stats_df.empty:
            stats_csv = stats_df.to_csv(index=False)
            st.download_button(
                label="Download Statistics (CSV)",
                data=stats_csv,
                file_name=f"force_statistics_{selected_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def main():
    """Main function to run the dashboard."""
    create_streamlit_dashboard()


if __name__ == "__main__":
    main() 
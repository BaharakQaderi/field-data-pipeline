# Field Data Pipeline

A comprehensive pipeline for analyzing field data with focus on synchronizing force measurements from multiple sources using accurate InfluxDB timestamps.

## 🎯 Problem Solved

This pipeline addresses the challenge of aligning time-series data from different sources with unreliable timestamps. Specifically:

- **InfluxDB data**: Contains accurate, synchronized timestamps and total force measurements (`FLIGHT_SEGMENT_loadcells_force`)
- **FORCES data**: Contains individual force components but unreliable timestamps (Raspberry Pi with unsynchronized clock)

**Solution**: Use total force values as matching keys to align individual force components with accurate timestamps.

## ✅ Current Status

**Data Processing Complete:**
- ✅ July 16, 2025: **85.5% match rate**, perfect correlation (-0.999)
- ✅ July 17, 2025: **86.9% match rate**, perfect correlation (-1.000)
- ✅ Zero mean force difference - perfect alignment precision
- ✅ 150MB+ of merged, validated data ready for analysis

## 🚀 Features

### Data Merging & Alignment
- Smart force-based alignment algorithm
- Handles mixed timestamp formats
- Quality tracking with match rates and force differences
- Comprehensive validation with correlation analysis

### Force Components
- `Backline_Left_kg` - Left backline force measurement
- `Backline_Right_kg` - Right backline force measurement  
- `5th_line_kg` - Fifth line force measurement
- `Frontline_kg` - Front line force measurement

### Validation & Quality Assurance
- Time series comparison plots
- Individual force component visualization
- Match quality histograms and statistics
- Correlation analysis between data sources

## 📁 Project Structure

```
field-data-pipeline/
├── main.py                              # Main entry point with CLI
├── src/
│   ├── data_merger.py                   # Core data alignment logic
│   └── force_validation_plots.py       # Validation visualization
├── data/                                # Raw data files
│   ├── INFLUX/                         # InfluxDB exports (accurate timestamps)
│   ├── FORCES/                         # Load cell readings (individual components)
│   └── IPOK/                           # Additional sensor data
├── outputs/                            # Processed results
│   ├── merged_flight_data_*.csv        # Aligned datasets
│   └── force_validation_*.png          # Validation plots
├── notebooks/                          # Jupyter analysis notebooks
│   └── EDA.ipynb                       # Exploratory data analysis
└── tests/                              # Test files
```

## ⚙️ Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BaharakQaderi/field-data-pipeline.git
   cd field-data-pipeline
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Install Jupyter kernel for the project:
   ```bash
   uv run python -m ipykernel install --user --name field-data-pipeline --display-name "Field Data Pipeline"
   ```

## 🔧 Usage

### Data Processing

```bash
# Process all available dates
uv run python main.py --merge

# Process specific date
uv run python main.py --merge --date 2025-07-16

# Start Jupyter for analysis
uv run python main.py --notebook
```

### Direct Module Usage

```bash
# Run data merger directly
uv run python src/data_merger.py

# Create validation plots
uv run python src/force_validation_plots.py
```

### In Jupyter Notebook

```python
# Load merged data
import pandas as pd
df = pd.read_csv('outputs/merged_flight_data_2025-07-16.csv')

# Analyze force components
force_components = ['Backline_Left_kg', 'Backline_Right_kg', '5th_line_kg', 'Frontline_kg']
matched_data = df[df['forces_matched'] == True]

print(f"Dataset: {len(df)} total records")
print(f"Matched: {len(matched_data)} records ({len(matched_data)/len(df)*100:.1f}%)")
```

## 📊 Results Summary

### July 16, 2025
- **Total records:** 306,872
- **Successfully matched:** 262,296 (85.5%)
- **Correlation:** -0.999 (nearly perfect)
- **Mean force difference:** 0.0000 kg
- **Output:** `outputs/merged_flight_data_2025-07-16.csv` (81MB)

### July 17, 2025  
- **Total records:** 244,490
- **Successfully matched:** 212,478 (86.9%)
- **Correlation:** -1.000 (mathematically perfect)
- **Mean force difference:** 0.0000 kg
- **Output:** `outputs/merged_flight_data_2025-07-17.csv` (69MB)

## 🔍 Data Quality Features

### Alignment Algorithm
- Uses `total_force_kg` ↔ `FLIGHT_SEGMENT_loadcells_force` matching
- Configurable tolerance (default: 0.1 kg)
- Best-match selection for overlapping records
- Quality indicators: `forces_matched`, `force_difference`

### Validation Plots Include
- Full day force comparison (original vs matched)
- Individual force component time series
- Match quality distribution histograms
- Correlation scatter plots with perfect-fit lines
- Detailed time window analysis
- Comprehensive summary statistics

## 🛠️ Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Visualization
- **plotly**: Interactive plots (ready for future use)
- **jupyter**: Interactive analysis environment

### File Mapping
- **July 16**: First 3 FORCES files (`loadcell_readings_20250715_*.csv`)
- **July 17**: Last 2 FORCES files (`loadcell_readings_*.csv`)

### Timestamp Handling
- InfluxDB: ISO format with timezone (`2025-07-16T09:05:30.307293+00:00`)
- FORCES: Mixed formats handled with pandas `format='mixed'`

## 🔬 Next Steps

With perfect data alignment achieved, the pipeline is ready for:

- [ ] Advanced force component analysis
- [ ] Time-based pattern detection
- [ ] Correlation analysis with flight parameters
- [ ] Anomaly detection in force measurements
- [ ] Integration with additional sensor data (IPOK directory)
- [ ] Real-time processing capabilities

## 📈 Analysis Ready

The merged datasets provide:
- **Accurate timestamps** from InfluxDB
- **Individual force components** from load cells
- **Complete flight parameters** from all sensors
- **Quality indicators** for data validation
- **150MB+ of clean, aligned data** ready for advanced analysis

## 🤝 Contributing

This project uses:
- **UV** for dependency management
- **Black** for code formatting  
- **GitHub** for version control
- **Jupyter** for interactive analysis

## 📄 License

MIT License

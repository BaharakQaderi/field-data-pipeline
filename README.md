# Field Data Pipeline

A comprehensive pipeline for analyzing field data with data processing, visualization, and reporting capabilities.

## Features

- Data ingestion from various sources (CSV, Excel, JSON)
- Data cleaning and preprocessing
- Statistical analysis and visualization
- Interactive dashboards with Plotly
- Automated reporting
- Jupyter notebook integration for exploratory analysis

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd field-data-pipeline
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

### Development Setup

Install development dependencies:
```bash
uv sync --extra dev
```

## Usage

Run the main pipeline:
```bash
uv run main.py
```

Start Jupyter for exploratory analysis:
```bash
uv run jupyter notebook
```

## Project Structure

```
field-data-pipeline/
├── main.py              # Main pipeline entry point
├── data/                # Data directory (to be created)
├── notebooks/           # Jupyter notebooks (to be created)
├── src/                 # Source code modules (to be created)
├── tests/               # Test files (to be created)
└── outputs/             # Analysis outputs (to be created)
```

## Contributing

1. Install development dependencies: `uv sync --extra dev`
2. Run tests: `uv run pytest`
3. Format code: `uv run black .`
4. Check types: `uv run mypy .`

## License

MIT License

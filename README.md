# Field Data Pipeline

A field data pipeline for analysis.

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

## Usage

Run the main pipeline:
```bash
uv run main.py
```

## Project Structure

```
field-data-pipeline/
├── main.py              # Main pipeline entry point
├── data/                # Data directory
├── notebooks/           # Jupyter notebooks  
├── src/                 # Source code modules
├── tests/               # Test files
└── outputs/             # Analysis outputs
```

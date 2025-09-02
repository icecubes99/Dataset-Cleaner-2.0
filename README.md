# Dataset Cleaner

A comprehensive Python tool for cleaning and preprocessing Excel datasets.

## Setup

### 1. Install Python
First, you need to install Python on your system:
- Download Python from [python.org](https://www.python.org/downloads/)
- During installation, make sure to check "Add Python to PATH"
- Restart your command prompt/PowerShell after installation

### 2. Create Virtual Environment
After installing Python, run these commands in PowerShell:

```powershell
# Navigate to your project directory
cd "d:\School\Cleaner"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

If you get an execution policy error when activating the virtual environment, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Usage

### Basic Usage
```python
from dataset_cleaner import DatasetCleaner

# Initialize the cleaner
cleaner = DatasetCleaner("RawThesisData.xlsx")

# Load and clean your data
cleaner.load_data()
cleaner.get_data_overview()
cleaner.remove_duplicates()
cleaner.handle_missing_values()
cleaner.save_cleaned_data()
```

### Quick Start
Simply run the main script:
```powershell
python dataset_cleaner.py
```

## Features

- **Data Loading**: Load Excel files with multiple sheet support
- **Data Overview**: Comprehensive dataset analysis including:
  - Shape and memory usage
  - Column types and null value analysis
  - Numerical and categorical summaries
- **Duplicate Detection**: Find and remove duplicate rows
- **Missing Value Handling**: Multiple strategies:
  - Auto (intelligent handling based on data type)
  - Drop rows with missing values
  - Fill with mean/median/mode/zero
- **Outlier Detection**: Using IQR or Z-score methods
- **Cleaning Reports**: Track all cleaning actions performed
- **Export Options**: Save cleaned data as Excel or CSV

## File Structure
```
d:\School\Cleaner\
├── RawThesisData.xlsx      # Your original dataset
├── dataset_cleaner.py      # Main cleaning script
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── venv/                  # Virtual environment (after setup)
```

## Dependencies

- pandas: Data manipulation and analysis
- openpyxl: Excel file handling
- numpy: Numerical operations
- matplotlib: Data visualization
- seaborn: Statistical plotting
- jupyter: Interactive notebooks
- scikit-learn: Machine learning utilities

## Troubleshooting

### Python not found
If you get "python is not recognized":
1. Install Python from python.org
2. During installation, check "Add Python to PATH"
3. Restart PowerShell
4. Try `py` instead of `python`

### Virtual environment activation issues
If `.\venv\Scripts\Activate.ps1` fails:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Package installation issues
If pip install fails:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

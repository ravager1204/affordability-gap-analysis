# Malaysia Affordability Gap Analysis

This project analyzes the affordability gap across Malaysian states by comparing average monthly household expenditure against median monthly household income.

## Project Structure
```
affordability-gap-analysis/
├── data/                   # Directory for input data files
├── src/                    # Source code directory
│   └── analyze.py         # Main analysis script
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Setup
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your data files in the `data/` directory:
   - Mean Monthly Household Expenditure by State (CSV/Excel)
   - Median Monthly Household Income by State (CSV/Excel)

## Usage
Run the analysis script:
```bash
python src/analyze.py
```

## Data Sources
- OpenDOSM (https://open.dosm.gov.my)
  - Mean Monthly Household Expenditure by State
  - Median Monthly Household Income by State

## Output
The analysis will generate:
- Affordability ratio calculations
- Visualizations of the affordability gap across states
- Identification of states with high expenditure-to-income ratios 
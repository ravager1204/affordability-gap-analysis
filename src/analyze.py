import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from visualizations import (
    plot_affordability_ratio,
    plot_income_expenditure_scatter,
    plot_income_expenditure_gap,
    plot_financial_pressure_categories
)

# Set this to True to load expenditure data from the online Parquet file
USE_PARQUET_URL = True
PARQUET_URL = 'https://storage.dosm.gov.my/hies/hies_state.parquet'

# Set this to True to load income data from the online Parquet file
USE_INCOME_PARQUET_URL = True
INCOME_PARQUET_URL = 'https://storage.dosm.gov.my/hies/hh_income_state.parquet'

# Set seaborn style
sns.set_style('whitegrid')

def load_data(expenditure_file, income_file):
    """
    Load and validate the expenditure and income datasets.
    
    Args:
        expenditure_file (str): Path to expenditure data file
        income_file (str): Path to income data file
    
    Returns:
        tuple: (expenditure_df, income_df)
    """
    try:
        if USE_PARQUET_URL:
            print(f"Loading expenditure data from Parquet URL: {PARQUET_URL}")
            expenditure_df = pd.read_parquet(PARQUET_URL)
            if 'date' in expenditure_df.columns:
                expenditure_df['date'] = pd.to_datetime(expenditure_df['date'])
        else:
            expenditure_df = pd.read_csv(expenditure_file) if expenditure_file.endswith('.csv') else pd.read_excel(expenditure_file)
        
        if USE_INCOME_PARQUET_URL:
            print(f"Loading income data from Parquet URL: {INCOME_PARQUET_URL}")
            income_df = pd.read_parquet(INCOME_PARQUET_URL)
            if 'date' in income_df.columns:
                income_df['date'] = pd.to_datetime(income_df['date'])
        else:
            income_df = pd.read_csv(income_file) if income_file.endswith('.csv') else pd.read_excel(income_file)
        
        print("Data loaded successfully!")
        print("\nExpenditure data preview:")
        print(expenditure_df.head())
        print("\nIncome data preview:")
        print(income_df.head())
        
        return expenditure_df, income_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def clean_data(expenditure_df, income_df):
    """
    Clean and normalize the datasets.
    
    Args:
        expenditure_df (pd.DataFrame): Expenditure data
        income_df (pd.DataFrame): Income data
    
    Returns:
        tuple: (cleaned_expenditure_df, cleaned_income_df)
    """
    # Create copies to avoid SettingWithCopyWarning
    expenditure_df = expenditure_df.copy()
    income_df = income_df.copy()
    
    # Remove any rows with missing values
    expenditure_df = expenditure_df.dropna()
    income_df = income_df.dropna()
    
    # Ensure state names are consistent
    expenditure_df['state'] = expenditure_df['state'].str.strip().str.title()
    income_df['state'] = income_df['state'].str.strip().str.title()
    
    # For expenditure data, we want the latest year's data
    latest_date = expenditure_df['date'].max()
    expenditure_df = expenditure_df[expenditure_df['date'] == latest_date]
    
    # For income data, we want the latest year's data
    latest_date = income_df['date'].max()
    income_df = income_df[income_df['date'] == latest_date]
    
    # Select and rename columns for analysis
    expenditure_df = expenditure_df[['state', 'expenditure_mean']].rename(columns={
        'state': 'State',
        'expenditure_mean': 'Expenditure'
    })
    
    income_df = income_df[['state', 'income_median']].rename(columns={
        'state': 'State',
        'income_median': 'Income'
    })
    
    return expenditure_df, income_df

def calculate_affordability(expenditure_df, income_df):
    """
    Calculate the affordability ratio for each state.
    
    Args:
        expenditure_df (pd.DataFrame): Cleaned expenditure data
        income_df (pd.DataFrame): Cleaned income data
    
    Returns:
        pd.DataFrame: DataFrame with affordability ratios
    """
    # Merge the datasets on state
    merged_df = pd.merge(expenditure_df, income_df, on='State', suffixes=('_exp', '_inc'))
    
    # Calculate affordability ratio
    merged_df['Affordability_Ratio'] = merged_df['Expenditure'] / merged_df['Income']
    
    # Sort by affordability ratio (descending)
    merged_df = merged_df.sort_values('Affordability_Ratio', ascending=False)
    
    return merged_df

def analyze_financial_pressure(affordability_df):
    """
    Categorize states based on their financial pressure and calculate gaps.
    
    Args:
        affordability_df (pd.DataFrame): DataFrame with affordability ratios
    
    Returns:
        pd.DataFrame: Enhanced analysis results
    """
    # Calculate income-expenditure gap
    affordability_df['Income_Expenditure_Gap'] = affordability_df['Income'] - affordability_df['Expenditure']
    affordability_df['Gap_Percentage'] = (affordability_df['Income_Expenditure_Gap'] / affordability_df['Income']) * 100
    
    # Categorize states based on affordability ratio
    conditions = [
        (affordability_df['Affordability_Ratio'] >= 0.90),
        (affordability_df['Affordability_Ratio'] >= 0.80) & (affordability_df['Affordability_Ratio'] < 0.90),
        (affordability_df['Affordability_Ratio'] >= 0.70) & (affordability_df['Affordability_Ratio'] < 0.80),
        (affordability_df['Affordability_Ratio'] < 0.70)
    ]
    choices = ['High Pressure', 'Moderate-High Pressure', 'Moderate-Low Pressure', 'Low Pressure']
    affordability_df['Financial_Pressure'] = pd.cut(affordability_df['Affordability_Ratio'], 
                                                   bins=[0, 0.70, 0.80, 0.90, 1.0],
                                                   labels=choices)
    
    return affordability_df

def visualize_results(affordability_df):
    """
    Create enhanced visualizations of the affordability gap.
    
    Args:
        affordability_df (pd.DataFrame): DataFrame with affordability ratios
    """
    # Create visualizations directory if it doesn't exist
    vis_dir = Path('visualizations')
    vis_dir.mkdir(exist_ok=True)
    
    # Generate all visualizations
    plot_affordability_ratio(affordability_df)
    plot_income_expenditure_scatter(affordability_df)
    plot_income_expenditure_gap(affordability_df)
    plot_financial_pressure_categories(affordability_df)
    
    print("\nVisualizations saved in the 'visualizations' directory:")
    print("1. affordability_ratio.png")
    print("2. income_expenditure_scatter.png")
    print("3. income_expenditure_gap.png")
    print("4. financial_pressure_categories.png")

def print_analysis_summary(affordability_df):
    """
    Print a detailed analysis summary.
    
    Args:
        affordability_df (pd.DataFrame): DataFrame with affordability ratios
    """
    print("\n=== Detailed Analysis Summary ===")
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"Average Affordability Ratio: {affordability_df['Affordability_Ratio'].mean():.2f}")
    print(f"Median Affordability Ratio: {affordability_df['Affordability_Ratio'].median():.2f}")
    print(f"Average Monthly Gap: RM{affordability_df['Income_Expenditure_Gap'].mean():.2f}")
    
    # Financial pressure categories
    print("\nStates by Financial Pressure Category:")
    for category in affordability_df['Financial_Pressure'].unique():
        states = affordability_df[affordability_df['Financial_Pressure'] == category]['State'].tolist()
        print(f"\n{category}:")
        print(", ".join(states))
    
    # Top and bottom performers
    print("\nTop 3 States with Highest Disposable Income:")
    top_states = affordability_df.nsmallest(3, 'Affordability_Ratio')
    for _, row in top_states.iterrows():
        print(f"{row['State']}: {row['Gap_Percentage']:.1f}% disposable income")
    
    print("\nTop 3 States with Highest Financial Pressure:")
    pressure_states = affordability_df.nlargest(3, 'Affordability_Ratio')
    for _, row in pressure_states.iterrows():
        print(f"{row['State']}: {row['Gap_Percentage']:.1f}% disposable income")

def main():
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Check for data files only if not using Parquet URLs
    if not USE_PARQUET_URL or not USE_INCOME_PARQUET_URL:
        expenditure_file = data_dir / 'expenditure_data.csv'  # or .xlsx
        income_file = data_dir / 'income_data.csv'  # or .xlsx
        
        if not expenditure_file.exists() or not income_file.exists():
            print("Please place your data files in the 'data' directory:")
            print("1. expenditure_data.csv (or .xlsx)")
            print("2. income_data.csv (or .xlsx)")
            return
    
    # Load and process data
    expenditure_df, income_df = load_data(
        data_dir / 'expenditure_data.csv' if not USE_PARQUET_URL else None,
        data_dir / 'income_data.csv' if not USE_INCOME_PARQUET_URL else None
    )
    if expenditure_df is None or income_df is None:
        return
    
    # Clean data
    expenditure_df, income_df = clean_data(expenditure_df, income_df)
    
    # Calculate affordability
    affordability_df = calculate_affordability(expenditure_df, income_df)
    
    # Perform additional analysis
    affordability_df = analyze_financial_pressure(affordability_df)
    
    # Print results
    print("\nAffordability Analysis Results:")
    print(affordability_df[['State', 'Income', 'Expenditure', 'Affordability_Ratio', 'Income_Expenditure_Gap', 'Financial_Pressure']])
    
    # Print detailed analysis
    print_analysis_summary(affordability_df)
    
    # Visualize results
    visualize_results(affordability_df)
    print("\nVisualization saved in the 'visualizations' directory:")

if __name__ == "__main__":
    main() 
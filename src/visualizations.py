import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import textwrap
import matplotlib.patches as mpatches

def plot_affordability_ratio(affordability_df, save_path='visualizations/affordability_ratio.png'):
    """
    Create an enhanced visualization of affordability ratios by state.
    
    Args:
        affordability_df (pd.DataFrame): DataFrame with affordability ratios
        save_path (str): Path to save the visualization
    """
    plt.figure(figsize=(15, 8))
    
    # Create the bar plot
    ax = sns.barplot(data=affordability_df, x='State', y='Affordability_Ratio')
    
    # Customize the plot
    plt.title('Affordability Ratio by State\n(Expenditure/Income)', pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('State', fontsize=12, labelpad=10)
    plt.ylabel('Expenditure/Income Ratio', fontsize=12, labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add a horizontal line at the mean ratio
    mean_ratio = affordability_df['Affordability_Ratio'].mean()
    plt.axhline(y=mean_ratio, color='red', linestyle='--', alpha=0.5)
    plt.text(len(affordability_df)-1, mean_ratio, f'Mean: {mean_ratio:.2f}', 
             color='red', ha='right', va='bottom')
    
    # Add value labels on top of each bar
    for i, v in enumerate(affordability_df['Affordability_Ratio']):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Add color gradient based on ratio values
    norm = plt.Normalize(affordability_df['Affordability_Ratio'].min(), 
                        affordability_df['Affordability_Ratio'].max())
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
    sm.set_array([])
    
    # Add colorbar with specified axes
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Financial Pressure Level', fontsize=10)
    
    # Add explanatory text
    plt.figtext(0.02, 0.02, 
                'Note: Higher ratio indicates higher financial pressure.\n'
                'Red line shows the mean ratio across all states.',
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_income_expenditure_scatter(affordability_df, save_path='visualizations/income_expenditure_scatter.png'):
    """
    Create an enhanced scatter plot of income vs expenditure with trend line and annotations.
    
    Args:
        affordability_df (pd.DataFrame): DataFrame with affordability ratios
        save_path (str): Path to save the visualization
    """
    # Ensure numeric and drop NaN
    df = affordability_df.copy()
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    df['Expenditure'] = pd.to_numeric(df['Expenditure'], errors='coerce')
    df = df.dropna(subset=['Income', 'Expenditure'])

    # Color by financial pressure
    pressure_palette = {
        'High Pressure': '#d73027',
        'Moderate-High Pressure': '#fc8d59',
        'Moderate-Low Pressure': '#fee08b',
        'Low Pressure': '#1a9850'
    }
    df['Color'] = df['Financial_Pressure'].map(pressure_palette)

    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    
    # Scatter plot with color coding
    for pressure, color in pressure_palette.items():
        sub = df[df['Financial_Pressure'] == pressure]
        ax.scatter(
            sub['Income'], sub['Expenditure'],
            s=160, c=color, label=pressure, edgecolor='black', alpha=0.8, zorder=3
        )
    
    # Add trend line using numpy float arrays
    x = df['Income'].astype(float).to_numpy()
    y = df['Expenditure'].astype(float).to_numpy()
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(df['Income'], p(x), "r--", alpha=0.8, label=f'Trend Line (y = {z[0]:.2f}x + {z[1]:.2f})', zorder=2)
    
    # Add 45-degree line (where income = expenditure)
    max_val = max(x.max(), y.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Income = Expenditure', zorder=1)
    
    # Add state labels with smart offset
    for i, row in df.iterrows():
        ax.annotate(
            row['State'],
            (row['Income'], row['Expenditure']),
            xytext=(7, 7), textcoords='offset points',
            fontsize=10, fontweight='bold', color='navy',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5)
        )
    
    # Customize the plot
    ax.set_title('Income vs Expenditure by State', pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel('Median Monthly Income (RM)', fontsize=13, labelpad=10)
    ax.set_ylabel('Mean Monthly Expenditure (RM)', fontsize=13, labelpad=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Add legend for financial pressure
    handles = [mpatches.Patch(color=color, label=pressure) for pressure, color in pressure_palette.items()]
    ax.legend(handles=handles + [ax.lines[0], ax.lines[1]],
              loc='upper left', fontsize=11, frameon=True)
    
    # Add explanatory text at the bottom, not overlapping
    plt.subplots_adjust(bottom=0.19)
    plt.gcf().text(
        0.01, 0.03,
        'Red dashed line: Trend (average spending pattern).\n'
        'Grey dashed line: Where income = expenditure.\n'
        'Points above grey line: spending > income. Below: spending < income.',
        fontsize=10, style='italic', ha='left', va='bottom'
    )
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_income_expenditure_gap(affordability_df, save_path='visualizations/income_expenditure_gap.png'):
    """
    Create an enhanced visualization of income-expenditure gaps by state.
    
    Args:
        affordability_df (pd.DataFrame): DataFrame with affordability ratios
        save_path (str): Path to save the visualization
    """
    plt.figure(figsize=(15, 8))
    
    # Sort data by gap
    sorted_df = affordability_df.sort_values('Income_Expenditure_Gap', ascending=False)
    
    # Create the bar plot
    ax = sns.barplot(data=sorted_df, x='State', y='Income_Expenditure_Gap')
    
    # Customize the plot
    plt.title('Monthly Income-Expenditure Gap by State', pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('State', fontsize=12, labelpad=10)
    plt.ylabel('Monthly Gap (RM)', fontsize=12, labelpad=10)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels above each bar and percentage labels inside bars
    for i, row in enumerate(sorted_df.itertuples()):
        # Value label above the bar
        ax.text(i, row.Income_Expenditure_Gap + max(sorted_df['Income_Expenditure_Gap']) * 0.01, 
                f'RM{row.Income_Expenditure_Gap:,.0f}', 
                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
        # Percentage label inside the bar
        percentage = (row.Income_Expenditure_Gap / row.Income) * 100
        ax.text(i, row.Income_Expenditure_Gap/2, 
                f'{percentage:.1f}%\nof income', 
                ha='center', va='center',
                color='white', fontweight='bold', fontsize=9, wrap=True)
    # Adjust y-limits for more space above bars
    ax.set_ylim(0, sorted_df['Income_Expenditure_Gap'].max() * 1.15)
    
    # Add color gradient based on gap values
    norm = plt.Normalize(sorted_df['Income_Expenditure_Gap'].min(), 
                        sorted_df['Income_Expenditure_Gap'].max())
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    
    # Add colorbar with specified axes
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Disposable Income Level', fontsize=10)
    
    # Add explanatory text
    plt.figtext(0.02, 0.02,
                'Note: Higher gap indicates more disposable income.\n'
                'Percentage shows the gap as a proportion of total income.',
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_financial_pressure_categories(affordability_df, save_path='visualizations/financial_pressure_categories.png'):
    """
    Create an enhanced visualization of financial pressure categories.
    
    Args:
        affordability_df (pd.DataFrame): DataFrame with affordability ratios
        save_path (str): Path to save the visualization
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot 1: Bar chart of categories with distinctive colors
    pressure_counts = affordability_df['Financial_Pressure'].value_counts()
    categories = pressure_counts.index.tolist()
    palette = sns.color_palette("Set2", len(categories))
    bar_colors = {cat: palette[i] for i, cat in enumerate(categories)}
    barplot = sns.barplot(x=categories, y=pressure_counts.values, ax=ax1, palette=palette)
    ax1.set_title('Number of States by Financial Pressure Category', pad=20, fontsize=16, fontweight='bold')
    ax1.set_xlabel('Financial Pressure Category', fontsize=13, labelpad=10)
    ax1.set_ylabel('Number of States', fontsize=13, labelpad=10)
    
    # Add value labels and state names inside bars
    for i, category in enumerate(categories):
        v = pressure_counts[category]
        states = affordability_df[affordability_df['Financial_Pressure'] == category]['State'].tolist()
        wrapped_states = '\n'.join(textwrap.wrap(', '.join(states), width=25))
        # Value label at the top of the bar
        ax1.text(i, v - 0.2, f'{v}', ha='center', va='top', fontsize=13, fontweight='bold', color='white')
        # State names inside the bar, centered
        ax1.text(i, v/2, wrapped_states, ha='center', va='center', fontsize=11, color='black', wrap=True, bbox=dict(facecolor=palette[i], alpha=0.3, boxstyle='round,pad=0.3'))
    
    # Set y-limits for better text placement
    ax1.set_ylim(0, max(pressure_counts.values) + 2)
    
    # Plot 2: Pie chart with percentage distribution and legend at the top
    wedges, texts, autotexts = ax2.pie(
        pressure_counts.values, 
        labels=None,  # Remove labels from the pie itself
        autopct='%1.1f%%',
        startangle=90,
        colors=palette,
        pctdistance=0.8
    )
    ax2.set_title('Distribution of States by Financial Pressure', pad=20, fontsize=16, fontweight='bold')
    
    # Add legend above the pie chart, horizontal
    legend_labels = []
    for i, category in enumerate(categories):
        states = affordability_df[affordability_df['Financial_Pressure'] == category]['State'].tolist()
        wrapped_states = ', '.join(states)
        legend_labels.append(f"{category}: {wrapped_states}")
    patches = [mpatches.Patch(color=palette[i], label=legend_labels[i]) for i in range(len(categories))]
    ax2.legend(
        handles=patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),  # 0.5 is center, -0.25 is below the axes
        ncol=1,  # or increase for horizontal layout if you wish
        fontsize=11,
        frameon=False,
        title="Financial Pressure (States)",
        title_fontsize=12
    )
    
    # Shrink pie chart to make room for legend
    ax2.set_position([0.58, 0.18, 0.32, 0.65])
    
    # Add explanatory text at the bottom, not overlapping
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.01, 0.04,
        'Note: Financial Pressure Categories are based on the ratio of expenditure to income:\n'
        'High Pressure (≥90%), Moderate-High (80-90%), Moderate-Low (70-80%), Low Pressure (<70%)',
        fontsize=10, style='italic', ha='left', va='bottom'
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
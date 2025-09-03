"""
Error Analysis and Visualization Tool

This script analyzes error data from different state estimation algorithms and generates
visualizations for comparison. It processes error data stored in a feather file and produces:

1. Violin plots showing error distributions across different features (roll rate, yaw rate, 
   velocities, angles, positions) for each estimator and sensor configuration
2. LaTeX tables with median error statistics and ranking information

The script expects input data with columns for:
- estimator: Different state estimation algorithms (End2End, FMUKF, UKF variants, etc.)
- sensor: Sensor configurations (ALL, GuessUV, etc.) 
- feature columns: Error values for each state variable (p, r, u, v, phi, psi, x, y)

Configuration is handled via Hydra, with settings for:
- Input/output file paths
- Estimator and feature labels/colors
- Plotting parameters (figure size, fonts, violin plot settings)
- Sensor configurations to include in analysis

Usage:
    python visualize.py
    python visualize.py io.input_file=path/to/data.feather
    python visualize.py plotting.use_pgf_rendering=false
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hydra
from omegaconf import DictConfig

# Register fmukf hydra resolvers
from fmukf.utils.hydra import register_fmukf_resolvers
register_fmukf_resolvers()


def load_error_data(file_path: str) -> pd.DataFrame:
    """
    Load error data from feather file containing estimator comparison results.
    
    Args:
        file_path: Path to the feather file containing error data
        
    Returns:
        DataFrame with columns for estimators, sensors, and error features
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the file format is invalid
    """
    try:
        df = pd.read_feather(file_path)
        print(f"Loaded error data from {file_path} with {len(df)} samples")
        return df
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        raise


def create_error_distribution_plots(df: pd.DataFrame, config: DictConfig) -> plt.Figure:
    """
    Create violin plots showing error distributions for each state variable.
    
    This function generates a grid of violin plots where each subplot shows the
    distribution of errors for a specific state variable (roll rate, yaw rate, etc.)
    across different estimators and sensor configurations.
    
    Args:
        df: DataFrame containing error data with estimator, sensor, and feature columns
        config: Hydra configuration containing plotting settings and labels
        
    Returns:
        matplotlib Figure object with the violin plot grid
    """
    
    # Configure matplotlib for LaTeX rendering if requested
    if config.plotting.use_pgf_rendering:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    
    # Extract configuration for features and estimators
    feature_order = list(config.features.labels.keys())
    feature_labels = config.features.labels
    estimator_order = list(config.estimators.labels.keys())
    estimator_formatting = config.estimators.labels
    
    # Create subplot grid
    n_rows, n_cols = config.plotting.subplot_grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=config.plotting.figure_size, 
                            dpi=config.plotting.dpi)
    
    # Handle single row/column cases
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    font_size = config.plotting.font_size
    
    # Create violin plot for each state variable
    for idx, feature in enumerate(feature_order):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Create violin plot showing error distribution by estimator and sensor
        vplot = sns.violinplot(
            ax=ax,
            data=df,
            x='estimator',
            y=feature,
            inner="quartiles",  # Show quartiles inside violin
            hue='sensor',       # Split by sensor configuration
            split=True,         # Split violins for different sensors
            bw_adjust=config.plotting.bw_adjust,
            legend=False,
            log_scale=True,     # Use log scale for error values
            width=config.plotting.violin_width,
            cut=config.plotting.violin_cut,
            density_norm='width',
            common_norm=True,
            order=estimator_order
        )
        
        # Apply custom colors to each estimator's violin plots
        for est_idx, est_key in enumerate(estimator_order):
            color = estimator_formatting[est_key]["color"]
            new_color = matplotlib.colors.to_rgba(color, alpha=config.plotting.violin_alpha)
            
            # Each violin split produces two collections (left and right halves)
            base_index = est_idx * 2
            ax.collections[base_index].set_facecolor(new_color)
            ax.collections[base_index + 1].set_facecolor(new_color)
            
            # Set edge properties for better visibility
            ax.collections[base_index].set_edgecolor('black')
            ax.collections[base_index + 1].set_edgecolor('black')
            ax.collections[base_index].set_linewidth(config.plotting.violin_edge_width)
            ax.collections[base_index + 1].set_linewidth(config.plotting.violin_edge_width)
        
        # Format subplot appearance
        ax.set_xlabel('')
        ax.set_xticklabels([])
        
        # Add y-axis label only for leftmost plots
        if col == 0:
            ax.set_ylabel("$log_{10}$ MAE", fontsize=font_size)
            ax.yaxis.set_label_coords(-0.2, 0.5)
        else:
            ax.set_ylabel('')
        
        # Add grid for better readability
        ax.set_axisbelow(True)
        ax.grid()
        
        # Add subplot title with state variable name
        ax.text(0.5, 0.99, fontsize=font_size, s=feature_labels[feature], 
                transform=ax.transAxes, horizontalalignment='center', verticalalignment='top')
        ax.tick_params(axis='y', labelsize=font_size)
    
    # Create legend showing estimator colors and names
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=estimator_formatting[estimator]["color"]) 
                      for estimator in estimator_order]
    
    fig.legend(handles=legend_elements,
               labels=[estimator_formatting[estimator]["display_name"] for estimator in estimator_order],
               loc='lower center',
               bbox_to_anchor=(0.525, -0.02),
               ncols=len(legend_elements),
               fontsize=font_size-2)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.05, left=0.09)
    
    return fig


def format_scientific_notation(x: float, precision: int = 3) -> str:
    """
    Format a number in scientific notation for LaTeX with specified precision.
    
    Args:
        x: Number to format
        precision: Number of decimal places to show
        
    Returns:
        LaTeX-formatted string in scientific notation
    """
    if x == 0:
        return "$0$"
    
    s = f"{x:.{precision}e}"
    if 'e' not in s:
        # Handle very small numbers that don't get scientific notation
        return f"${x:.{precision}f}$"
    
    mantissa, exp_str = s.split('e')
    exp_int = int(exp_str)
    return rf"${mantissa} \mathrm{{e}}{{{exp_int}}}$"


def compute_estimator_rankings(df: pd.DataFrame, config: DictConfig) -> dict[str, dict[str, float]]:
    """
    Compute mean rankings for each estimator across all features and sensors.
    
    For each sensor configuration and feature, this function ranks estimators
    by their median error (lower is better), then computes the average rank
    across all features for each estimator.
    
    Args:
        df: DataFrame containing error data
        config: Hydra configuration with feature and estimator settings
        
    Returns:
        Dictionary mapping sensor -> estimator -> mean_rank
    """
    rankings = {}
    
    # Process each sensor configuration
    for sensor in config.features.sensors:
        df_sensor = df[df["sensor"] == sensor]
        rankings[sensor] = {estimator: [] for estimator in config.estimators.table_labels}
        
        # Compute rankings for each feature
        for feature in config.features.table_labels:
            # Calculate median error for each estimator
            est_2_median = {estimator: df_sensor[df_sensor["estimator"] == estimator][feature].median() 
                           for estimator in config.estimators.table_labels}
            
            # Sort by median error (ascending) and assign ranks
            est_2_medians = sorted(est_2_median.items(), key=lambda x: x[1])
            est_2_rank = {estimator: i+1 for i, (estimator, _) in enumerate(est_2_medians)}
            
            # Store rank for this feature
            for estimator in config.estimators.table_labels:
                rankings[sensor][estimator].append(est_2_rank[estimator])
    
    # Compute mean ranks across all features
    mean_rankings = {sensor: {estimator: sum(rankings[sensor][estimator])/len(rankings[sensor][estimator]) 
                              for estimator in rankings[sensor]} for sensor in rankings}
    
    return mean_rankings


def generate_latex_error_table(df: pd.DataFrame, config: DictConfig) -> str:
    """
    Generate LaTeX table with error statistics and rankings.
    
    Creates a LaTeX table showing median errors for each feature, estimator, and
    sensor configuration, along with overall rankings. The table structure is:
    - Rows: Features (p, r, u, v, phi, psi, x, y)
    - Columns: Estimators
    - Subrows: Sensor configurations (h1, h2, etc.)
    - Final rows: Mean rankings
    
    Args:
        df: DataFrame containing error data
        config: Hydra configuration with labels and settings
        
    Returns:
        Complete LaTeX table as a string
    """
    
    # Compute rankings for all estimators
    mean_rankings = compute_estimator_rankings(df, config)
    
    # Build table header
    table_str = "\\begin{tabular}{ll" + "".join(["|c"]*len(config.estimators.table_labels)) + "} \n"
    
    # Add header row with estimator names
    table_str += "   &  "
    for estimator in config.estimators.table_labels:
        table_str += f"& {config.estimators.table_labels[estimator]} "
    table_str += "\\\\ \\hline \n"
    
    # Add data rows for each feature
    for feature in config.features.table_labels:
        # Add subrows for each sensor configuration
        for sensor_idx, sensor in enumerate(config.features.sensors):
            # Use sensor labels from config instead of hard-coded values
            sensor_label = f"$h_{sensor_idx + 1}$" if sensor_idx == 0 else f"$h_{sensor_idx + 1}$"
            
            # First subrow gets feature name, others are empty
            if sensor_idx == 0:
                table_str += f" {config.features.table_labels[feature]} & {sensor_label} "
            else:
                table_str += " & " + sensor_label
            
            # Add median errors for each estimator
            for estimator in config.estimators.table_labels:
                err = df[(df["estimator"] == estimator) & (df["sensor"] == sensor)][feature].median()
                table_str += f"& {format_scientific_notation(err, 2)} "
            
            # Add line break after first row, add line break with hline after second row
            if sensor_idx == 0:
                table_str += "\\\\ \n"
            else:
                table_str += "\\\\ \\hline \n"
    
    # Add ranking rows
    for sensor_idx, sensor in enumerate(config.features.sensors):
        sensor_label = f"$h_{sensor_idx + 1}$" if sensor_idx == 0 else f"$h_{sensor_idx + 1}$"
        
        if sensor_idx == 0:
            table_str += f" rank & {sensor_label} "
        else:
            table_str += f"  & {sensor_label} "
            
        for estimator in config.estimators.table_labels:
            rank = mean_rankings[sensor][estimator]
            # Format to 3 significant figures
            table_str += f"& {rank:.3g} "
        
        # Add line break after first row, add line break with hline after second row
        if sensor_idx == 0:
            table_str += "\\\\ \n"
        else:
            table_str += "\\\\ \\hline \n"
    table_str += " \n \\end{tabular} \n"
    
    return table_str


@hydra.main(version_base=None, config_path="config", config_name="visualize")
def main(config: DictConfig):
    """
    Main function to generate error analysis visualizations.
    
    This function orchestrates the entire visualization pipeline:
    1. Loads error data from the specified input file
    2. Creates violin plots showing error distributions
    3. Generates LaTeX table with error statistics and rankings
    4. Saves outputs to specified file paths
    
    Configuration options (see config/visualize.yaml for details)
    - io.input_file: Path to input feather file with error data
    - io.output_violin: Path to save violin plot PDF
    - io.output_table: Path to save LaTeX table
    - plotting.*: Various plotting parameters (figure size, fonts, etc.)
    - estimators.*: Estimator names, colors, and display labels
    - features.*: Feature names, labels, and sensor configurations
    """
    
    # Load error data
    df = load_error_data(config.io.input_file)
    
    # Create violin plot showing error distributions
    print("Creating error distribution plots...")
    fig = create_error_distribution_plots(df, config)
    fig.savefig(config.io.output_violin, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved violin plot to {config.io.output_violin}")
    
    # Generate LaTeX table with error statistics
    print("Generating LaTeX error table...")
    table_str = generate_latex_error_table(df, config)
    
    # Save table to file
    with open(config.io.output_table, 'w') as f:
        f.write(table_str)
    print(f"Saved LaTeX table to {config.io.output_table}")
    
    # Print table to console for easy viewing
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print(table_str)
    print("="*80)


if __name__ == "__main__":
    main() 
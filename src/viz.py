import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd
import seaborn as sns  # legacy dependency kept for compatibility
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import cm
import numpy as np
import pycountry

# Load country code dataframe
COUNTRY_CODE_PATH = "./data/raw/Country Code.csv"
ctry_df = pd.read_csv(COUNTRY_CODE_PATH)


def ctry_convert(ctry_input: str, ctry_df: pd.DataFrame = ctry_df) -> Optional[str]:
    """
    Convert between country names and codes using a reference dataframe.
    
    Args:
        ctry_input: Country name or code to convert
        ctry_df: Reference dataframe with country data
        
    Returns:
        Converted country name or code, or None if no match found
    """
    input_len = len(ctry_input)

    if input_len <= 1:
        # Too short to be valid
        return None
    elif input_len == 2:
        # Handle alpha2 code
        matches = ctry_df[ctry_df["Alpha2"] == ctry_input]
        if not matches.empty:
            return matches["Full Name"].values[0]
        return None
    elif input_len == 3:
        # Handle alpha3 code
        matches = ctry_df[ctry_df["Alpha3"] == ctry_input]
        if not matches.empty:
            return matches["Full Name"].values[0]
        return None
    else:
        # Handle country name
        
        # Exact match
        exact_matches = ctry_df[ctry_df["Full Name"] == ctry_input]
        if not exact_matches.empty:
            return exact_matches["Alpha2"].values[0]
            
        # Match countries that start with the input
        starts_with_matches = ctry_df[ctry_df["Full Name"].str.startswith(ctry_input)]
        if not starts_with_matches.empty:
            return starts_with_matches["Alpha2"].values[0]
        
        # Match countries containing the input
        try:
            contains_matches = ctry_df[ctry_df["Full Name"].str.contains(ctry_input)]
            if not contains_matches.empty:
                return contains_matches["Alpha2"].values[0]
        except:
            pass
            
        # Try matching with common name
        try:
            common_name_matches = ctry_df[ctry_df["Common Name"] == ctry_input]
            if not common_name_matches.empty:
                return common_name_matches["Alpha2"].values[0]
        except:
            pass
                
        # No matches found
        return None


def get_alpha_3(country_name: str) -> str:
    """
    Get alpha-3 country code from country name.
    
    Args:
        country_name: Country name to convert
    
    Returns:
        Alpha-3 code or original name if not found
    """
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return country_name


def _save_or_show_plot(plt, output_dir: Optional[str], save_name: str, fig_suffix: str, 
                      show_plot: bool, dpi: int = 1200) -> None:
    """
    Helper function to save or show a plot.
    
    Args:
        plt: Matplotlib pyplot object
        output_dir: Directory to save the plot
        save_name: Filename for the plot
        fig_suffix: File extension
        show_plot: Whether to display the plot
        dpi: Resolution for saved plot
    """
    if output_dir:
        output_path = os.path.abspath(output_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        
        save_path = os.path.join(output_path, f"{save_name}.{fig_suffix}")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_rer_imb(df: pd.DataFrame, log_scale: bool = True, annotate: bool = True, 
                show_plot: bool = False, point_alpha: float = 0, text_alpha: float = 0.8, 
                fontsize: int = 10, save_name: str = "Average Trade Imbalance vs Average Real Exchange Rate by Country (2013-2023)", 
                fig_suffix: str = "png", output_dir: Optional[str] = None) -> None:
    """
    Plot real effective exchange rate vs trade imbalance scatter plot.
    
    Args:
        df: Dataframe with country macroeconomic data
        log_scale: Whether to use log scale for y-axis
        annotate: Whether to add country code labels
        show_plot: Whether to display the plot
        point_alpha: Transparency of scatter points
        text_alpha: Transparency of text labels
        fontsize: Size of text labels
        save_name: Filename for saving the plot
        fig_suffix: File extension for saved plot
        output_dir: Directory to save the plot
    """
    df["Country Code"] = df["Country Name"].apply(get_alpha_3)
    plt.figure(figsize=(12, 8))
    plt.scatter(df["Trade Imbalance to GDP Ratio"], df["Real Effective Exchange Rate"], alpha=point_alpha)
    
    if annotate:
        for _, row in df.iterrows():
            plt.text(row["Trade Imbalance to GDP Ratio"], row["Real Effective Exchange Rate"], row["Country Code"],
                     fontsize=fontsize, alpha=text_alpha)

    plt.xlabel("Average Trade Imbalance / GDP (2013–2023)")
    plt.ylabel("Average Real Effective Exchange Rate (2013–2023)")
    plt.title("Trade Imbalance vs Real Effective Exchange Rate")
    plt.grid(True)
    plt.tight_layout()
    if log_scale:
        plt.yscale("log")

    _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot)


def _setup_single_country_plot(df: pd.DataFrame, country: str, y_column: str) -> pd.DataFrame:
    """
    Set up single country time series plot.
    
    Args:
        df: Dataframe with country data
        country: Country name to plot
        y_column: Column to plot on y-axis
        
    Returns:
        Filtered dataframe for the specified country
    """
    country_code = ctry_convert(country)
    if not country_code:
        raise ValueError(f"Could not find country code for: {country}")
        
    ctry_df_filtered = df[df['Country Name'] == country_code].sort_values("Year")
    if ctry_df_filtered.empty:
        raise ValueError(f"No data found for {country} (code: {country_code})")
        
    return ctry_df_filtered


def plot_ctry_imb_series(df: pd.DataFrame, output_dir: Optional[str] = None, country: Optional[str] = None, 
                        show_plot: bool = False, save_name: Optional[str] = None, 
                        fig_suffix: str = "png") -> None:
    """
    Plot trade imbalance time series for a country.
    
    Args:
        df: Dataframe with country data
        output_dir: Directory to save the plot
        country: Country name to plot
        show_plot: Whether to display the plot
        save_name: Filename for saving the plot
        fig_suffix: File extension for saved plot
    """
    ctry_df_filtered = _setup_single_country_plot(df, country, "Trade Imbalance")
    
    plt.figure(figsize=(10, 5))
    plt.plot(
        ctry_df_filtered["Year"], 
        ctry_df_filtered["Trade Imbalance"],
        marker='o', color='red', linewidth=2
    )

    plt.title(f"{country}: Trade Imbalance (2013–2023)")
    plt.xlabel("Year")
    plt.ylabel("Trade Imbalance")
    plt.grid(True)
    plt.xticks(ctry_df_filtered["Year"])
    plt.tight_layout()

    _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot)


def plot_ctry_gdp_imb_series(df: pd.DataFrame, output_dir: Optional[str] = None, country: Optional[str] = None, 
                            show_plot: bool = False, save_name: Optional[str] = None, 
                            fig_suffix: str = "png") -> None:
    """
    Plot trade imbalance to GDP ratio time series for a country.
    
    Args:
        df: Dataframe with country data
        output_dir: Directory to save the plot
        country: Country name to plot
        show_plot: Whether to display the plot
        save_name: Filename for saving the plot
        fig_suffix: File extension for saved plot
    """
    ctry_df_filtered = _setup_single_country_plot(df, country, "Trade Imbalance to GDP Ratio")
    
    plt.figure(figsize=(10, 5))
    plt.plot(
        ctry_df_filtered["Year"], 
        ctry_df_filtered["Trade Imbalance to GDP Ratio"],
        marker='o', color='orange', linewidth=2
    )

    plt.title(f"{country}: Trade Imbalance to GDP Ratio (2013–2023)")
    plt.xlabel("Year")
    plt.ylabel("Trade Imbalance / GDP")
    plt.grid(True)
    plt.xticks(ctry_df_filtered["Year"])
    plt.tight_layout()

    _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot)


def plot_ctry_eff_series(df: pd.DataFrame, output_dir: Optional[str] = None, country: Optional[str] = None, 
                        show_plot: bool = False, save_name: Optional[str] = None, 
                        fig_suffix: str = "png") -> None:
    """
    Plot real effective exchange rate time series for a country.
    
    Args:
        df: Dataframe with country data
        output_dir: Directory to save the plot
        country: Country name to plot
        show_plot: Whether to display the plot
        save_name: Filename for saving the plot
        fig_suffix: File extension for saved plot
    """
    ctry_df_filtered = _setup_single_country_plot(df, country, "Real Effective Exchange Rate")
    
    plt.figure(figsize=(10, 5))
    plt.plot(
        ctry_df_filtered["Year"], 
        ctry_df_filtered["Real Effective Exchange Rate"],
        marker='o', color='blue', linewidth=2  # Changed color to blue to distinguish from GDP imbalance
    )

    plt.title(f"{country}: Real Effective Exchange Rate (2013–2023)")
    plt.xlabel("Year")
    plt.ylabel("Real Effective Exchange Rate")
    plt.grid(True)
    plt.xticks(ctry_df_filtered["Year"])
    plt.tight_layout()

    _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot)


def _setup_multi_country_plot(figsize: tuple = (14, 8), colors: Optional[List[str]] = None, 
                             markers: Optional[List[str]] = None, color_scheme: str = 'diverse', 
                             cmap_name: str = 'viridis') -> tuple:
    """
    Set up multi-country time series plot with color schemes.
    
    Args:
        figsize: Size of the figure
        colors: List of colors for each country line
        markers: List of markers for each country line
        color_scheme: Type of color scheme ('palette', 'cmap', 'diverse')
        cmap_name: Name of the colormap to use if color_scheme is 'cmap'
        
    Returns:
        Tuple of (colors, markers)
    """
    plt.figure(figsize=figsize)
    
    # Markers if not provided
    if not markers:
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'h', '+', '1', '2', '3', '4', '8', 'P']
    
    # Color schemes
    if colors is None:
        if color_scheme == 'palette':
            # Predefined harmonious color palette
            colors = [
                '#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', 
                '#7f7f7f', '#bcbd22', '#17becf', '#d62728', '#ff7f0e',
                '#aec7e8', '#98df8a', '#c5b0d5', '#c49c94', '#f7b6d2',
                '#c7c7c7', '#dbdb8d', '#9edae5', '#ff9896', '#ffbb78'
            ]
        elif color_scheme == 'cmap':
            # Use a matplotlib colormap for a smooth gradient
            n_colors = 20  # Number of colors to extract from the colormap
            cmap = cm.get_cmap(cmap_name, n_colors)
            colors = [cmap(i/n_colors) for i in range(n_colors)]
        else:  # diverse
            # Default diverse colors
            colors = [
                'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 
                'olive', 'cyan', 'darkblue', 'darkorange', 'darkgreen', 'darkred', 
                'darkviolet', 'goldenrod'
            ]
    
    return colors, markers


def _plot_multi_country_series(df: pd.DataFrame, countries: List[str], y_column: str, 
                              y_label: str, title: str, colors: List[str], 
                              markers: List[str]) -> int:
    """
    Helper function to plot multi-country time series.
    
    Args:
        df: Dataframe with country data
        countries: List of countries to plot
        y_column: Column to plot on y-axis
        y_label: Label for y-axis
        title: Plot title
        colors: List of colors for each country line
        markers: List of markers for each country line
        
    Returns:
        Number of successfully plotted countries
    """
    # Ensure we have enough colors and markers
    while len(colors) < len(countries):
        colors.extend(colors)
    while len(markers) < len(countries):
        markers.extend(markers)
    
    successful_plots = 0
    for i, country in enumerate(countries):
        try:
            country_code = ctry_convert(country)
            if country_code:
                ctry_df_filtered = df[df['Country Name'] == country_code].sort_values("Year")
                
                if not ctry_df_filtered.empty:
                    plt.plot(
                        ctry_df_filtered["Year"], 
                        ctry_df_filtered[y_column],
                        marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)],
                        linewidth=2,
                        label=country
                    )
                    successful_plots += 1
                else:
                    print(f"No data found for {country} (code: {country_code})")
            else:
                print(f"Could not convert country name: {country}")
        except Exception as e:
            print(f"Error plotting {country}: {e}")
    
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(y_label)
    plt.grid(True)
    
    return successful_plots


def plot_multi_ctry_imb_series(df: pd.DataFrame, countries: List[str], output_dir: Optional[str] = None, 
                              show_plot: bool = False, save_name: str = "Multi_Country_Trade_Imbalance", 
                              fig_suffix: str = "png", figsize: tuple = (14, 8), 
                              title: str = "Trade Imbalance to GDP Ratio Comparison (2013-2023)",
                              colors: Optional[List[str]] = None, markers: Optional[List[str]] = None, 
                              legend_loc: str = 'center left', bbox_to_anchor: tuple = (1.02, 0.5), 
                              legend_fontsize: int = 9, color_scheme: str = 'diverse', 
                              cmap_name: str = 'viridis') -> None:
    """
    Plot trade imbalance to GDP ratio time series for multiple countries.
    
    Args:
        df: Dataframe with country data
        countries: List of countries to plot
        output_dir: Directory to save the plot
        show_plot: Whether to display the plot
        save_name: Filename for saving the plot
        fig_suffix: File extension for saved plot
        figsize: Size of the figure
        title: Plot title
        colors: List of colors for each country line
        markers: List of markers for each country line
        legend_loc: Location of the legend
        bbox_to_anchor: Position to place the legend
        legend_fontsize: Font size for legend text
        color_scheme: Type of color scheme ('palette', 'cmap', 'diverse')
        cmap_name: Name of the colormap to use if color_scheme is 'cmap'
    """
    colors, markers = _setup_multi_country_plot(figsize, colors, markers, color_scheme, cmap_name)
    
    successful_plots = _plot_multi_country_series(
        df, countries, "Trade Imbalance to GDP Ratio", "Trade Imbalance / GDP", 
        title, colors, markers
    )
    
    # Place legend outside the plot area to the right
    if successful_plots > 0:
        plt.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, 
                  fontsize=legend_fontsize, framealpha=0.9)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legend on the right
    
    _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot)


def plot_multi_ctry_eff_series(df: pd.DataFrame, countries: List[str], output_dir: Optional[str] = None, 
                              show_plot: bool = False, save_name: str = "Multi_Country_Real_Effective_Exchange_Rate", 
                              fig_suffix: str = "png", figsize: tuple = (14, 8), 
                              title: str = "Real Effective Exchange Rate Comparison (2013-2023)",
                              colors: Optional[List[str]] = None, markers: Optional[List[str]] = None, 
                              legend_loc: str = 'center left', bbox_to_anchor: tuple = (1.02, 0.5), 
                              legend_fontsize: int = 9, color_scheme: str = 'diverse', 
                              cmap_name: str = 'viridis') -> None:
    """
    Plot real effective exchange rate time series for multiple countries.
    
    Args:
        df: Dataframe with country data
        countries: List of countries to plot
        output_dir: Directory to save the plot
        show_plot: Whether to display the plot
        save_name: Filename for saving the plot
        fig_suffix: File extension for saved plot
        figsize: Size of the figure
        title: Plot title
        colors: List of colors for each country line
        markers: List of markers for each country line
        legend_loc: Location of the legend
        bbox_to_anchor: Position to place the legend
        legend_fontsize: Font size for legend text
        color_scheme: Type of color scheme ('palette', 'cmap', 'diverse')
        cmap_name: Name of the colormap to use if color_scheme is 'cmap'
    """
    colors, markers = _setup_multi_country_plot(figsize, colors, markers, color_scheme, cmap_name)
    
    successful_plots = _plot_multi_country_series(
        df, countries, "Real Effective Exchange Rate", "Real Effective Exchange Rate", 
        title, colors, markers
    )
    
    # Place legend outside the plot area to the right
    if successful_plots > 0:
        plt.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, 
                  fontsize=legend_fontsize, framealpha=0.9)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legend on the right
    
    _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot)


def plot_neglog_plgdpo_vs_imb(
    df: pd.DataFrame,
    countries: List[str],
    output_dir: Optional[str] = None,
    show_plot: bool = False,
    save_name: str = "NegLog_PLGDPO_vs_Imbalance",
    fig_suffix: str = "png",
    start_year: int = 2009,
    end_year: int = 2019,
    annotate: bool = True,
    point_alpha: float = 0.7,
    text_alpha: float = 0.9,
    fontsize: int = 9,
    color_scheme: str = "cmap",
    cmap_name: str = "viridis",
) -> None:
    """
    Scatter plot of −log(pl_gdpo) (x) vs Trade Imbalance/GDP (y).

    Each *country* is represented by **one point**, the mean of the two
    variables across *start_year* … *end_year* (inclusive).
    
    Args:
        df: Dataframe with country data
        countries: List of countries to plot
        output_dir: Directory to save the plot
        show_plot: Whether to display the plot
        save_name: Filename for saving the plot
        fig_suffix: File extension for saved plot
        start_year: Starting year for calculating means
        end_year: Ending year for calculating means
        annotate: Whether to add country code labels
        point_alpha: Transparency of scatter points
        text_alpha: Transparency of text labels
        fontsize: Size of text labels
        color_scheme: Type of color scheme ('cmap' or other)
        cmap_name: Name of the colormap to use if color_scheme is 'cmap'
    """
    required = {"Country Name", "Year", "pl_gdpo", "Trade Imbalance to GDP Ratio"}
    lacking = required - set(df.columns)
    if lacking:
        raise KeyError(f"DataFrame missing required columns: {lacking}")

    # Filter years
    sub = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)].copy()
    if sub.empty:
        raise ValueError("No data in the specified year window.")

    # Compute −log(pl_gdpo)
    with np.errstate(divide="ignore"):
        sub["NegLog_pl_gdpo"] = -np.log(sub["pl_gdpo"])

    # Calculate means per country
    mean_df = (sub.groupby("Country Name")[["NegLog_pl_gdpo", "Trade Imbalance to GDP Ratio"]]
                  .mean()
                  .reset_index())

    # Restrict to selected countries (after conversion)
    resolved_codes = {ctry_convert(c) or c: c for c in countries}
    mean_df = mean_df[mean_df["Country Name"].isin(resolved_codes.keys())]

    if mean_df.empty:
        raise ValueError("No matching countries found after processing.")

    # Colour map
    if color_scheme == "cmap":
        cmap = cm.get_cmap(cmap_name, len(mean_df))
        colors = [cmap(i) for i in range(len(mean_df))]
    else:
        colors = [None] * len(mean_df)  # Let matplotlib decide default cycle

    # Draw scatter
    plt.figure(figsize=(10, 7))
    plt.scatter(mean_df["NegLog_pl_gdpo"],
                mean_df["Trade Imbalance to GDP Ratio"],
                alpha=point_alpha, c=colors, edgecolor="k")

    if annotate:
        for idx, row in mean_df.iterrows():
            country_alpha3 = get_alpha_3(resolved_codes[row["Country Name"]])
            plt.text(row["NegLog_pl_gdpo"], row["Trade Imbalance to GDP Ratio"],
                     country_alpha3, fontsize=fontsize, alpha=text_alpha,
                     ha="center", va="center")

    plt.xlabel("−log(pl_gdpo)  (mean {}‑{})".format(start_year, end_year))
    plt.ylabel("Trade Imbalance / GDP  (mean {}‑{})".format(start_year, end_year))
    plt.title("−log(pl_gdpo) vs Trade Imbalance / GDP")
    plt.grid(True)
    plt.tight_layout()

    _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot)
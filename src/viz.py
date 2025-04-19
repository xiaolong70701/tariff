#!/usr/bin/env python3
"""
visualization module for macroeconomic data analysis
provides functions to plot trade imbalance and exchange rate data
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import pycountry
import matplotlib.cm as cm
import numpy as np


# load country code dataframe
COUNTRY_CODE_PATH = "./data/raw/Country Code.csv"
ctry_df = pd.read_csv(COUNTRY_CODE_PATH)


def ctry_convert(ctry_input, ctry_df=ctry_df):
    """
    convert between country names and codes using a reference dataframe
    
    args:
        ctry_input: country name or code to convert
        ctry_df: reference dataframe with country data
        
    returns:
        converted country name or code, or none if no match found
    """
    input_len = len(ctry_input)

    if input_len <= 1:
        # too short to be valid
        return None
    elif input_len == 2:
        # handle alpha2 code
        matches = ctry_df[ctry_df["Alpha2"] == ctry_input]
        if not matches.empty:
            return matches["Full Name"].values[0]
        return None
    elif input_len == 3:
        # handle alpha3 code
        matches = ctry_df[ctry_df["Alpha3"] == ctry_input]
        if not matches.empty:
            return matches["Full Name"].values[0]
        return None
    else:
        # handle country name
        
        # exact match
        exact_matches = ctry_df[ctry_df["Full Name"] == ctry_input]
        if not exact_matches.empty:
            return exact_matches["Alpha2"].values[0]
            
        # match countries that start with the input
        starts_with_matches = ctry_df[ctry_df["Full Name"].str.startswith(ctry_input)]
        if not starts_with_matches.empty:
            return starts_with_matches["Alpha2"].values[0]
        
        # match countries containing the input
        try:
            contains_matches = ctry_df[ctry_df["Full Name"].str.contains(ctry_input)]
            if not contains_matches.empty:
                return contains_matches["Alpha2"].values[0]
        except:
            pass
            
        # try matching with common name
        try:
            common_name_matches = ctry_df[ctry_df["Common Name"] == ctry_input]
            if not common_name_matches.empty:
                return common_name_matches["Alpha2"].values[0]
        except:
            pass
                
        # no matches found
        return None


def get_alpha_3(country_name):
    """
    get alpha-3 country code from country name
    
    args:
        country_name: country name to convert
    
    returns:
        alpha-3 code or original name if not found
    """
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return country_name


def _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot, dpi=1200):
    """
    helper function to save or show a plot
    
    args:
        plt: matplotlib pyplot object
        output_dir: directory to save the plot
        save_name: filename for the plot
        fig_suffix: file extension
        show_plot: whether to display the plot
        dpi: resolution for saved plot
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


def plot_rer_imb(df, log_scale=True, annotate=True, show_plot=False, 
                point_alpha=0, text_alpha=0.8, fontsize=10, 
                save_name="Average Trade Imbalance vs Average Real Exchange Rate by Country (2013-2023)", 
                fig_suffix="png", output_dir=None):
    """
    plot real effective exchange rate vs trade imbalance scatter plot
    
    args:
        df: dataframe with country macroeconomic data
        log_scale: whether to use log scale for y-axis
        annotate: whether to add country code labels
        show_plot: whether to display the plot
        point_alpha: transparency of scatter points
        text_alpha: transparency of text labels
        fontsize: size of text labels
        save_name: filename for saving the plot
        fig_suffix: file extension for saved plot
        output_dir: directory to save the plot
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


def _setup_single_country_plot(df, country, y_column):
    """
    set up single country time series plot
    
    args:
        df: dataframe with country data
        country: country name to plot
        y_column: column to plot on y-axis
        
    returns:
        filtered dataframe for the specified country
    """
    country_code = ctry_convert(country)
    if not country_code:
        raise ValueError(f"Could not find country code for: {country}")
        
    ctry_df_filtered = df[df['Country Name'] == country_code].sort_values("Year")
    if ctry_df_filtered.empty:
        raise ValueError(f"No data found for {country} (code: {country_code})")
        
    return ctry_df_filtered


def plot_ctry_imb_series(df, output_dir=None, country=None, show_plot=False, 
                        save_name=None, fig_suffix="png"):
    """
    plot trade imbalance time series for a country
    
    args:
        df: dataframe with country data
        output_dir: directory to save the plot
        country: country name to plot
        show_plot: whether to display the plot
        save_name: filename for saving the plot
        fig_suffix: file extension for saved plot
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


def plot_ctry_gdp_imb_series(df, output_dir=None, country=None, show_plot=False, 
                            save_name=None, fig_suffix="png"):
    """
    plot trade imbalance to gdp ratio time series for a country
    
    args:
        df: dataframe with country data
        output_dir: directory to save the plot
        country: country name to plot
        show_plot: whether to display the plot
        save_name: filename for saving the plot
        fig_suffix: file extension for saved plot
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


def plot_ctry_eff_series(df, output_dir=None, country=None, show_plot=False, 
                        save_name=None, fig_suffix="png"):
    """
    plot real effective exchange rate time series for a country
    
    args:
        df: dataframe with country data
        output_dir: directory to save the plot
        country: country name to plot
        show_plot: whether to display the plot
        save_name: filename for saving the plot
        fig_suffix: file extension for saved plot
    """
    ctry_df_filtered = _setup_single_country_plot(df, country, "Real Effective Exchange Rate")
    
    plt.figure(figsize=(10, 5))
    plt.plot(
        ctry_df_filtered["Year"], 
        ctry_df_filtered["Real Effective Exchange Rate"],
        marker='o', color='blue', linewidth=2  # changed color to blue to distinguish from GDP imbalance
    )

    plt.title(f"{country}: Real Effective Exchange Rate (2013–2023)")
    plt.xlabel("Year")
    plt.ylabel("Real Effective Exchange Rate")
    plt.grid(True)
    plt.xticks(ctry_df_filtered["Year"])
    plt.tight_layout()

    _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot)


def _setup_multi_country_plot(figsize=(14, 8), colors=None, markers=None, color_scheme='diverse', cmap_name='viridis'):
    """
    set up multi-country time series plot with color schemes
    
    args:
        figsize: size of the figure
        colors: list of colors for each country line
        markers: list of markers for each country line
        color_scheme: type of color scheme ('palette', 'cmap', 'diverse')
        cmap_name: name of the colormap to use if color_scheme is 'cmap'
        
    returns:
        tuple of (colors, markers)
    """
    plt.figure(figsize=figsize)
    
    # markers if not provided
    if not markers:
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'h', '+', '1', '2', '3', '4', '8', 'P']
    
    # color schemes
    if colors is None:
        if color_scheme == 'palette':
            # predefined harmonious color palette
            colors = [
                '#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', 
                '#7f7f7f', '#bcbd22', '#17becf', '#d62728', '#ff7f0e',
                '#aec7e8', '#98df8a', '#c5b0d5', '#c49c94', '#f7b6d2',
                '#c7c7c7', '#dbdb8d', '#9edae5', '#ff9896', '#ffbb78'
            ]
        elif color_scheme == 'cmap':
            # use a matplotlib colormap for a smooth gradient
            n_colors = 20  # number of colors to extract from the colormap
            cmap = cm.get_cmap(cmap_name, n_colors)
            colors = [cmap(i/n_colors) for i in range(n_colors)]
        else:  # diverse
            # default diverse colors
            colors = [
                'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 
                'olive', 'cyan', 'darkblue', 'darkorange', 'darkgreen', 'darkred', 
                'darkviolet', 'goldenrod'
            ]
    
    return colors, markers


def _plot_multi_country_series(df, countries, y_column, y_label, title, colors, markers):
    """
    helper function to plot multi-country time series
    
    args:
        df: dataframe with country data
        countries: list of countries to plot
        y_column: column to plot on y-axis
        y_label: label for y-axis
        title: plot title
        colors: list of colors for each country line
        markers: list of markers for each country line
    """
    # ensure we have enough colors and markers
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
                    print(f"no data found for {country} (code: {country_code})")
            else:
                print(f"could not convert country name: {country}")
        except Exception as e:
            print(f"error plotting {country}: {e}")
    
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(y_label)
    plt.grid(True)
    
    return successful_plots


def plot_multi_ctry_imb_series(df, countries, output_dir=None, show_plot=False, 
                         save_name="Multi_Country_Trade_Imbalance", fig_suffix="png", 
                         figsize=(14, 8), title="Trade Imbalance to GDP Ratio Comparison (2013-2023)",
                         colors=None, markers=None, legend_loc='center left', 
                         bbox_to_anchor=(1.02, 0.5), legend_fontsize=9,
                         color_scheme='diverse', cmap_name='viridis'):
    """
    plot trade imbalance to gdp ratio time series for multiple countries
    
    args:
        df: dataframe with country data
        countries: list of countries to plot
        output_dir: directory to save the plot
        show_plot: whether to display the plot
        save_name: filename for saving the plot
        fig_suffix: file extension for saved plot
        figsize: size of the figure
        title: plot title
        colors: list of colors for each country line
        markers: list of markers for each country line
        legend_loc: location of the legend
        bbox_to_anchor: position to place the legend
        legend_fontsize: font size for legend text
        color_scheme: type of color scheme ('palette', 'cmap', 'diverse')
        cmap_name: name of the colormap to use if color_scheme is 'cmap'
    """
    colors, markers = _setup_multi_country_plot(figsize, colors, markers, color_scheme, cmap_name)
    
    successful_plots = _plot_multi_country_series(
        df, countries, "Trade Imbalance to GDP Ratio", "Trade Imbalance / GDP", 
        title, colors, markers
    )
    
    # place legend outside the plot area to the right
    if successful_plots > 0:
        plt.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, 
                  fontsize=legend_fontsize, framealpha=0.9)
    
    # adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # make room for legend on the right
    
    _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot)


def plot_multi_ctry_eff_series(df, countries, output_dir=None, show_plot=False, 
                         save_name="Multi_Country_Real_Effective_Exchange_Rate", fig_suffix="png", 
                         figsize=(14, 8), title="Real Effective Exchange Rate Comparison (2013-2023)",
                         colors=None, markers=None, legend_loc='center left', 
                         bbox_to_anchor=(1.02, 0.5), legend_fontsize=9,
                         color_scheme='diverse', cmap_name='viridis'):
    """
    plot real effective exchange rate time series for multiple countries
    
    args:
        df: dataframe with country data
        countries: list of countries to plot
        output_dir: directory to save the plot
        show_plot: whether to display the plot
        save_name: filename for saving the plot
        fig_suffix: file extension for saved plot
        figsize: size of the figure
        title: plot title
        colors: list of colors for each country line
        markers: list of markers for each country line
        legend_loc: location of the legend
        bbox_to_anchor: position to place the legend
        legend_fontsize: font size for legend text
        color_scheme: type of color scheme ('palette', 'cmap', 'diverse')
        cmap_name: name of the colormap to use if color_scheme is 'cmap'
    """
    colors, markers = _setup_multi_country_plot(figsize, colors, markers, color_scheme, cmap_name)
    
    successful_plots = _plot_multi_country_series(
        df, countries, "Real Effective Exchange Rate", "Real Effective Exchange Rate", 
        title, colors, markers
    )
    
    # place legend outside the plot area to the right
    if successful_plots > 0:
        plt.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, 
                  fontsize=legend_fontsize, framealpha=0.9)
    
    # adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # make room for legend on the right
    
    _save_or_show_plot(plt, output_dir, save_name, fig_suffix, show_plot)
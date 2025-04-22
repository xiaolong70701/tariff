import os
import pandas as pd
from tqdm import tqdm
from src.analysis import compute_trade_imbalance_ratio, filter_valid_countries
from src.viz import *

def load_small_states(file_path):
    """
    load a list of small states from a text file
    
    args:
        file_path: path to the text file containing small states names
        
    returns:
        list of small states names
    """
    with open(file_path, "r") as f:
        return [country.strip() for country in f.readlines()]


def generate_country_charts(df_filtered, country, output_dir, fig_suffix="pdf"):
    """
    generate individual country charts for trade imbalance and exchange rate
    
    args:
        df_filtered: filtered dataframe containing macroeconomic data
        country: country name to generate charts for
        output_dir: directory to save the charts
        fig_suffix: file extension for the charts
        
    returns:
        None
    """
    try:
        country_code = get_alpha_3(country)
        
        # define chart filenames
        gdp_imb_name = f"Trade_Imbalance_{country_code}"
        imb_name = f"Trade_Imbalance_Raw_{country_code}"
        eff_name = f"REER_{country_code}"
        
        # generate charts
        plot_ctry_gdp_imb_series(
            df_filtered, 
            output_dir=output_dir, 
            country=country, 
            save_name=gdp_imb_name, 
            show_plot=False, 
            fig_suffix=fig_suffix
        )
        plot_ctry_imb_series(
            df_filtered, 
            output_dir=output_dir, 
            country=country, 
            save_name=imb_name, 
            show_plot=False, 
            fig_suffix=fig_suffix
        )
        plot_ctry_eff_series(
            df_filtered, 
            output_dir=output_dir, 
            country=country, 
            save_name=eff_name, 
            show_plot=False, 
            fig_suffix=fig_suffix
        )
        return True
    except Exception as e:
        tqdm.write(f"error generating chart for {country}: {e}")
        return False


def generate_multi_country_charts(df_filtered, countries_list, output_dir, fig_suffix="pdf"):
    """
    generate multi-country comparison charts for trade imbalance and exchange rate
    
    args:
        df_filtered: filtered dataframe containing macroeconomic data
        countries_list: list of countries to include in the charts
        output_dir: directory to save the charts
        fig_suffix: file extension for the charts
        
    returns:
        None
    """
    # generate trade imbalance multi-country chart
    plot_multi_ctry_imb_series(
        df_filtered, 
        countries_list,
        output_dir=output_dir, 
        save_name="Major_Economies_Trade_Imbalance", 
        title="Trade Imbalance Comparison: Major Economies (2013-2023)",
        fig_suffix=fig_suffix
    )
    
    # generate real effective exchange rate multi-country chart
    plot_multi_ctry_eff_series(
        df_filtered,
        countries_list,
        output_dir=output_dir, 
        save_name="Major_Economies_REER", 
        title="Real Effective Exchange Rate Comparison: Major Economies (2013-2023)",
        fig_suffix=fig_suffix
    )


def main():
    """main function to process data and generate visualizations"""
    # define directory paths
    PROCESSED_DIR = "./data/processed/"
    RAW_DIR = "./data/raw"
    OUTPUT_DIR = "./output"
    FIG_FORMAT = "pdf"  # use pdf for publication quality
    
    # ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # load and process data
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "macro_data_merged.csv"))
    small_states = load_small_states(os.path.join(RAW_DIR, "small_countries.txt"))
    
    # calculate trade imbalance ratio
    df = compute_trade_imbalance_ratio(df)
    
    # filter for recent decade (2013-2023)
    df_filtered = df[(df['Year'] >= 2013) & (df['Year'] <= 2023)]
    
    # calculate country averages
    avg_df = df_filtered.groupby("Country Name").agg({
        "Trade Imbalance": "mean",
        "Trade Imbalance to GDP Ratio": "mean",
        "Real Effective Exchange Rate": "mean"
    }).reset_index()

    # save processed data
    avg_df.to_csv(os.path.join(OUTPUT_DIR, "average_macro_data.csv"), 
                 encoding="utf-8-sig", index=False)

    # filter out small states
    filtered_df = filter_valid_countries(avg_df, small_states)

    # create global visualization of REER vs Trade Imbalance
    global_chart_name = "REER_vs_Trade_Imbalance_Global"
    plot_rer_imb(
        filtered_df, 
        annotate=True, 
        output_dir=OUTPUT_DIR, 
        save_name=global_chart_name, 
        fig_suffix=FIG_FORMAT
    )
    
    # list of countries for individual analysis
    countries_list = [
        "Taiwan",
        "China",
        "Hong Kong",
        "South Korea",
        "Japan",
        "Vietnam",         
        "Malaysia",
        "Indonesia",
        "Singapore",
        "United States",
        "United Kingdom",
        "Germany",
        "France",
        "Canada",
        "Mexico"
    ]
    
    # generate multi-country comparison charts
    generate_multi_country_charts(df_filtered, countries_list, OUTPUT_DIR, FIG_FORMAT)
    plot_neglog_plgdpo_vs_imb(
        df, countries_list, output_dir="./output",
        save_name="NegLogPLGDPO_vs_Imb_2009_2019", fig_suffix="pdf", point_alpha= 0,
    )

    # create individual country visualizations with progress bar
    successful_charts = 0
    with tqdm(total=len(countries_list), desc="Processing countries") as pbar:
        for country in countries_list:
            result = generate_country_charts(df_filtered, country, OUTPUT_DIR, FIG_FORMAT)
            if result:
                successful_charts += 1
            pbar.update(1)
    
    print(f"Chart generation complete. Successfully processed {successful_charts}/{len(countries_list)} countries.")


if __name__ == "__main__":
    main()
import os
import pandas as pd
from src.analysis import compute_trade_imbalance_ratio, filter_valid_countries
from src.viz import plot_rer_vs_imbalance
from src.viz import plot_taiwan_imbalance_time_series

if __name__ == "__main__":
    PROCESSED_DIR = "./data/processed/"
    RAW_DIR = "./data/raw"
    OUTPUT_DIR = "./output"
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "macro_data_merged.csv"))
    
    small_states = []
    with open(os.path.join(RAW_DIR, "small_countries.txt"), "r") as f:
        countries = f.readlines()
        for country in countries:
            small_states.append(country.strip())

    df = compute_trade_imbalance_ratio(df)
    # df = compute_rer(df)
    df_filtered = df[(df['Year'] >= 2013) & (df['Year'] <= 2023)]
    
    avg_df = df_filtered.groupby("Country Name").agg({
        "Trade Imbalance to GDP Ratio": "mean",
        "Real Effective Exchange Rate": "mean"
    }).reset_index()

    avg_df.to_csv(os.path.join(OUTPUT_DIR, "average_macro_data.csv"), encoding="utf-8-sig", index=False)

    filtered_df = filter_valid_countries(avg_df, small_states)

    # plot
    reer_vs_imbalance_name = "Real_Effective_Exchange_Rate_vs_Trade_Imbalance_to_GDP_Ratio_Global"
    taiwan_imbalance_ts_name = "Real_Effective_Exchange_Rate_vs_Trade_Imbalance_to_GDP_Ratio_TW"
    plot_rer_vs_imbalance(filtered_df, annotate=True, output_dir=OUTPUT_DIR, save_name=reer_vs_imbalance_name, fig_suffix="pdf")
    plot_taiwan_imbalance_time_series(df_filtered, output_dir=OUTPUT_DIR, save_name=taiwan_imbalance_ts_name, show_plot=False, fig_suffix="pdf")

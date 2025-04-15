import os
import pandas as pd
from src.dataprocessor import DataProcessor

if __name__ == "__main__":
    MACRO_DATA_DIR = "./data/raw/macro"
    OUTPUT_DIR = "./data/processed"

    # read all macro data
    cpi_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "CPI_by_country.csv"))
    gdp_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "GDP_by_country.csv"))
    trade_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "trade_by_country.csv"))
    # exchange_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "exchangeRate_by_country.csv"))
    eff_exchange_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "eff_exchangeRate_by_country.csv"))
    taiwan_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "Taiwan_data.csv"))

    # process macro data
    processor = DataProcessor(cpi_df, gdp_df, trade_df, eff_exchange_df)
    # convert data format
    processor.melt_and_convert()
    # filter data by CPI
    processor.filter_data_by_cpi()
    # merge data
    final_df = processor.merge_datasets()
    taiwan_df["Country Name"] = "TW"
    taiwan_df = taiwan_df[["Country Name", "Year", "CPI", "GDP", "Trade Imbalance", "Exchange Rate"]]
    final_df = pd.concat([final_df, taiwan_df], ignore_index=True)
    # save data
    final_df.to_csv(os.path.join(OUTPUT_DIR, "macro_data_merged.csv"), encoding="utf-8-sig", index=False)
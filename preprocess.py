import os
import pandas as pd
from src.dataprocessor import DataProcessor

if __name__ == "__main__":
    # define directory paths
    MACRO_DATA_DIR = "./data/raw/macro"
    OUTPUT_DIR = "./data/processed"

    # load macroeconomic datasets
    cpi_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "CPI_by_country.csv"))
    gdp_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "GDP_by_country.csv"))
    trade_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "trade_by_country.csv"))
    eff_exchange_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "eff_exchangeRate_by_country.csv"))
    pwt_df = pd.read_stata(os.path.join(MACRO_DATA_DIR, "pwt1001.dta"))
    taiwan_df = pd.read_csv(os.path.join(MACRO_DATA_DIR, "Taiwan_data.csv"))

    # initialize processor and process macro datasets
    processor = DataProcessor(cpi_df, gdp_df, trade_df, eff_exchange_df, pwt_df)
    processor.melt_and_convert()
    processor.filter_data_by_cpi()
    processor.process_pwt_data()
    final_df = processor.merge_datasets()
    final_df = processor.add_uval_index(final_df)

    # export final dataframe to csv
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df.to_csv(os.path.join(OUTPUT_DIR, "macro_data_merged.csv"), encoding="utf-8-sig", index=False)

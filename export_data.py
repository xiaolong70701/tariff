import os
import pandas as pd

if __name__ == "__main__":
    RAW_DIR = "./data/raw"

    TW_US_export_HS = pd.read_csv(os.path.join(RAW_DIR, "TW_US_Export_Ranking.csv"))
    print(TW_US_export_HS)
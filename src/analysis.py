def compute_trade_imbalance_ratio(df):
    df["Trade Imbalance to GDP Ratio"] = df["Trade Imbalance"] / df["GDP"]
    return df

# def compute_rer(df, base_country="US"):
#     base_cpi = df.loc[df["Country Name"] == base_country, "CPI"].values[0]
#     df["RER"] = (df["Exchange Rate"] * df["CPI"]) / base_cpi
#     return df

def filter_valid_countries(df, exclude_list):
    return df[~df["Country Name"].isin(exclude_list)].dropna()

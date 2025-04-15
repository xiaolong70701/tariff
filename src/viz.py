import os
import matplotlib.pyplot as plt
import pycountry

def get_alpha_3(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return country_name

def plot_rer_vs_imbalance(df, log_scale=True, annotate=True, show_plot=False, point_alpha=0, text_alpha=0.8, fontsize=10, save_name="Average Trade Imbalance vs Average Real Exchange Rate by Country (2013-2023)", fig_suffix="png", output_dir=None):
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

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, save_name + f".{fig_suffix}"), dpi=1200)

    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_taiwan_imbalance_time_series(df, output_dir=None, show_plot=False, save_name="taiwan_trade_imbalance", fig_suffix="png"):
    taiwan_df_filtered = df[df['Country Name'] == "TW"].sort_values("Year")
    
    plt.figure(figsize=(10, 5))
    plt.plot(
        taiwan_df_filtered["Year"], 
        taiwan_df_filtered["Trade Imbalance to GDP Ratio"],
        marker='o', color='orange', linewidth=2
    )

    plt.title("Taiwan: Trade Imbalance to GDP Ratio (2013–2023)")
    plt.xlabel("Year")
    plt.ylabel("Trade Imbalance / GDP")
    plt.grid(True)
    plt.xticks(taiwan_df_filtered["Year"])
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, save_name + f".{fig_suffix}"), dpi=1200)

    if show_plot:
        plt.show()
    else:
        plt.close()

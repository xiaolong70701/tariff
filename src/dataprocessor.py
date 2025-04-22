import pycountry
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import statsmodels.api as sm

class DataProcessor:
    def __init__(self, cpi_df, gdp_df, trade_df, exchange_df, pwt_df):
        self.cpi_df = cpi_df
        self.gdp_df = gdp_df
        self.trade_df = trade_df
        self.exchange_df = exchange_df
        self.pwt_df = pwt_df

        self.special_regions = {
            "Hong Kong": "HK", "Taiwan": "TW", "Macao": "MO",
            "Korea, Republic of": "KR", "Korea, South": "KR", "South Korea": "KR",
            "Korea, Democratic People's Republic of": "KP", "North Korea": "KP",
            "Viet Nam": "VN", "Vietnam": "VN"
        }

    @staticmethod
    def convert_country_column(df, country_column):
        def get_country_alpha2(country_name):
            special_regions = {
                "Hong Kong": "HK", "Taiwan": "TW", "Macao": "MO",
                "Korea, Republic of": "KR", "Korea, South": "KR", "South Korea": "KR",
                "Korea, Democratic People's Republic of": "KP", "North Korea": "KP",
                "Viet Nam": "VN", "Vietnam": "VN"
            }
            for region_name, code in special_regions.items():
                if region_name.lower() in country_name.lower():
                    return code
            for attr in ['name', 'official_name', 'common_name']:
                try:
                    country = pycountry.countries.get(**{attr: country_name})
                    if country:
                        return country.alpha_2
                except (AttributeError, KeyError):
                    pass
            try:
                country_names = [c.name for c in pycountry.countries]
                matched_country, score = process.extractOne(country_name, country_names)
                if score >= 75:
                    return pycountry.countries.get(name=matched_country).alpha_2
            except Exception:
                pass
            return country_name

        df_copy = df.copy()
        df_copy[country_column] = df_copy[country_column].apply(get_country_alpha2)
        df_copy = df_copy[df_copy[country_column].notna()]
        return df_copy

    def melt_and_convert(self):
        self.cpi_df = self.convert_country_column(self.cpi_df, "COUNTRY")
        self.cpi_df = self.cpi_df.rename(columns={"COUNTRY": "Country Name"})
        self.cpi_df_melted = self.cpi_df.melt(id_vars=["Country Name"], var_name="Year", value_name="CPI")

        self.gdp_df = self.convert_country_column(self.gdp_df, "Country Name")
        self.gdp_df_melted = self.gdp_df.melt(id_vars=["Country Name"], var_name="Year", value_name="GDP")

        self.trade_df = self.convert_country_column(self.trade_df, "Country Name")
        self.trade_df_melted = self.trade_df.melt(id_vars=["Country Name"], var_name="Year", value_name="Trade Imbalance")

        self.exchange_df = self.convert_country_column(self.exchange_df, "Country Name")
        self.exchange_melted = self.exchange_df.melt(id_vars=["Country Name"], var_name="Year", value_name="Real Effective Exchange Rate")

        for df in [self.cpi_df_melted, self.gdp_df_melted, self.trade_df_melted, self.exchange_melted]:
            df["Year"] = pd.to_numeric(df["Year"], errors='coerce')

        self._log_country_counts()

    def _log_country_counts(self):
        datasets = {
            "CPI": self.cpi_df_melted,
            "GDP": self.gdp_df_melted,
            "Trade": self.trade_df_melted,
            "Exchange": self.exchange_melted
        }
        for name, df in datasets.items():
            countries = set(df["Country Name"].unique())
            print(f"countries in {name}: {len(countries)}")
            for region, code in self.special_regions.items():
                if code in countries:
                    print(f"  - {region} ({code}) found in {name}")

    def filter_data_by_cpi(self):
        countries_in_cpi = set(self.cpi_df_melted["Country Name"].unique())
        print(f"total countries in CPI: {len(countries_in_cpi)}")

        missing_regions = []
        for region, code in self.special_regions.items():
            if code not in countries_in_cpi:
                missing_regions.append(f"{region} ({code})")
        if missing_regions:
            print(f"special regions not found in CPI: {', '.join(missing_regions)}")

        self.gdp_df_filtered = self.gdp_df_melted[self.gdp_df_melted["Country Name"].isin(countries_in_cpi)]
        self.trade_df_filtered = self.trade_df_melted[self.trade_df_melted["Country Name"].isin(countries_in_cpi)]
        self.exchange_df_filtered = self.exchange_melted[self.exchange_melted["Country Name"].isin(countries_in_cpi)]

        for name, df in [
            ("GDP", self.gdp_df_filtered),
            ("Trade", self.trade_df_filtered),
            ("Exchange", self.exchange_df_filtered)
        ]:
            print(f"countries in {name} after filtering: {len(df['Country Name'].unique())}")

    def process_pwt_data(self):
        print("Processing PWT data...")
        id_vars = ['country', 'year']
        df = pd.melt(self.pwt_df, id_vars=id_vars, var_name="variable", value_name="value")
        df = df.rename(columns={'country': 'Country Name', 'year': 'Year'})

        pwt_plgdpo = df[df['variable'] == 'pl_gdpo'][['Country Name', 'Year', 'value']].rename(columns={'value': 'pl_gdpo'})
        pwt_rgdpo  = df[df['variable'] == 'rgdpo'][['Country Name', 'Year', 'value']].rename(columns={'value': 'rgdpo'})
        pwt_pop    = df[df['variable'] == 'pop'][['Country Name', 'Year', 'value']].rename(columns={'value': 'pop'})
        pwt_xr     = df[df['variable'] == 'xr'][['Country Name', 'Year', 'value']].rename(columns={'value': 'xr'})

        self.pwt_processed = pwt_plgdpo \
            .merge(pwt_rgdpo, on=["Country Name", "Year"], how="outer") \
            .merge(pwt_pop, on=["Country Name", "Year"], how="outer") \
            .merge(pwt_xr, on=["Country Name", "Year"], how="outer")

        self.pwt_processed = self.convert_country_column(self.pwt_processed, "Country Name")
        print(f"PWT processed: {self.pwt_processed.shape[0]} rows")

    def merge_datasets(self):
        merged_df = pd.merge(self.cpi_df_melted, self.gdp_df_filtered, how='left', on=["Country Name", "Year"])
        merged_df = pd.merge(merged_df, self.trade_df_filtered, how='left', on=["Country Name", "Year"])
        merged_df = pd.merge(merged_df, self.exchange_df_filtered, how='left', on=["Country Name", "Year"])

        if hasattr(self, "pwt_processed"):
            merged_df = pd.merge(merged_df, self.pwt_processed, how='left', on=["Country Name", "Year"])
            print("✓ Merged with PWT data.")

        unique_countries = merged_df["Country Name"].unique()
        print(f"Total countries in merged dataset: {len(unique_countries)}")

        for region, code in self.special_regions.items():
            status = "retained" if code in unique_countries else "LOST"
            print(f"Special region {status}: {region} ({code})")

        return merged_df

    def add_special_regions_manually(self, merged_df, special_region_df_dict):
        """
        add special regions data manually to the merged dataframe
        
        args:
            merged_df: dataframe with merged economic data
            special_region_df_dict: dict mapping region codes to dataframes
            
        returns:
            updated dataframe with special regions added
        """
        dfs_to_concat = [merged_df]
        added_regions = 0
        
        for region_code, region_df in special_region_df_dict.items():
            if region_df is None or region_df.empty:
                print(f"skipping empty data for {region_code}")
                continue
                
            # ensure correct format for the region dataframe
            formatted_df = self._format_region_dataframe(region_code, region_df)
            if formatted_df is not None:
                dfs_to_concat.append(formatted_df)
                added_regions += 1
        
        # concatenate all dataframes
        final_df = pd.concat(dfs_to_concat, ignore_index=True)
        print(f"added {added_regions} special regions to final dataset")
        return final_df
    
    def _format_region_dataframe(self, region_code, region_df):
        """
        format a special region dataframe to match the merged dataset structure
        
        args:
            region_code: code for the special region
            region_df: dataframe with the region's data
            
        returns:
            formatted dataframe or None if formatting fails
        """
        try:
            df_copy = region_df.copy()
            
            # ensure country name column exists
            if "Country Name" not in df_copy.columns:
                df_copy["Country Name"] = region_code
            
            # check for required columns
            required_columns = ["Country Name", "Year", "CPI", "GDP", "Trade Imbalance", "Real Effective Exchange Rate"]
            missing_columns = set(required_columns) - set(df_copy.columns)
            
            # add missing columns with NaN values
            if missing_columns:
                print(f"adding missing columns for {region_code}: {missing_columns}")
                for col in missing_columns:
                    df_copy[col] = pd.NA
            
            # select only required columns in the correct order
            formatted_df = df_copy[required_columns]
            print(f"added {region_code} data with {len(formatted_df)} rows")
            return formatted_df
            
        except Exception as e:
            print(f"error formatting {region_code} data: {e}")
            return None

    def add_uval_index(self, df):
        df = df.copy()

        # Ensure numeric types
        for col in ["pl_gdpo", "rgdpo", "pop"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Filter valid rows: all values > 0 and not missing
        df = df[(df["pl_gdpo"] > 0) & (df["rgdpo"] > 0) & (df["pop"] > 0)].copy()

        # === Step 1: compute log(RER) and log(GDP per capita) ===
        df["log_rer"] = -np.log(df["pl_gdpo"])
        df["log_rgdpo_pc"] = np.log(df["rgdpo"] / df["pop"])

        # === Step 2: fit regression: log(RER) ~ log(GDP per capita) ===
        X = sm.add_constant(df["log_rgdpo_pc"])
        y = df["log_rer"]
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        print(model.summary())

        # === Step 3: get predicted RER and undervaluation ===
        df["log_rer_hat"] = model.predict(X)
        df["log_uval"] = df["log_rer"] - df["log_rer_hat"]

        return df

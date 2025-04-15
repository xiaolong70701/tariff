import pycountry
import pandas as pd
from fuzzywuzzy import process

class DataProcessor:
    def __init__(self, cpi_df, gdp_df, trade_df, exchange_df):
        """
        Initialize the data processor and set the datasets.
        
        :param cpi_df: CPI dataset
        :param gdp_df: GDP dataset
        :param trade_df: Trade balance dataset
        :param exchange_df: Exchange rate dataset
        """
        self.cpi_df = cpi_df
        self.gdp_df = gdp_df
        self.trade_df = trade_df
        self.exchange_df = exchange_df
    
    @staticmethod
    def convert_country_column(df, country_column):
        """
        Convert the country name column in a DataFrame to ISO 3166-1 alpha-2 codes.
        Uses fuzzy matching to handle inconsistencies in country names.
        
        :param df: DataFrame containing the country name column
        :param country_column: Name of the country column
        :return: Updated DataFrame with country names converted to alpha-2 codes
        """
        def get_country_alpha2(country_name):
            """
            Retrieve ISO 3166-1 alpha-2 code from a country name using exact or fuzzy matching.
            
            :param country_name: Country name
            :return: ISO alpha-2 code, or None if no match is found
            """
            # Exact match
            country = pycountry.countries.get(name=country_name)
            if country:
                return country.alpha_2
            else:
                # Fuzzy match
                matched_country, score = process.extractOne(country_name, [country.name for country in pycountry.countries])
                if score >= 80:  # Matching threshold
                    return pycountry.countries.get(name=matched_country).alpha_2
                return None  # Return None if no match found
        
        df[country_column] = df[country_column].apply(get_country_alpha2)
        return df

    def melt_and_convert(self):
        """
        Convert datasets to long format and process each field.
        """
        # Convert CPI data
        self.cpi_df = self.convert_country_column(self.cpi_df, "COUNTRY")
        self.cpi_df = self.cpi_df.rename(columns={"COUNTRY": "Country Name"})
        self.cpi_df_melted = self.cpi_df.melt(id_vars=["Country Name"], var_name="Year", value_name="CPI")
        
        # Convert GDP data
        self.gdp_df = self.convert_country_column(self.gdp_df, "Country Name")
        self.gdp_df_melted = self.gdp_df.melt(id_vars=["Country Name"], var_name="Year", value_name="GDP")

        # Convert Trade Balance data
        self.trade_df = self.convert_country_column(self.trade_df, "Country Name")
        self.trade_df_melted = self.trade_df.melt(id_vars=["Country Name"], var_name="Year", value_name="Trade Imbalance")

        # Convert Exchange Rate data
        self.exchange_df = self.convert_country_column(self.exchange_df, "Country Name")
        self.exchange_melted = self.exchange_df.melt(id_vars=["Country Name"], var_name="Year", value_name="Real Effective Exchange Rate")
        
        # Convert "Year" column to numeric
        for df in [self.cpi_df_melted, self.gdp_df_melted, self.trade_df_melted, self.exchange_melted]:
            df["Year"] = pd.to_numeric(df["Year"], errors='coerce')

    def filter_data_by_cpi(self):
        """
        Filter all datasets to only include countries present in the CPI dataset.
        """
        countries_in_cpi = self.cpi_df_melted["Country Name"].unique()
        
        # Filter other datasets
        self.gdp_df_filtered = self.gdp_df_melted[self.gdp_df_melted["Country Name"].isin(countries_in_cpi)]
        self.trade_df_filtered = self.trade_df_melted[self.trade_df_melted["Country Name"].isin(countries_in_cpi)]
        self.exchange_df_filtered = self.exchange_melted[self.exchange_melted["Country Name"].isin(countries_in_cpi)]

    def merge_datasets(self):
        """
        Merge all datasets into one unified DataFrame.
        """
        merged_df = pd.merge(self.cpi_df_melted, self.gdp_df_filtered, how='inner', on=["Country Name", "Year"])
        merged_df = pd.merge(merged_df, self.trade_df_filtered, how='inner', on=["Country Name", "Year"])
        merged_df = pd.merge(merged_df, self.exchange_df_filtered, how='inner', on=["Country Name", "Year"])
        return merged_df

#!/usr/bin/env python3
"""
data processing module for macroeconomic analysis
handles country code conversion, data transformation and merging
"""

import pycountry
import pandas as pd
from fuzzywuzzy import process


class DataProcessor:
    def __init__(self, cpi_df, gdp_df, trade_df, exchange_df):
        """
        initialize the data processor with datasets
        
        args:
            cpi_df: consumer price index dataset
            gdp_df: gross domestic product dataset
            trade_df: trade balance dataset
            exchange_df: exchange rate dataset
        """
        self.cpi_df = cpi_df
        self.gdp_df = gdp_df
        self.trade_df = trade_df
        self.exchange_df = exchange_df
        
        # special regions needing custom handling
        self.special_regions = {
            "Hong Kong": "HK",
            "Taiwan": "TW",
            "Macao": "MO",
            "Korea, Republic of": "KR",
            "Korea, South": "KR",
            "South Korea": "KR",
            "Korea, Democratic People's Republic of": "KP", 
            "North Korea": "KP",
            "Viet Nam": "VN",
            "Vietnam": "VN"
        }
    
    @staticmethod
    def convert_country_column(df, country_column):
        """
        convert country names to ISO 3166-1 alpha-2 codes
        
        args:
            df: dataframe containing country names
            country_column: column name holding country names
            
        returns:
            dataframe with converted country codes
        """
        def get_country_alpha2(country_name):
            """
            get alpha-2 code from country name using exact or fuzzy matching
            
            args:
                country_name: name of the country to convert
                
            returns:
                alpha-2 code or original name if no match found
            """
            # handle special regions
            special_regions = {
                "Hong Kong": "HK",
                "Taiwan": "TW",
                "Macao": "MO",
                "Korea, Republic of": "KR",
                "Korea, South": "KR", 
                "South Korea": "KR",
                "Korea, Democratic People's Republic of": "KP",
                "North Korea": "KP",
                "Viet Nam": "VN",
                "Vietnam": "VN"
            }
            
            # check for special region matches
            for region_name, code in special_regions.items():
                if region_name.lower() in country_name.lower():
                    return code
            
            # try exact match with different pycountry attributes
            for attr in ['name', 'official_name', 'common_name']:
                try:
                    country = pycountry.countries.get(**{attr: country_name})
                    if country:
                        return country.alpha_2
                except (AttributeError, KeyError):
                    pass
                
            # try fuzzy matching as last resort
            try:
                country_names = [c.name for c in pycountry.countries]
                matched_country, score = process.extractOne(country_name, country_names)
                if score >= 75:  # lower threshold for better coverage
                    return pycountry.countries.get(name=matched_country).alpha_2
            except Exception:
                pass
                
            # return original name if all matching fails
            return country_name
        
        # create a copy and apply conversion
        df_copy = df.copy()
        df_copy[country_column] = df_copy[country_column].apply(get_country_alpha2)
        
        # filter out null values
        df_copy = df_copy[df_copy[country_column].notna()]
        
        return df_copy

    def melt_and_convert(self):
        """
        convert datasets from wide to long format and standardize country codes
        """
        # process cpi data
        self.cpi_df = self.convert_country_column(self.cpi_df, "COUNTRY")
        self.cpi_df = self.cpi_df.rename(columns={"COUNTRY": "Country Name"})
        self.cpi_df_melted = self.cpi_df.melt(id_vars=["Country Name"], var_name="Year", value_name="CPI")
        
        # process gdp data
        self.gdp_df = self.convert_country_column(self.gdp_df, "Country Name")
        self.gdp_df_melted = self.gdp_df.melt(id_vars=["Country Name"], var_name="Year", value_name="GDP")

        # process trade balance data
        self.trade_df = self.convert_country_column(self.trade_df, "Country Name")
        self.trade_df_melted = self.trade_df.melt(id_vars=["Country Name"], var_name="Year", value_name="Trade Imbalance")

        # process exchange rate data
        self.exchange_df = self.convert_country_column(self.exchange_df, "Country Name")
        self.exchange_melted = self.exchange_df.melt(id_vars=["Country Name"], var_name="Year", value_name="Real Effective Exchange Rate")
        
        # convert year column to numeric in all dataframes
        for df in [self.cpi_df_melted, self.gdp_df_melted, self.trade_df_melted, self.exchange_melted]:
            df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
            
        self._log_country_counts()
    
    def _log_country_counts(self):
        """
        log the number of countries in each dataset for debugging
        """
        datasets = {
            "CPI": self.cpi_df_melted,
            "GDP": self.gdp_df_melted,
            "Trade": self.trade_df_melted,
            "Exchange": self.exchange_melted
        }
        
        for name, df in datasets.items():
            countries = set(df["Country Name"].unique())
            print(f"countries in {name}: {len(countries)}")
            
            # check for special regions
            for region, code in self.special_regions.items():
                if code in countries:
                    print(f"  - {region} ({code}) found in {name}")

    def filter_data_by_cpi(self):
        """
        filter all datasets to only include countries present in cpi dataset
        """
        countries_in_cpi = set(self.cpi_df_melted["Country Name"].unique())
        print(f"total countries in CPI: {len(countries_in_cpi)}")
        
        # check if special regions are in cpi data
        missing_regions = []
        for region, code in self.special_regions.items():
            if code not in countries_in_cpi:
                missing_regions.append(f"{region} ({code})")
                
        if missing_regions:
            print(f"special regions not found in CPI: {', '.join(missing_regions)}")
                
        # filter other datasets
        self.gdp_df_filtered = self.gdp_df_melted[self.gdp_df_melted["Country Name"].isin(countries_in_cpi)]
        self.trade_df_filtered = self.trade_df_melted[self.trade_df_melted["Country Name"].isin(countries_in_cpi)]
        self.exchange_df_filtered = self.exchange_melted[self.exchange_melted["Country Name"].isin(countries_in_cpi)]
        
        # log filtered dataset counts
        for name, df in [
            ("GDP", self.gdp_df_filtered),
            ("Trade", self.trade_df_filtered),
            ("Exchange", self.exchange_df_filtered)
        ]:
            print(f"countries in {name} after filtering: {len(df['Country Name'].unique())}")

    def merge_datasets(self):
        """
        merge all datasets into one unified dataframe
        
        returns:
            merged dataframe with all economic indicators
        """
        # perform left joins to preserve all countries from cpi dataset
        merged_df = pd.merge(self.cpi_df_melted, self.gdp_df_filtered, how='left', on=["Country Name", "Year"])
        merged_df = pd.merge(merged_df, self.trade_df_filtered, how='left', on=["Country Name", "Year"])
        merged_df = pd.merge(merged_df, self.exchange_df_filtered, how='left', on=["Country Name", "Year"])
        
        # log merged data status
        unique_countries = merged_df["Country Name"].unique()
        print(f"total countries in merged dataset: {len(unique_countries)}")
        
        # check for missing special regions
        for region, code in self.special_regions.items():
            status = "retained" if code in unique_countries else "LOST"
            print(f"special region {status} in merged dataset: {region} ({code})")
        
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
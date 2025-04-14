import pycountry
import pandas as pd
from fuzzywuzzy import process

class DataProcessor:
    def __init__(self, cpi_df, gdp_df, trade_df, exchange_df):
        """
        初始化資料處理器，並設置資料集。
        
        :param cpi_df: CPI 資料
        :param gdp_df: GDP 資料
        :param trade_df: 貿易順差資料
        :param exchange_df: 匯率資料
        """
        self.cpi_df = cpi_df
        self.gdp_df = gdp_df
        self.trade_df = trade_df
        self.exchange_df = exchange_df
    
    @staticmethod
    def convert_country_column(df, country_column):
        """
        將 DataFrame 中的國家名稱欄位轉換為 ISO 3166-1 alpha-2 代碼。
        使用模糊匹配以處理部分不一致的國家名稱。
        
        :param df: 包含國家名稱欄位的 DataFrame
        :param country_column: 國家名稱欄位的名稱
        :return: 更新後的 DataFrame，國家名稱欄位已轉換為 alpha-2 代碼
        """
        def get_country_alpha2(country_name):
            """
            根據國家名稱獲取 ISO 3166-1 alpha-2 代碼，若匹配不到則使用模糊匹配。
            
            :param country_name: 國家名稱
            :return: ISO 3166-1 alpha-2 代碼，若找不到則返回 None
            """
            # 精確匹配
            country = pycountry.countries.get(name=country_name)
            if country:
                return country.alpha_2
            else:
                # 模糊匹配
                matched_country, score = process.extractOne(country_name, [country.name for country in pycountry.countries])
                if score >= 80:  # 設置匹配閾值
                    return pycountry.countries.get(name=matched_country).alpha_2
                return None  # 無法匹配則返回 None
        
        df[country_column] = df[country_column].apply(get_country_alpha2)
        return df

    def melt_and_convert(self):
        """
        將資料集轉換為長格式並處理各個欄位。
        """
        # 轉換 CPI 資料
        self.cpi_df = self.convert_country_column(self.cpi_df, "COUNTRY")
        self.cpi_df = self.cpi_df.rename(columns={"COUNTRY": "Country Name"})
        self.cpi_df_melted = self.cpi_df.melt(id_vars=["Country Name"], var_name="Year", value_name="CPI")
        
        # 轉換 GDP 資料
        self.gdp_df = self.convert_country_column(self.gdp_df, "Country Name")
        self.gdp_df_melted = self.gdp_df.melt(id_vars=["Country Name"], var_name="Year", value_name="GDP")

        # 轉換 Trade 資料
        self.trade_df = self.convert_country_column(self.trade_df, "Country Name")
        self.trade_df_melted = self.trade_df.melt(id_vars=["Country Name"], var_name="Year", value_name="Trade Imbalance")

        # 轉換 Exchange 資料
        self.exchange_df = self.convert_country_column(self.exchange_df, "Country Name")
        self.exchange_melted = self.exchange_df.melt(id_vars=["Country Name"], var_name="Year", value_name="Exchange Rate")
        
        # 轉換 "Year" 欄位為數值
        for df in [self.cpi_df_melted, self.gdp_df_melted, self.trade_df_melted, self.exchange_melted]:
            df["Year"] = pd.to_numeric(df["Year"], errors='coerce')

    def filter_data_by_cpi(self):
        """
        根據 CPI 資料中的國家名稱篩選其他資料集。
        """
        countries_in_cpi = self.cpi_df_melted["Country Name"].unique()
        
        # 篩選其他資料集
        self.gdp_df_filtered = self.gdp_df_melted[self.gdp_df_melted["Country Name"].isin(countries_in_cpi)]
        self.trade_df_filtered = self.trade_df_melted[self.trade_df_melted["Country Name"].isin(countries_in_cpi)]
        self.exchange_df_filtered = self.exchange_melted[self.exchange_melted["Country Name"].isin(countries_in_cpi)]

    def merge_datasets(self):
        """
        合併所有資料集。
        """
        merged_df = pd.merge(self.cpi_df_melted, self.gdp_df_filtered, how='inner', on=["Country Name", "Year"])
        merged_df = pd.merge(merged_df, self.trade_df_filtered, how='inner', on=["Country Name", "Year"])
        merged_df = pd.merge(merged_df, self.exchange_df_filtered, how='inner', on=["Country Name", "Year"])
        return merged_df
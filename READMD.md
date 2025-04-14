# Project Overview: Trade Imbalance and Real Exchange Rate Analysis

This project exploits visualization to demonstrate the relationship between trade imbalances and real exchange rates (RER) across countries, with a specific focus on Taiwan's trade with the United States. It integrates multiple macroeconomic datasets—such as GDP, CPI, nominal exchange rates, and international trade flows—to generate country-year panel data, compute derived indicators, and visualize international economic patterns.

## Data Sources

This project draws on a range of international and national macroeconomic datasets spanning the period from 2013 to 2023. The primary data categories include gross domestic product (GDP), consumer price index (CPI), official exchange rates, and international trade balances. For most countries, these indicators are obtained from widely recognized global institutions such as the International Monetary Fund (IMF) and the World Bank, ensuring a high degree of consistency and comparability across countries.

Specifically, CPI data are sourced from the IMFs economic databases, while GDP, exchange rate, and trade statistics are retrieved from the World Bank's development indicators. These sources provide standardized, cross-country data suitable for international macroeconomic analysis.

However, due to Taiwan's unique political status and its exclusion from major international organizations, equivalent macroeconomic data for Taiwan are not available through global datasets. Instead, official data for Taiwan are obtained directly from domestic government institutions. The Directorate-General of Budget, Accounting and Statistics (DGBAS) supplies comprehensive national statistics, including GDP, CPI, and exchange rate data. Taiwan's trade statistics, particularly those concerning exports to the United States, are sourced from the Ministry of Finance's official trade reporting system.


| Dataset Name       | Time Span     | Source        | URL                                                                 |
|--------------------|---------------|---------------|----------------------------------------------------------------------|
| Consumer Price Index (CPI)       | 2013–2023 | International Monetary Fund (IMF) | https://www.imf.org/en/Data |
| Gross Domestic Product (GDP)     | 2013–2023 | World Bank | https://data.worldbank.org/indicator/NY.GDP.MKTP.CD |
| Trade Balance (Exports and Imports) | 2013–2023 | World Bank | https://data.worldbank.org/indicator/NE.RSB.GNFS.CD |
| Exchange Rate (Official)        | 2013–2023 | World Bank | https://data.worldbank.org/indicator/PA.NUS.FCRF |
| Taiwan GDP, CPI, Exchange Rate        | 2013–2023 | Directorate-General of Budget, Accounting and Statistics (DGBAS), Taiwan | https://nstatdb.dgbas.gov.tw/dgbasall/webmain.aspx?sys=100&funid=defjsp |
| Taiwan Trade Statistics               | 2013–2023 | Ministry of Finance, R.O.C. (Taiwan)           | https://web02.mof.gov.tw/njswww/WebMain.aspx?sys=100&funid=defjsptgl |

## Methodology

### Trade Imbalance to GDP Ratio

The trade imbalance of a country represents the difference between the value of its total exports and total imports of goods and services. A positive trade imbalance (also referred to as a trade surplus) indicates that a country exports more than it imports, while a negative trade imbalance (a trade deficit) suggests the opposite.

To make cross-country comparisons more meaningful, the trade imbalance is expressed relative to the size of each country's economy. Specifically, the ratio of trade imbalance to gross domestic product (GDP) is computed for each country and each year as follows:

$$
\text{Trade Imbalance to GDP Ratio} = \dfrac{\text{Trade Imbalance}}{\text{GDP}}
$$

### Real Exchange Rate (RER)

For real exchange rate, there is no direct data from databases such as IMF, World Bank, OECD. To obtain real exchange rate data from open source, it is feasible to apply the formula of real exchange rate, which is defined as follow:
$$
\text{RER} = \dfrac{e_{t} \cdot P^{*}_{t}}{P_{t}}
$$

where $e_{t}$ denotes the nominal exchange rate; $P^{*}_{t}$ the price index of foreign country, and $P_{t}$ the price index of home country. In this case, home country is set to be the United States. For simplicity, the data of price index of countries are replaced by consumer price index (CPI), which is the easiest and plausible data to get from the Internet.

### Filtering and Aggregation

To ensure analytical consistency and mitigate the influence of volatility-driven outliers, the dataset is restricted to cover the years 2013 through 2023 and excludes a predefined list of small states and microeconomies. These small states—typically characterized by limited economic scale, high trade openness, and heightened vulnerability to external shocks—may exhibit disproportionate fluctuations in trade and exchange rate indicators that could obscure broader macroeconomic trends.

The list of excluded countries is based on the World Bank's official classification of Small States and Small States Forum (SSF) members. Specifically, countries identified in the following sources were excluded from the analysis:

- [World Bank: Small States and SSF Members (2024)](https://www.worldbank.org/en/news/statement/2024/06/26/small-states-and-small-states-forum-members)  
- [World Bank: Country Classification for Small States (PDF)](https://pubdocs.worldbank.org/en/922761504726183951/COUNTRY-LINK-Small-States.pdf)

## Visual Outputs

A cross-country scatter plot of average trade imbalance-to-GDP ratio versus average real exchange rate. Annotated plot with ISO alpha-3 country codes. 

![Trade Imbalance vs RER](output/avg_trade_imbalance_vs_rer_2013_2023.png)

A time-series plot of Taiwan's trade imbalance ratio from 2013 to 2023. Export composition charts (optional) showing Taiwan's top export items to the US by value.

![](output/taiwan_trade_imbalance.png)

## Directory Structure

```bash
tariff_project/
├── data/
│   ├── raw/             # Unprocessed data (CPI, GDP, exchange rate, trade, exports)
│   └── processed/       # Cleaned and merged dataset for analysis
├── src/                 # Modularized analysis code (data processing, plotting)
├── scripts/             # Executable scripts (preprocessing, analysis)
├── output/
│   ├── figures/         # Final visualizations
│   └── average_macro_data.csv
├── report/              # Optional LaTeX report and compiled PDF
```

## Summary

This project serves as a flexible and extensible framework for conducting macroeconomic trade analysis across countries. By integrating trade data with price-level-adjusted exchange rates and focusing on a specific country case (Taiwan), it provides both broad cross-sectional insights and focused time-series observations. The modular structure of the code allows future researchers or analysts to adapt the workflow for extended datasets, policy evaluation, or academic inquiry.
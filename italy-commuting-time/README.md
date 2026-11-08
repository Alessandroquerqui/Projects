# Commuting Time and Transportation Modes in Italy (2001–2024)

**Solo project** – Big Data for Official Statistics, MSc in Data Science, Sapienza University of Rome

**Questions:** how long do Italians take to get to work, how has this changed since 2001, and how much do type of area, means of transport and income explain long commutes?

## Data

Two ISTAT datasets:
- *Spostamenti per andare a lavoro* – commuting time and means of transport by region and type of municipality, 2001–2024 (2004 missing) – [link](INSERT_ISTAT_LINK_1)
- *Reddito disponibile delle famiglie nelle regioni italiane* – household disposable income by region, up to 2023 – [link](INSERT_ISTAT_LINK_2)

## Approach

- **Cleaning and integration:** cleaned the raw ISTAT tables and built shares by means of transport (car, public transport, motorbike, walking/cycling). Split into an urban-area dataset and a regional dataset merged with income.
- **Clustering:** K-means on four mobility features (car use, public transport use, active mobility, long commutes); 4 clusters chosen with the elbow method.
- **Random Forest:** ranked which types of area best predict long commutes (> 30 min) and short commutes (< 15 min).
- **Linear regression:** effect of means of transport and regional income on commuting time, also over time.
- **Logistic regression:** classification of "critical" areas and regions with commuting times above the national median.

## Key findings

- **Small towns look alike:** all urban areas under 50,000 inhabitants show almost identical mobility behaviour, so ISTAT's finer distinctions for small municipalities add little information on commuting.
- **Peripheries are worse off:** the periphery of metropolitan areas is the strongest predictor of long commutes, and it has become worse over time.
- **Public transport is slower:** it is strongly associated with commutes over 30 minutes, which quantifies its time cost (waiting, transfers).
- **Motorbikes save time:** they turn long car commutes into medium-length ones, probably by filtering through traffic.
- **Income and distance:** higher regional income goes with longer commutes, consistent with wealthier households moving out of city centres.
- **Prediction:** the models classify critical areas with 86–91% accuracy. Long commutes are predicted well from the region alone (86%), short ones much less well (68%): proximity depends on neighbourhood design rather than region.

## Files

- [Google Colab Notebook](progetto_BDOS.ipynb)
- [Project Presentation](Commuting%20Time%20and%20Transportation%20Modes%20in%20Italy.pdf)

## Tools

Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn

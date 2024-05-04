# Data-Science-Project_01_Roadcrisis_prevention_analysis

![image](https://github.com/Chu-c-git/Data-Science-Project_01_Roadcrisis_prevention_analysis/assets/141092596/a99e1bde-9662-40a7-81cf-85b253d76c74)

## Introduction
This analysis project sought to identify the key contributing factors that lead to road crises, and make potential risk map for prevention. By examining architectural and urban planning aspects, the project raised a series of critical questions. To address these questions, various open data sources were utilized. Additionally, the project aimed to develop predictive models to anticipate the occurrence of road crises.
![image](https://github.com/Chu-c-git/Data-Science-Project_01_Roadcrisis_prevention_analysis/blob/main/Visualization/Hotspot_map.gif)

## Feature

- **Open data ETL** | Integrated 23 datasets from 4 opendata platform into different training features and analysis. The format of datasets includes csv, xml, shp, geojson and geopackage. A Selenium crawler to collect building cases information wich used to analyze the possibility that sinkhole case happened next to a building case.
- **Data preprocess** | Process spatial and time-series training features based on different geographic scale. (eg. Case location / Villages / 5m Hexes across Taipei City)
- **Model training** | Using XGBoost and LightGBM to find out important features.
- **Visualization** | Using Tableau, QGIS and also matplotlib to demonstrate important findings.

[Crawler Demo](https://reurl.cc/0v6X1Y) | [Open data List](https://github.com/Chu-c-git/Data-Science-Project_01_Roadcrisis_prevention_analysis/blob/main/Opendata_list.xlsx) |

## Quick Start
1. install python 3.X and anaconda
2. create environment using anaconda and [Yaml file](https://github.com/Chu-c-git/Data-Science-Project_01_Roadcrisis_prevention_analysis/blob/main/practice_02_environment.yml).
   ```
   conda env create --file environment_name.yaml
   ```
4. Run related ipynb to make your own training data.

## Tools
| Tool | Description |
|---|---|
| Selenium | Automated web information crawling |
| Dask | Parallel processing for Python |
| Geopandas | Geographic data analysis in Python |
| XGBoost | Scalable and efficient gradient boosting machine learning |
| LightGBM | Gradient boosting machine learning with high performance |
| Optuna | Hyperparameter optimization library for machine learning |
| QGIS | Spatial visualization |

# Taxi Demand Prediction around Central Park - README

## Introduction

This repository contains the source code and data for a project that aims to predict taxi demand around Central Park using time series data. The project focuses on converting the time series data into a tabular format and utilizing it to predict the number of pickups for a given hour based on the previous hour's data.

### Table of Contents

1. [Introduction](#introduction)
2. [Data](#data)
3. [Notebooks](#notebooks)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Data

The data folder contains two subdirectories: `raw` and `transformed`.

- `raw`: This directory contains raw data files in Parquet format representing taxi rides for each month of the year 2022. The files are named as follows: `rides_2022-MM.parquet`, where `MM` represents the month.

- `transformed`: This directory contains transformed data files in Parquet format, including the final tabular data used for modeling (`tabular_data.parquet`). Additionally, there are intermediate files generated during the data transformation process.

- `taxi_zones.csv`: is required for the geolocation of the taxi zones. 

## Notebooks

The `notebooks` directory consists of Jupyter notebooks used for different stages of the project:

1. `00_functions.ipynb`: A notebook containing custom functions used throughout the project.

2. `01_load_and_validate_raw_data.ipynb`: A notebook for loading and validating the raw data.

3. `02_transform_raw_data_to_time_series.ipynb`: A notebook that transforms raw data into time series format.

4. `03_time_series_data.ipynb`: A notebook exploring and analyzing time series data.

5. `04_transform_raw_data_into_features_and_targets.ipynb`: A notebook responsible for feature engineering.

6. `05_visualize_training_data.ipynb`: A notebook used to visualize the training data.

7. `06_baseline_model.ipynb`: A notebook implementing a baseline model for prediction.

8. `07_XGBoost_model.ipynb`: A notebook presenting an XGBoost model for prediction.

9. `08_catboost.ipynb`: A notebook demonstrating the CatBoost model.

10. `09_catboost_model_with_feature_engineering.ipynb`: A notebook combining CatBoost with feature engineering.

11. `10_catboost_with_hyperparameter_tuning.ipynb`: A notebook for hyperparameter tuning of CatBoost.

The `catboost_info` directory stores additional files related to CatBoost training.

## Installation

To set up the environment for this project, you can use `poetry` to install the required dependencies. Use the provided `pyproject.toml` and `poetry.lock` files to manage dependencies. Run the following command to create the environment:

```
poetry install
```

## Usage

After installing the required dependencies, you can use the Jupyter notebooks in the `notebooks` directory to explore the project and run the code cells sequentially.

## Contributing

Contributions to this project are welcome! If you find any issues or have ideas for improvements, please open an issue or submit a pull request to contribute.


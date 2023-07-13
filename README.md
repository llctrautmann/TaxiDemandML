Sure! Here's an updated version of the README.md file with the additional information:

# Taxi Demand Prediction around Central Park

This repository contains the source code and data for a project that aims to predict taxi demand around Central Park using time series data. The project focuses on converting the time series data into tabular format and utilizing it to predict the number of pickups for a given hour based on the previous hour's data.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Notebooks](#notebooks)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The purpose of this project is to develop a predictive model for taxi demand around Central Park based on historical time series data. By converting the time series data into a tabular format, each row represents a pickup event within a year, and each column represents the number of pickups for a previous hour. This tabular representation allows for easier analysis and prediction using various machine learning techniques.

## Data

The `data` directory contains the raw and transformed data used in the project.

### Raw Data

The `data/raw` directory stores the raw data files in Parquet format. Each file corresponds to rides data for a specific month in the year 2022.

- `rides_2022-01.parquet`
- `rides_2022-02.parquet`
- `rides_2022-03.parquet`
- `rides_2022-04.parquet`
- `rides_2022-05.parquet`
- `rides_2022-06.parquet`
- `rides_2022-07.parquet`
- `rides_2022-08.parquet`
- `rides_2022-09.parquet`
- `rides_2022-10.parquet`
- `rides_2022-11.parquet`
- `rides_2022-12.parquet`

The data is __NOT__ included in the repository.

### Transformed Data

The `data/transformed` directory contains the transformed data obtained from the raw data.
- `tabular_data.parquet`: Tabular representation of the transformed data. Each row represents a pickup event within a year, and each column represents the number of pickups for a previous hour. (not included in the repository)

## Notebooks

The `notebooks` directory contains Jupyter notebooks that demonstrate the project workflow and analysis.

- `00_functions.ipynb`: Notebook with utility functions used throughout the project.
- `01_load_and_validate_raw_data.ipynb`: Notebook for loading and validating the raw data.
- `02_transform_raw_data_to_time_series.ipynb`: Notebook for transforming the raw data into time series format.
- `03_time_series_data.ipynb`: Notebook for exploring and analyzing the time series data.
- `04_transform_raw_data_into_features_and_targets.ipynb`: Notebook for transforming the raw data into features and targets.
- `05_visualize_training_data.ipynb`: Notebook for visualizing the training data.

## Installation

To install the project dependencies, use the Poetry package manager. The necessary information for installation is provided in the `pyproject.toml` file.

1. Ensure Poetry is installed on your system. If not, follow the [official Poetry installation guide](https://python-poetry.org/docs/#installation) to install it.
2. Navigate to the project root directory.
3. Run the following command to install the dependencies:

```bash
poetry install
```

This will create a virtual environment and install all the required packages.

## Usage

[Explain how to use the project, including any necessary steps or commands. Provide examples if applicable.]

## Contributing

[Specify how others can contribute to the project. Include guidelines for submitting issues or pull requests.]

## License

[Specify the license under which the project is distributed. Include any necessary disclaimers or terms of use.]

Feel free to modify the content as per your project's requirements.

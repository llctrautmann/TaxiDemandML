from pathlib import Path
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
from datetime import timedelta
from typing import Optional, List
from paths import *

def download_file(year: int , month: int) -> Path:
    """
    download the raw data files from src
    
    """
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"

        open(path, 'wb').write(response.content)
        return path
    else:
        raise Exception(f"Failed to download {URL}")
    

    
def validate_raw_data(
        rides: pd.DataFrame,
        month: int,
        year: int) -> pd.DataFrame:
    '''
    Removes faulty data from the raw dataframe if the date are outside their valid range
    '''

    start_month_date = f'{year}-{month:02d}-01'
    end_month_date = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'

    rides = rides[rides.pickup_datetime >= start_month_date]
    rides = rides[rides.pickup_datetime < end_month_date]

    return rides



def load_raw_data(
        year: int,
        months: Optional[List[int]] = None # Explanation of Optional https://stackoverflow.com/a/51710151
        ) -> pd.DataFrame:
        '''
        Automated function that completes most of the data preprocessing for the raw dataframes
        '''

        rides = pd.DataFrame()

        if months is None:
             months = list(range(1,13))
        elif isinstance(months, int):
            months = [months]

        for month in months:

            local_file = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"

            if not local_file.exists():
                 try:
                    print(f'Downloading raw data file: rides_{year}-{month:02d}')
                    download_file(year,month)
                 
                 except:
                    print(f'rides_{year}-{month:02d} is not available.')
                    continue
            else:
                print('file is already downloaded and in the data directory.')


            # load file into pandas data frame
            rides_single_month = pd.read_parquet(local_file)



            # select columns to keep and rename columns to desired lable
            rides_single_month = rides_single_month[["tpep_pickup_datetime","PULocationID"]]

            rides_single_month.rename(columns = {"tpep_pickup_datetime":"pickup_datetime",
                                                "PULocationID": "pickup_location_id"
                                                },inplace=True)


            # Remove invalid entires from the dataframe

            valid_single_month = validate_raw_data(rides_single_month,month=month,year=year)

            # append month to existing dataframe rides

            rides = pd.concat([rides, valid_single_month])

        rides = rides[["pickup_datetime","pickup_location_id"]]

        return rides


def add_missing_slots(agg_rides: pd.DataFrame) -> pd.DataFrame:
    '''
    https://stackoverflow.com/questions/19324453/add-missing-dates-to-pandas-dataframe/19324591#19324591
    '''
    location_ids = agg_rides['pickup_location_id'].unique()
    full_range = pd.date_range(
        agg_rides['pickup_hour'].min(), agg_rides['pickup_hour'].max(), freq='H')
    output = pd.DataFrame()

    for location_id in tqdm(location_ids):

        agg_rides_i = agg_rides.loc[agg_rides.pickup_location_id == location_id, ['pickup_hour','rides']]

        agg_rides_i.set_index('pickup_hour', inplace=True)
        agg_rides_i.index = pd.DatetimeIndex(agg_rides_i.index)
        agg_rides_i = agg_rides_i.reindex(full_range, fill_value=0)

        agg_rides_i['pickup_location_id'] = location_id

        output = pd.concat([output, agg_rides_i])

    output = output.reset_index().rename(columns={"index": 'pickup_hour'})

    return output



def get_cutoff_indices(
    data: pd.DataFrame,
    n_features: int,
    step_size: int
    ) -> list:

        stop_position = len(data) - 1
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        subseq_mid_idx = n_features
        subseq_last_idx = n_features + 1
        indices = []
        
        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
            
            subseq_first_idx += step_size
            subseq_mid_idx += step_size
            subseq_last_idx += step_size

        return indices


def transform_raw_data_into_ts_data(
    rides: pd.DataFrame
) -> pd.DataFrame:
    """"""
    # sum rides per location and pickup_hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0: 'rides'}, inplace=True)

    # add rows for (locations, pickup_hours)s with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots


def transform_ts_data_into_features_and_target(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
) -> pd.DataFrame:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models
    """
    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        
        # keep only ts data for this `location_id`
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id, 
            ['pickup_hour', 'rides']
        ]

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices(
            ts_data_one_location,
            input_seq_len,
            step_size
        )

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy -> pandas
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # numpy -> pandas
        targets_one_location = pd.DataFrame(y, columns=[f'target_rides_next_hour'])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']


def plot_one_sample(
    example_id: int,
    features: pd.DataFrame,
    targets: Optional[pd.Series] = None,
    predictions: Optional[pd.Series] = None
):
    """"""
    features_ = features.iloc[example_id]
    
    if targets is not None:
        target_ = targets.iloc[example_id]
    else:
        target_ = None
    
    ts_columns = [c for c in features.columns if c.startswith('rides_previous_')]
    ts_values = [features_[c] for c in ts_columns] + [target_]
    ts_dates = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
        features_['pickup_hour'],
        freq='H'
    )
    
    # line plot with past values
    title = f'Pick up hour={features_["pickup_hour"]}, location_id={features_["pickup_location_id"]}'
    fig = px.line(
        x=ts_dates, y=ts_values,
        template='plotly_dark',
        markers=True, title=title
    )
    
    if targets is not None:
        # green dot for the value we wanna predict
        fig.add_scatter(x=ts_dates[-1:], y=[target_],
                        line_color='green',
                        mode='markers', marker_size=10, name='actual value') 
        
    if predictions is not None:
        # big red X for the predicted value, if passed
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(x=ts_dates[-1:], y=[prediction_],
                        line_color='red',
                        mode='markers', marker_symbol='x', marker_size=15,
                        name='prediction')             
    return fig


def plot_ts(
    ts_data: pd.DataFrame,
    locations: Optional[List[int]] = None
    ):
    """
    Plot time-series data
    """
    ts_data_to_plot = ts_data[ts_data.pickup_location_id.isin(locations)] if locations else ts_data

    fig = px.line(
        ts_data,
        x="pickup_hour",
        y="rides",
        color='pickup_location_id',
        template='none',
    )

    fig.show()
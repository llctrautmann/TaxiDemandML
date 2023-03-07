from pathlib import Path
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px

from typing import Optional, List
from taxiML.paths import *

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

    rides = rides[rides.pickup_time >= start_month_date]
    rides = rides[rides.pickup_time < end_month_date]

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

            rides_single_month.rename(columns = {"tpep_pickup_datetime":"pickup_time",
                                                "PULocationID": "pickup_location"
                                                },inplace=True)


            # Remove invalid entires from the dataframe

            valid_single_month = validate_raw_data(rides_single_month,month=month,year=year)

            # append month to existing dataframe rides

            rides = pd.concat([rides, valid_single_month])

        rides = rides[["pickup_time","pickup_location"]]

        return rides


def add_missing_slots(agg_rides: pd.DataFrame) -> pd.DataFrame:
    '''
    https://stackoverflow.com/questions/19324453/add-missing-dates-to-pandas-dataframe/19324591#19324591
    '''
    location_ids = agg_rides['pickup_location'].unique()
    full_range = pd.date_range(
        agg_rides['pickup_hour'].min(), agg_rides['pickup_hour'].max(), freq='H')
    output = pd.DataFrame()

    for location_id in tqdm(location_ids):

        agg_rides_i = agg_rides.loc[agg_rides.pickup_location == location_id, ['pickup_hour','rides']]

        agg_rides_i.set_index('pickup_hour', inplace=True)
        agg_rides_i.index = pd.DatetimeIndex(agg_rides_i.index)
        agg_rides_i = agg_rides_i.reindex(full_range, fill_value=0)

        agg_rides_i['pickup_location'] = location_id

        output = pd.concat([output, agg_rides_i])

    output = output.reset_index().rename(columns={"index": 'pickup_hour'})

    return output



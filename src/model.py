import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline

import catboost as cb
import xgboost as xgb
import numpy as np

def get_coordinates(filepath='../data/taxi_zones.csv'):
    df = pd.read_csv(filepath_or_buffer=filepath)
    df = df[['the_geom', 'zone', 'LocationID']]
    df = df.rename(columns={'LocationID': 'pickup_location_id'})
    lat_list = []
    long_list = []

    for i in range(0,263):
        l = df['the_geom'][i][16:].replace(',','').replace('(','').replace(')','').split()

        long = [float(l[j]) for j in range(len(l)) if j % 2 == 0]
        lat = [float(l[j]) for j in range(len(l)) if j % 2 != 0]
        mean_lat = np.mean([max(lat),min(lat)])
        mean_long = np.mean([max(long),min(long)])

        lat_list.append(mean_lat)
        long_list.append(mean_long)

    coords = pd.DataFrame(list(zip(lat_list, long_list)),columns=['latitude', 'longitude']) 

    df = pd.concat([df[['pickup_location_id']], coords],axis=1)

    return df



def add_coordinates(df: pd.DataFrame):
    coord_df = get_coordinates()
    df = df.merge(coord_df, on='pickup_location_id', how='left')
    # df.drop(columns=['pickup_location_id'],inplace=True)

    df['latitude'] = df['latitude'].fillna(40.730610) # general average value for NYC
    df['longitude'] = df['longitude'].fillna(-73.935242) # general average value for NYC

    return df


def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rides from
    - 7 days ago
    - 14 days ago
    - 21 days ago
    - 28 days ago
    """
    X['average_rides_last_4_weeks'] = 0.25*(
        X[f'rides_previous_{7*24}_hour'] + \
        X[f'rides_previous_{2*7*24}_hour'] + \
        X[f'rides_previous_{3*7*24}_hour'] + \
        X[f'rides_previous_{4*7*24}_hour']
    )
    return X


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn data transformation that adds 2 columns
    - hour
    - day_of_week
    and removes the `pickup_hour` datetime column.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        
        # Generate numeric columns from datetime
        X_["hour"] = X_['pickup_hour'].dt.hour
        X_["day_of_week"] = X_['pickup_hour'].dt.dayofweek
        
        return X_.drop(columns=['pickup_hour'])

def get_pipeline(**hyperparams) -> Pipeline:

    add_latitude_longitude = FunctionTransformer(
        add_coordinates, validate=False
    )

    # sklearn transform
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks, validate=False)
    
    # sklearn transform
    add_temporal_features = TemporalFeaturesEngineer()

    # sklearn pipeline
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        # add_latitude_longitude,
        cb.CatBoostRegressor(**hyperparams),
        # xgb.XGBRegressor(**hyperparams)
    )
# Import standard libraries
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import extra modules
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
RANDOM_STATE = 42

CAT_COLS = ['merch', 'cat_id', 'name_1', 'name_2',
           'gender', 'street', 'one_city', 'us_state', 'post_code', 'jobs']

def add_time_features(df):
    logger.debug('Adding time features...')
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['year'] = df.transaction_time.dt.year
    df['month'] = df.transaction_time.dt.month
    df['day'] = df.transaction_time.dt.day
    df['hour'] = df.transaction_time.dt.hour
    df['weekday'] = df.transaction_time.dt.weekday
    df.drop(columns='transaction_time', inplace=True)
    return df

def add_cartesian_features(df):
    logger.debug('Calculating cartesian...')
    R = 6378  # радиус земли
    df['x'] = R * np.cos(np.radians(df.lat)) * np.cos(np.radians(df.lon))
    df['y'] = R * np.cos(np.radians(df.lat)) * np.sin(np.radians(df.lon))
    df['z'] = R * np.sin(np.radians(df.lat))
    df['x_m'] = R * np.cos(np.radians(df.merchant_lat)) * np.cos(np.radians(df.merchant_lon))
    df['y_m'] = R * np.cos(np.radians(df.merchant_lat)) * np.sin(np.radians(df.merchant_lon))
    df['z_m'] = R * np.sin(np.radians(df.merchant_lat))
    return df.drop(columns=['lat', 'lon', 'merchant_lat', 'merchant_lon'])

def load_train_data():
    """Load and preprocess training data"""
    logger.info('Loading training data...')
    train_path = Path('./train_data/train.csv')
    train = pd.read_csv(train_path).dropna()
    
    # Initialize encoders
    encoders = {col: LabelEncoder() for col in CAT_COLS}
    
    # Fit encoders
    logger.info('Fitting encoders...')
    for col in CAT_COLS:
        if col in train.columns:
            logger.debug(f'Fitting encoder for {col}')
            unique_values = train[col].astype(str).unique()
            encoders[col].fit(unique_values)
    logger.info('Encoders fitted.')
    
    # Encode categoricals
    logger.info('Encoding categorical features...')
    cat_df = train[CAT_COLS].copy()
    for col in CAT_COLS:
        if col in cat_df.columns:
            cat_df[col] = encoders[col].transform(cat_df[col].astype(str))
    
    # Add features
    logger.info('Adding time features...')
    train = add_time_features(train)
    logger.info('Adding cartesian features...')
    train = add_cartesian_features(train)
    
    # Merge and drop columns
    logger.info('Finalizing dataframe...')
    train = train.drop(columns=CAT_COLS)
    cat_df.index = train.index
    train = train.join(cat_df)
    
    logger.info(f'Train data processed. Shape: {train.shape}')
    return train

def run_preproc(input_df, update_encoders=True):
    """Preprocess input data"""
    logger.info('Running preprocessing...')
    
    # Create new encoders each time for simplicity
    encoders = {col: LabelEncoder() for col in CAT_COLS}
    
    # Fit encoders
    for col in CAT_COLS:
        if col in input_df.columns:
            unique_values = input_df[col].astype(str).unique()
            encoders[col].fit(unique_values)
    
    # Encode categoricals
    cat_df = input_df[CAT_COLS].copy()
    for col in CAT_COLS:
        if col in cat_df.columns:
            cat_df[col] = encoders[col].transform(cat_df[col].astype(str))
    
    # Add features
    input_df = add_time_features(input_df)
    input_df = add_cartesian_features(input_df)
    
    # Merge and drop columns
    input_df = input_df.drop(columns=CAT_COLS)
    cat_df.index = input_df.index
    output_df = input_df.join(cat_df)
    
    logger.info(f'Preprocessing completed. Shape: {output_df.shape}')
    return output_df
# Import standard libraries
import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
import os

# Import extra modules
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
RANDOM_STATE = 42

CAT_COLS = ['merch', 'cat_id', 'name_1', 'name_2',
           'gender', 'street', 'one_city', 'us_state', 'post_code', 'jobs']
DROP_COLS = ['transaction_time', 'lat', 'lon', 'merchant_lat', 'merchant_lon']
DATA_DIR = Path('./train_data')

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

def save_encoders(encoders):
    """Save label encoders to disk for later use"""
    DATA_DIR.mkdir(exist_ok=True)
    for col, encoder in encoders.items():
        with open(DATA_DIR / f'{col}_encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)

    logger.info("Labels saved")

def load_label_encoders():
    """Load label encoders from disk with creating new encoders"""
    encoders = {}
    for col in CAT_COLS:
        try:
            with open(DATA_DIR / f'{col}_encoder.pkl', 'rb') as f:
                encoders[col] = pickle.load(f)
        except (FileNotFoundError, EOFError):
            encoders[col] = LabelEncoder()
            logger.info(f"Created new encoder for {col}")
    return encoders

def update_encoder(encoder, new_values):
    """Update encoder with new values"""
    new_values = [str(v) for v in new_values if pd.notna(v)]
    existing_classes = set(encoder.classes_)
    new_classes = [v for v in new_values if v not in existing_classes]
    
    if new_classes:
        encoder.classes_ = np.concatenate([encoder.classes_, new_classes])
    logger.info(f"New values in encoder {encoder}")

    return encoder
    

def encode_column(encoder, data):
    """Encode data with handling of new unseen values"""
    return data.apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else len(encoder.classes_))

def load_train_data():
    """Load and preprocess training data, saving label encoders"""
    logger.info('Loading training data...')
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv')).dropna()
    
    # Load or create encoders
    encoders = {}
    encoders_exist = all((DATA_DIR / f'{col}_encoder.pkl').exists() for col in CAT_COLS)
    
    if encoders_exist:
        encoders = load_label_encoders()
        logger.info('Loaded existing encoders.')
           
        for col in CAT_COLS:
            if col in train.columns:
                unique_values = train[col].astype(str).unique()
                encoders[col] = update_encoder(encoders[col], unique_values)
        save_encoders(encoders)
        logger.info('Encoders updated and saved.')
    else:
        for col in CAT_COLS:
            encoders[col] = LabelEncoder()
            unique_values = train[col].astype(str).unique()
            encoders[col].fit(unique_values)
        save_encoders(encoders)
        logger.info('Created and saved new encoders.')
    
    save_encoders(encoders)
    logger.info('Encoders created')

    # Encode categoricals
    cat_df = train[CAT_COLS].copy()
    for col in CAT_COLS:
        if col in cat_df.columns:
            cat_df[col] = encode_column(encoders[col], cat_df[col].astype(str))
    logger.info('Cat features encoded')
    
    # Add features
    train = add_time_features(train)
    train = add_cartesian_features(train)
    
    # Merge and drop columns
    train = train.drop(columns=CAT_COLS + DROP_COLS)
    cat_df.index = train.index
    train = train.join(cat_df)
    
    logger.info(f'Train data processed. Shape: {train.shape}')
    return train

def run_preproc(input_df, update_encoders=True):
    """Preprocess input data with option to update encoders"""
    logger.info('Running preprocessing...')
    
    encoders = load_label_encoders()
    
    if update_encoders:
        for col in CAT_COLS:
            if col in input_df.columns:
                unique_values = input_df[col].astype(str).unique()
                encoders[col] = update_encoder(encoders[col], unique_values)
        save_encoders(encoders)
    
    # Encode categoricals
    cat_df = input_df[CAT_COLS].copy()
    for col in CAT_COLS:
        if col in cat_df.columns:
            cat_df[col] = encode_column(encoders[col], cat_df[col].astype(str))
    
    # Add features
    input_df = add_time_features(input_df)
    input_df = add_cartesian_features(input_df)
    
    # Merge and drop columns
    input_df = input_df.drop(columns=CAT_COLS + DROP_COLS)
    cat_df.index = input_df.index
    output_df = input_df.join(cat_df)
    
    logger.info(f'Preprocessing completed. Shape: {output_df.shape}')
    return output_df
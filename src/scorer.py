import pandas as pd
import logging
import xgboost as xgb
from pathlib import Path

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = xgb.XGBClassifier()
model_path = Path('./models/xgb.bin')
model.load_model(model_path)

logger.info('Pretrained model imported successfully...')

# Make prediction
def make_pred(dt, path_to_file):

    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': model.predict(dt)
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Return proba for positive class
    return submission

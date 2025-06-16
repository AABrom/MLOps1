import os
import sys
import pandas as pd
import time
import logging
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from watchdog.observers.polling import PollingObserver #лучше реагирует на windows
from pathlib import Path

sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred, model

log_dir = Path('./logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'service.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        
        self.input_dir = Path('./input')
        self.output_dir = Path('./output')
        
        # Создаем папки, если их нет
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.train = load_train_data()
        
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path)

            logger.info('Starting preprocessing')
            processed_df = run_preproc(input_df)
            
            logger.info('Making prediction')
            submission = make_pred(processed_df, file_path)
            
            logger.info('Prepraring submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(file_path))[0]


            output_filename = f"predictions_{timestamp}_{base_name}.csv"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)

            self.save_feature_importances(base_name, timestamp)
            
            self.save_score_distribution(submission['prediction'], base_name, timestamp)

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return
        
    def save_feature_importances(self, base_name, timestamp):
        """Save top-5 features to json"""
        features = pd.DataFrame({
                'feature': model.feature_names_in_,
                'importance': model.feature_importances_
            })
        top_features = features.sort_values('importance', ascending=False).head(5)
        result = dict(zip(top_features['feature'], top_features['importance']))
            
        output_file = os.path.join(self.output_dir, f"feature_importances_{timestamp}_{base_name}.json"
            )
            
        with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
        logger.info('Feature importances saved to %s', output_file)
        
    def save_score_distribution(self, scores, base_name, timestamp):
        """Generates score distribution"""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(scores, fill=True)
        plt.title('Score Distribution')
        plt.xlabel('Prediction Score')
        
        output_file = os.path.join(self.output_dir, f"score_distribution_{timestamp}_{base_name}.png")
        
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info('Score distribution plot saved to %s', output_file)


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        logger.info(f"Event: {event}")
        if not event.is_directory:
            # здесь сложности с путями, поэтому несколько проверок
            file_path = os.path.normpath(event.src_path)  
            if file_path.endswith(".csv"):
                logger.info(f'Found new file: {file_path}')
                if os.path.exists(file_path):  
                    self.service.process_single_file(file_path)
                else:
                    logger.error(f"File is not found: {file_path}")
        

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()

    if not os.path.exists(service.input_dir):
        logger.error(f"Input directory does not exist: {service.input_dir}")
        os.makedirs(service.input_dir, exist_ok=True)
        logger.info(f"Created input directory: {service.input_dir}")

    logger.info(f"Contents of input directory: {os.listdir(service.input_dir)}")
    
    observer = PollingObserver()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()
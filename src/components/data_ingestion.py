import os
import sys
from src.exception import CustomException
from src.logger import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv('notebook/data/dataset.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Saved raw data")

            X_temp = df.drop(columns=['Credit_Score'])
            y_temp = df['Credit_Score']
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
            )
            
            train_set = pd.concat([X_train_temp, y_train_temp], axis=1)
            test_set = pd.concat([X_test_temp, y_test_temp], axis=1)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info(f"Train shape: {train_set.shape}, Test shape: {test_set.shape}")


            logging.info("Ingestion of data is completed")
            print("Columns in DataFrame:", df.columns.tolist())

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred in data ingestion")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
    
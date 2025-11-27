import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

class ModelTrainer:
    def __init__(self, preprocessor_path, train_path, test_path):
        self.preprocessor_path = preprocessor_path
        self.train_path = train_path
        self.test_path = test_path
        self.model_path = os.path.join("artifacts", "random_forest_model.pkl")

    def train_model(self):
        try:
            logging.info("Loading preprocessed train and test data")
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            preprocessor = load_object(self.preprocessor_path)
            
            target_column = 'Credit_Score'

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Transforming data using preprocessor")
            X_train_t = preprocessor.transform(X_train)
            X_test_t = preprocessor.transform(X_test)

            logging.info("Training Random Forest Classifier")
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train_t, y_train)

            y_pred = rf.predict(X_test_t)
            acc = accuracy_score(y_test, y_pred)
            logging.info(f"Test Accuracy: {acc}")
            print("Classification Report:\n", classification_report(y_test, y_pred))
            logging.info("Saving trained model")
            save_object(self.model_path, rf)
            return self.model_path
        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)

if __name__ == "__main__":
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")

    model_trainer = ModelTrainer(preprocessor_path, train_path, test_path)
    model_trainer.train_model()
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    feature_selector_path = os.path.join('artifacts',"feature_selector.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, X_train):
        """
        Creates the preprocessing pipeline:
        - Cleans numeric columns (converts dirty strings to NaN)
        - Imputes missing values
        - Encodes categorical columns
        """
        try:
            logging.info("Data Transformation initiated")

            # --------------------
            # Identify categorical and numeric columns automatically
            # --------------------
            categorical_columns = X_train.select_dtypes(include='object').columns.tolist()
            # Remove target from categorical if present
            if 'Credit_Score' in categorical_columns:
                categorical_columns.remove('Credit_Score')

            numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # --------------------
            # Pipelines
            # --------------------
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessing pipelines created successfully")
            return preprocessor

        except Exception as e:
            logging.error("Error in creating preprocessing object")
            raise CustomException(e, sys)
        

        def prune_correlated_features(self, X_train_df, threshold=0.9):
            """Exact correlation pruning from notebook"""
            corr_matrix = X_train_df.corr().abs()
            redundant = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > threshold:
                        redundant.add(corr_matrix.columns[j])
            
            selected_features = [col for col in X_train_df.columns if col not in redundant]
            logging.info(f"Pruned {len(redundant)} features, kept {len(selected_features)}")
            return selected_features                                                                                                                                        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test CSV files")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train-Test Data Loaded")

            # Drop irrelevant columns
            drop_cols = ['ID', 'Customer_ID', 'Name', 'SSN']
            train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], inplace=True)
            test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], inplace=True)

            # Clean numeric columns
            train_df = self.clean_numeric_columns(train_df)
            test_df = self.clean_numeric_columns(test_df)

            target_column_name = 'Credit_Score'
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # Get preprocessor
            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)

            # Fit-transform
            logging.info("Applying preprocessing to training and testing data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessor
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("Error in initiate_data_transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Run full pipeline
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)
    
    trainer = ModelTrainer()
    score = trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Pipeline Complete! Final Test Accuracy: {score:.4f}")

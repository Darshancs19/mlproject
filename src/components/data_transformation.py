import os
import sys
from dataclasses import dataclass
import numpy as np  
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")
            categorical_columns = ['Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour','Credit_Score', 'Income_Bin']
            numerical_columns = ['Month', 'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
       'Interest_Rate', 'Annual_Income_qt', 'Auto Loan', 'Credit-Builder Loan',
       'Debt Consolidation Loan', 'Home Equity Loan', 'Mortgage Loan',
       'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan', 'DTI',
       'EMI_to_Income', 'Invest_to_Income', 'Balance_to_Income',
       'Avg_Delay_if_Delayed', 'Has_Delays', 'High_Utilization',
       'Total_Financial_Products', 'Inquiries_per_Year', 'Limit_Decrease_Flag',
       'Large_Limit_Change', 'Num_Loan_Types']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical and categorical pipeline completed")
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )   
            return preprocessor
        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys)
        
    # def initiate_data_transformation(self, train_path, test_path):
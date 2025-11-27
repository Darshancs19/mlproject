# src/components/predict.py
import os
import sys
import pandas as pd
import joblib
from src.exception import CustomException
from src.logger import logging

class Predictor:
    numeric_cols = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate', 'DTI',
        'EMI_to_Income', 'Invest_to_Income', 'Balance_to_Income', 'Avg_Delay_if_Delayed',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
        'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
        'Credit_History_Age', 'Limit_Decrease_Flag', 'Large_Limit_Change', 'Num_Loan_Types',
        'Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan',
        'Mortgage Loan', 'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan'
    ]

    categorical_cols = [
        'Month', 'Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score',
        'Annual_Income_qt', 'Income_Bin'
    ]

    def __init__(self, model_path: str, preprocessor_path: str):
        try:
            logging.info("Loading model and preprocessor")
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def _clean_input(self, df: pd.DataFrame):
        """
        Fully robust cleaning:
        - Convert numeric columns to float (invalid -> 0)
        - Convert categorical columns to string (missing -> 'Missing')
        - Align columns with preprocessor
        """
        try:
            logging.info("Cleaning input DataFrame")
            
            # Ensure numeric columns are floats
            for col in self.numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else:
                    df[col] = 0  # add missing numeric column

            # Ensure categorical columns are strings
            for col in self.categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('Missing')
                else:
                    df[col] = 'Missing'

            # Align columns with preprocessor
            preprocessor_cols = self.preprocessor.feature_names_in_
            for col in preprocessor_cols:
                if col not in df.columns:
                    if col in self.numeric_cols:
                        df[col] = 0
                    else:
                        df[col] = 'Missing'
                else:
                    # force numeric columns to float
                    if col in self.numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    else:
                        df[col] = df[col].astype(str).fillna('Missing')

            print(self.preprocessor.feature_names_in_)


            # Reorder exactly as preprocessor expects
            df = df[preprocessor_cols]

            # Log dtypes safely
            logging.info("Data types after cleaning:\n%s", df.dtypes)
            logging.info("Input DataFrame after cleaning and alignment:\n%s", df)

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def predict_single(self, input_dict: dict):
        try:
            logging.info("Predicting single record")
            df = pd.DataFrame([input_dict])
            df = self._clean_input(df)
            X_transformed = self.preprocessor.transform(df)
            prediction = self.model.predict(X_transformed)[0]
            logging.info("Single record prediction completed")
            return prediction
        except Exception as e:
            raise CustomException(e, sys)

    def predict_batch(self, df: pd.DataFrame):
        try:
            df = self._clean_input(df)
            logging.info("Predicting batch data")
            X_transformed = self.preprocessor.transform(df)
            predictions = self.model.predict(X_transformed)
            logging.info("Batch predictions completed")
            return predictions
        except Exception as e:
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    try:
        model_path = os.path.join("models", "model.pkl")
        preprocessor_path = os.path.join("models", "preprocessor.pkl")
        
        predictor = Predictor(model_path, preprocessor_path)
        
        example = {
            "Month": 1,
            "Age": 36,
            "Occupation": "Scientist",
            "Annual_Income": 500000,
            "Monthly_Inhand_Salary": 40000,
            "Num_Bank_Accounts": 2,
            "Num_Credit_Card": 1,
            "Interest_Rate": 13.5,
            "Num_of_Loan": 2,
            "Type_of_Loan": "Personal Loan",
            "Delay_from_due_date": 0,
            "Num_of_Delayed_Payment": 0,
            "Changed_Credit_Limit": 0,
            "Num_Credit_Inquiries": 2,
            "Credit_Mix": "Standard",
            "Outstanding_Debt": 100000,
            "Credit_Utilization_Ratio": 25,
            "Credit_History_Age": 48,
            "Payment_of_Min_Amount": "Yes",
            "Total_EMI_per_month": 15000,
            "Amount_invested_monthly": 5000,
            "Payment_Behaviour": "High_spent_Small_value_payments",
            "Monthly_Balance": 20000
        }
        
        prediction = predictor.predict_single(example)
        logging.info(f"Predicted Credit Score: {prediction}")
        
    except Exception as e:
        logging.error(f"Error in prediction: {e}")

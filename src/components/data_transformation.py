import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation settings.
    """
    artifact_dir: str = os.path.join(artifact_folder)
    transformed_train_file_path: str = os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path: str = os.path.join(artifact_dir, 'test.npy') 
    transformed_object_file_path: str = os.path.join(artifact_dir, 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, feature_store_file_path: str):
        """
        Initializes the DataTransformation class with the path to the feature store.
        """
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        """
        Reads the raw data from the given file path and returns it as a pandas DataFrame.
        Renames the target column to the one specified in the constants.
        """
        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"default payment next month": TARGET_COLUMN}, inplace=True)
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a preprocessor pipeline that includes an imputer and scaler.
        """
        try:
            # Define the preprocessor steps
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(steps=[imputer_step, scaler_step])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> tuple:
        """
        Performs data transformation including:
        1. Reading and cleaning data.
        2. Splitting data into train and test.
        3. Scaling and saving the preprocessor.
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation class")

        try:
            # Step 1: Load and prepare the data
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)
            
            # Check that the target column exists
            if TARGET_COLUMN not in dataframe.columns:
                raise CustomException(f"Target column '{TARGET_COLUMN}' not found in the data.", sys)

            # Separate features (X) and target (y)
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = np.where(dataframe[TARGET_COLUMN] == 0, 1, 0)  # Replacing the target values as per the requirement
            
            # Step 2: Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Step 3: Get the preprocessor and apply transformations
            preprocessor = self.get_data_transformer_object()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Step 4: Save the preprocessor object
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path, obj=preprocessor)

            # Step 5: Combine transformed data and target labels into final arrays
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # Return the transformed arrays and the preprocessor path
            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys) from e

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifact", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Creating data transformation object")

            num_features = [
                "writing_score",
                "reading_score",
            ]
            cat_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical data preprocessing
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            logging.info("numerical tranformation completed")

            # Categorical data preprocessing
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("categorical tranformation completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, num_features),
                    ("cat", cat_pipeline, cat_features),
                ]
            )

            logging.info("Data transformation object created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # pyrefly: ignore [implicit-any-parameter]
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation started")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data loaded successfully")

            preprocessor_obj = self.get_data_transformation_obj()

            target_column_name = "math_score"
            target_column_train = train_df[target_column_name]
            target_column_test = test_df[target_column_name]

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            input_feature_test_df = test_df.drop(columns=[target_column_name])

            train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[train_arr, target_column_train]
            test_arr = np.c_[test_arr, target_column_test]

            logging.info("Data transformation completed successfully")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     from src.components.data_ingestion import DataIngestionConfig, DataIngestion
#     config = DataIngestionConfig()
#     obj = DataIngestion(config)
#     train_data, test_data = obj.initiate_data_ingestion()
# 
#     data_transformation = DataTransformation()
#     train_arr, test_arr, preprocessor_obj_file_path = (
#         data_transformation.initiate_data_transformation(train_data, test_data)
#     )

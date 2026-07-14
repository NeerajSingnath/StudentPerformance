import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifact", "data.csv")
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion started")
            df = pd.read_csv("notebooks/data/stud.csv")
            logging.info("Data loaded successfully")

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.train_data_path),
                exist_ok=True,
            )

            df.to_csv(
                self.data_ingestion_config.raw_data_path, index=False, header=True
            )

            logging.info("Data saved successfully")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.data_ingestion_config.train_data_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Train test split completed successfully")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     data_ingestion_config = DataIngestionConfig()
#     data_ingestion = DataIngestion(data_ingestion_config)
#     data_ingestion.initiate_data_ingestion()

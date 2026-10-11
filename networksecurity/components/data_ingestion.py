from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from urllib.parse import quote_plus
from dotenv import load_dotenv
import certifi


# Load environment variables
load_dotenv()

# MongoDB credentials
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = quote_plus(os.getenv("MONGO_PASS"))   # encode @
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
MONGO_DB = os.getenv("MONGO_DB")

# MongoDB URI
MONGO_DB_URL = (
    f"mongodb+srv://{MONGO_USER}:{MONGO_PASS}"
    f"@{MONGO_CLUSTER}/{MONGO_DB}"
    f"?retryWrites=true&w=majority"
)


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            # self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.mongo_client = pymongo.MongoClient(
            MONGO_DB_URL,
            tls=True,
            tlsCAFile=certifi.where()
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Read data from MongoDB and return DataFrame
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace({"na": np.nan}, inplace=True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)

            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )

            logging.info("Train-test split completed successfully")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # def initiate_data_ingestion(self) -> DataIngestionArtifact:
    #     try:
    #         dataframe = self.export_collection_as_dataframe()
    #         dataframe = self.export_data_into_feature_store(dataframe)
    #         self.split_data_as_train_test(dataframe)

    #         return DataIngestionArtifact(
    #             trained_file_path=self.data_ingestion_config.training_file_path,
    #             test_file_path=self.data_ingestion_config.testing_file_path
    #         )

    #     except Exception as e:
    #         raise NetworkSecurityException(e, sys)
        
    def initiate_data_ingestion(self):
        try:
            print(">>> Fetching data from MongoDB")
            dataframe = self.export_collection_as_dataframe()

            print(">>> Saving feature store CSV")
            dataframe = self.export_data_into_feature_store(dataframe)

            print(">>> Performing train-test split")
            self.split_data_as_train_test(dataframe)

            print(">>> Creating artifact")
            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)



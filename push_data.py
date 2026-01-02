import os
import sys
import json
import pandas as pd
import pymongo
import certifi
from urllib.parse import quote_plus
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException

load_dotenv()

# Mongo credentials
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = quote_plus(os.getenv("MONGO_PASS"))
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")

MONGO_DB_URL = (
    f"mongodb+srv://{MONGO_USER}:{MONGO_PASS}"
    f"@{MONGO_CLUSTER}/?retryWrites=true&w=majority"
)


class NetworkDataExtract:
    def __init__(self):
        try:
            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                tls=True,
                tlsCAFile=certifi.where()
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path: str):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            db = self.mongo_client[database]
            col = db[collection]
            col.insert_many(records)
            return len(records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    FILE_PATH = r"Network_data\phisingData.csv"
    DATABASE = "OMKAR"
    COLLECTION = "NETWORK"

    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(FILE_PATH)

    print(f"Total records to insert: {len(records)}")

    no_of_records = networkobj.insert_data_mongodb(
        records,
        DATABASE,
        COLLECTION
    )

    print(f"✅ Inserted {no_of_records} records into MongoDB")

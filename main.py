# from networksecurity.components.data_ingestion import DataIngestion
# from networksecurity.exception.exception import NetworkSecurityException
# from networksecurity.logging.logger import logging
# from networksecurity.entity.config_entity import (
#     DataIngestionConfig,
#     TrainingPipelineConfig
# )

# import sys


# if __name__ == "__main__":
#     try:
#         training_pipeline_config = TrainingPipelineConfig()
#         data_ingestion_config = DataIngestionConfig(training_pipeline_config)

#         data_ingestion = DataIngestion(data_ingestion_config)

#         logging.info("Initiating data ingestion pipeline")
#         data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

#         print(data_ingestion_artifact)

#     except Exception as e:
#         logging.error(NetworkSecurityException(e, sys))

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig
)

from networksecurity.components.data_transformation import DataTransformation

import sys

from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import ModelTrainerConfig

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)

        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("initiate data ingestion")
        print(">>> Starting data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logging.info("data initiation completed")
        print(data_ingestion_artifact)
        print(">>> Data ingestion completed")

        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact,data_validation_config)

        logging.info("initiate data validion")
        print(">>> Starting data validation")

        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("data validation completed")
        print("data validation completed")

        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        logging.info("data Transformation started")
        print("data Transformation started")
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("data Transformation completed")
        print("data Transformation completed")

        logging.info("Model Training started")
        print("model training started")

        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")
        print("model training completed")

    except Exception as e:
        print(NetworkSecurityException(e, sys))


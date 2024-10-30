import os
import sys

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig():
    raw_dataset_path = os.path.join("artifacts" , "dataset.csv")
    
    ##--------------------Database credentials
    driver:str = "ODBC+Driver+17+for+SQL+Server"
    server_name:str = "localhost"
    database:str = "BDdatasets"
    UID:str = "sa"
    PWD:str = "0440"

    connection_string:str = f"mssql+pyodbc://{UID}:{PWD}@{server_name}/{database}?driver={driver}"

class DataIngestion():
    def __init__(self):
        self.dataconfig = DataIngestionConfig()
    
    def initate_data_ingestion(self):
        logging.info("enter to the data ingestion method")
        try:
            engine = create_engine(self.dataconfig.connection_string)
            query  = "Select * FROM CarsBuyClassification"

            logging.info("reading database as dataframe")
            
            df = pd.read_sql_query(query , engine)

            logging.info("saving top 5 records")

            df.head().to_csv(self.dataconfig.raw_dataset_path ,header = True , index = False)

            logging.info("dividing the dataset")
            
            train_df ,test_df = train_test_split(df, test_size = 0.2 , random_state = 42)

            return (train_df ,test_df) 

        
        except Exception as e:
            raise CustomException(e , sys)
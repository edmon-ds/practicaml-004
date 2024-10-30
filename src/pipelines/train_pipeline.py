from src.components.data_ingestion import *
from src.components.data_transformation import *

data_ingestion = DataIngestion()
data_transformation = DataTransformation()
print("initating data ingestion...")

train_df ,test_df = data_ingestion.initate_data_ingestion()
print(train_df.head())
print("="*40)
train_array , test_array  = data_transformation.initiate_data_transformation(train_df ,test_df)
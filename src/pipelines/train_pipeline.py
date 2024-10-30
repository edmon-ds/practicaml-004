from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

data_ingestion = DataIngestion()
data_transformation = DataTransformation()
model_trainer = ModelTrainer()

print("initating data ingestion...")
train_df ,test_df = data_ingestion.initate_data_ingestion()

print("initating data transformation...")
train_array , test_array = data_transformation.initiate_data_transformation(train_df ,test_df)

print("initiating model training")
best_model_name, best_model_score ,best_model = model_trainer.iniatiate_model_training(train_array , test_array)
print()
print("report of the models")
model_trainer.show_report()
print()
print(f"the best model is {best_model_name} with a average score of {best_model_score}")
print()
print("train pipeline complete ")
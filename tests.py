import pandas as pd
from src.utils import *
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig

input_preprocessor = load_object(DataTransformationConfig.input_preprocessor_path)
output_encoder = load_object(DataTransformationConfig.output_encoder_path)
model = load_object(ModelTrainerConfig.model_path)
data = pd.read_csv("artifacts/dataset.csv" , dtype= str).head(1).drop(columns= ["class"])

#print(data.info())
X_transformed = input_preprocessor.transform(data)

print(X_transformed)

pred = model.predict(X_transformed)
print(pred)
print()
print(output_encoder.inverse_transform([pred]))
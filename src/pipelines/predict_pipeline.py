import pandas as pd
from src.utils import *
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig

class CustomData():
    def __init__(self , buying,maint,doors,persons,lug_boot,safety):
        self.user_data = pd.DataFrame(
           {"buying":[buying],
            "maint":[maint],
            "doors":[doors],
            "persons":[persons],
            "lug_boot":[lug_boot],
            "safety":[safety]}
        , dtype= str )
    def get_data_as_dataframe(self):
        return self.user_data

class PredictPipeline():
    def __init__(self):
        self.input_preprocessor = load_object(DataTransformationConfig.input_preprocessor_path)
        self.output_encoder = load_object(DataTransformationConfig.output_encoder_path)
        self.model = load_object(ModelTrainerConfig.model_path)

    def predict(self , user_data):
        try:
            data_transformed = self.input_preprocessor.transform(user_data) 
            preds = self.model.predict(data_transformed)

            #transform num predictions to category           
            preds = self.output_encoder.inverse_transform([preds])
            
            return preds 
        except Exception as e:
            raise CustomException(e , sys)
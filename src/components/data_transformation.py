import sys
import os
from src.logger import logging 
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import numpy as np


from sklearn.base import BaseEstimator , TransformerMixin

@dataclass
class DataTransformationConfig():
    input_preprocessor_path:str = os.path.join("artifacts" , "input_preprocessor.pkl") # pipelines
    output_encoder_path:str = os.path.join("artifacts" , "output_encoder.pkl") # encoder

class RestoreNameTransformer(BaseEstimator , TransformerMixin):
    def __init__(self , features_names):
        self.features_names = features_names
    def fit(self , X ):
        return self
    def transform(self, X):
        return pd.DataFrame(X , columns= self.features_names)
    

class DataTransformation():
    def __init__(self):
        self.dataconfig = DataTransformationConfig()
        self.categorical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        self.categorical_columns_values = [
                ["low" , "med" , "high" ,"vhigh"] , # buying
                ["low" , "med" , "high" ,"vhigh"],     # maint
                ['2', '3', '4', '5more'],            # doors
                ['2', '4', 'more'],                  # persons
                ['small', 'med', 'big'],             # lug_boot
                ['low', 'med', 'high']               # safety
                                ]
        self.label = "class"
        self.label_categories = [['unacc' ,'acc' , 'good']]
        
    def get_preprocessor(self):

        #pipelines
        #steps for cleaning 
        cleaning_pipeline = ColumnTransformer(
                             [
                                 ("cleaning_steps" , SimpleImputer(strategy="most_frequent") , self.categorical_columns)
                             ]    
                                 )
        restore_names_transformer = RestoreNameTransformer(self.categorical_columns)
        #steps for preprocessing the inputs
        preprocessing_pipeline = ColumnTransformer(
                             [
                                 ("preprocessing_steps" , OrdinalEncoder(categories=self.categorical_columns_values) , self.categorical_columns )
                                 ]
                                                )
        
        
        #full pipelines, this  clean and preprocess the inputs
        input_preprocessor = Pipeline(steps=[
            ("cleaning_pipeline" , cleaning_pipeline ) , 
            ("recover_names" , restore_names_transformer) , 
            ("preprocessing_pipeline" , preprocessing_pipeline)
        ])

        #label preprocessor
        output_encoder =  OrdinalEncoder(categories=self.label_categories)

        return (input_preprocessor , output_encoder)
    
    def initiate_data_transformation(self ,train_df: pd.DataFrame ,test_df: pd.DataFrame ):
        logging.info("enter to initiate_data_transformation function")
        try:
            logging.info("joining good and vgood labels")
            train_df.loc[train_df[self.label] =="vgood" , self.label] = "good"
            test_df.loc[test_df[self.label] =="vgood" , self.label] = "good"

            logging.info("creating preprocessor object")
            input_preprocessor , output_encoder = self.get_preprocessor()
            
            train_input_raw = train_df.drop(columns=[self.label])
            train_label_raw = train_df[[self.label]]

            test_input_raw = test_df.drop(columns=[self.label])
            test_label_raw = test_df[[self.label]]

            logging.info("applying preprocessing to the dataset")

            train_input_array = input_preprocessor.fit_transform(train_input_raw)
            test_input_array = input_preprocessor.transform(test_input_raw)
            
            train_label_array = output_encoder.fit_transform(train_label_raw)
            test_label_array =  output_encoder.transform(test_label_raw)

            logging.info("joining features and labels columns")

            train_array = np.hstack((train_input_array , train_label_array))
            test_array = np.hstack((test_input_array , test_label_array))

            logging.info("saving input and labels preprocessor")

            save_object(file_path= self.dataconfig.input_preprocessor_path ,  obj=input_preprocessor)
            save_object(file_path= self.dataconfig.output_encoder_path , obj= output_encoder)

            return ( train_array , test_array )

        except Exception as e:
            raise CustomException(e , sys)




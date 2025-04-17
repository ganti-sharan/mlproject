import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            
            self.target_column_name = 'math_score'
            # Here you can write a method to get col names dynamically like this X.select_dtypes(exclude="object").columns
            raw_data = pd.read_csv('/Users/sharan/Documents/ML Study/MLProject/artifacts/raw.csv')
            self.numerical_columns = raw_data.select_dtypes(exclude="object").columns.tolist()
            self.numerical_columns.remove(self.target_column_name)  ##Target Variable check and you can generalize this line too
            
            self.cat_columns = raw_data.select_dtypes(include="object").columns.tolist()
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy= "most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info('Pipeline created for numerical and categorical columns for missing and scaling values')
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pileline", num_pipeline, self.numerical_columns),
                    ("cat_pipeline", cat_pipeline, self.cat_columns)
                ]
            )
            
            logging.info('sent columns through the pipeline')
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self,train_path, test_path):
        
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Train and Test data imported')
            
            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj = self.get_data_transformer_object()
            
            input_feature_train_df = train_df.drop(columns = [self.target_column_name], axis = 1)
            target_feature_train_df = train_df[self.target_column_name]
            
            input_feature_test_df = test_df.drop(columns = [self.target_column_name], axis = 1)
            target_feature_test_df = test_df[self.target_column_name]
            
            logging.info("Now applying preproceesing object to the test and train sets")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            
            
            logging.info('Finished sending the train and test data through the preprocessing pipeline')
            logging.info('Saving preprocessing object')
            
            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            logging.info("Pickle file saved")
            
            return(
                train_arr,
                test_arr
            )
            
        except Exception as e:
            raise CustomException(e, sys)
            
            
    
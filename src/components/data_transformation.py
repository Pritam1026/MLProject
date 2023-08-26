import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    #A pickle file path is created

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    #data_transformation_config will have the sub variable preprocessed pickle file

    def data_transformation_object(self):
        """
        This function is responsible for data transformation.
        """
        try:
            numerical_columns=["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline=Pipeline(
                [('imputer',SimpleImputer(strategy='median')),
                 ('scaler',StandardScaler())]
            )

            cat_pipeline=Pipeline(
                [('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoding',OneHotEncoder())]
                                   )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer([('num_pipeline',num_pipeline,numerical_columns),
                                            ('cat_pipeline',cat_pipeline,categorical_columns)])
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        """
        This function will the data transformation using the data_transformation object method.
        """
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            processing_object=self.data_transformation_object()
            #Internally called the preprocessor object Now this preprocessing object will do the data transformation.

            target_column='math_score'

            input_feature_train_df=train_df.drop(columns=target_column,axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=target_column,axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info('Feature transformation initiated')

            input_feature_train_arr=processing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=processing_object.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Training and test features are transformed')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=processing_object
            )

            logging.info('Preprocessor trained object is saved.')

            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
               train_array[:,:-1],
               train_array[:,-1],
               test_array[:,:-1],
               test_array[:,-1]
            )
            models={
               "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN Regressor": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }
            params={
                "Decision Tree":{
                     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Gradient Boosting":{
                     'n_estimators':[8,16,32,64,128,256],
                     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                     'learning_rate':[0.1,0.01,0.05,0.001]
                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "AdaBoost":{
                      'n_estimators':[8,16,32,64,128,256],
                       'learning_rate':[0.1,0.01,0.05,0.001] ,
                       'loss':['linear', 'square', 'exponential']
                },
                "KNN Regressor":{
                     'n_neighbors':[5,7,9,11],
                },
                "XGBoost":{
                      'n_estimators':[8,16,32,64,128,256],
                      'learning_rate':[0.1,0.01,0.05,0.001] ,
                },
                "CatBoost":{
                       'learning_rate':[0.1,0.01,0.05,0.001] ,
                       'depth':[6,8,10]
                },




            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,
                                                y_test=y_test,models=models,params=params)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)   ###its like --->list(model_report.keys())[2]
            ]
            
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging .info("best model found on training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_val=r2_score(y_test,predicted)
            return r2_val
        except Exception as e:
             raise CustomException(e,sys)

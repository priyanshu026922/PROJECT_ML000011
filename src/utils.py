import os
import sys
import dill

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj )

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
     try:
         report={}
         for model_name,model in models.items():
             para=params[model_name]
             if para:
                gs=GridSearchCV(model,para,cv=3,n_jobs=3)
                gs.fit(X_train,y_train)
                best_model=gs.best_estimator_
             else:
                 model.fit(X_train,y_train)
                 best_model=model

             
             y_test_pred=best_model.predict(X_test)
 
             test_model_score=r2_score(y_test,y_test_pred)
             report[model_name]=test_model_score
         
         return report
     
     except Exception as e:
         raise CustomException(e,sys)
     


def load_object(file_path):
    try:
        with open (file_path,"rb")as file_obj:
            return dill.load(file_obj)

    except Exception as e:
         raise CustomException(e,sys)
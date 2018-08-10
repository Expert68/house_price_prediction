import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb



# --------------------------定义模型类------------------------------

class Models:

    def svm_regressor(self):
        param_grid1 = [{
            'degree':[i for i in range(2,10)],
        }]
        self.svm_regressor = SVR()
        svm_grid_search = GridSearchCV(self.svm_regressor,param_grid1,cv=5,n_jobs=4,verbose=1)
        return svm_grid_search

    def dt_regressor(self):
        param_grid2 = [{
            'max_depth':[i for i in range(2,10)],
            'max_features':[i for i in range(2,16)]
        }]
        self.df_regressor = DecisionTreeRegressor()
        dt_grid_search = GridSearchCV(self.df_regressor,param_grid2,cv=5,n_jobs=4,verbose=1)
        return dt_grid_search

    def xgb_regressor(self):
        param_grid3 = [{
            'max_depth':[i for i in range(2,10)],
        }]
        self.xgb_regressor = xgb.XGBRegressor()
        xgb_grid_search = GridSearchCV(self.xgb_regressor,param_grid3,cv=5,n_jobs=4,verbose=1)
        return xgb_grid_search





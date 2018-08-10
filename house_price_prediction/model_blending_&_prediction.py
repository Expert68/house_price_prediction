import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from mlxtend.regressor import StackingRegressor
from models import Models

# ---------------------------------------读取数据-----------------------------------------
x_train = pd.read_csv('cleaned_data/cleaned_x_train')
x_test = pd.read_csv('cleaned_data/cleaned_x_test')
y_train = pd.read_csv('cleaned_data/cleaned_y_train')
model = Models()


# ---------------------------------------定义混合类----------------------------------------
class Blend:
    def __init__(self, x_train, x_test, y_train):
        x_train.drop(['Id'], axis=1, inplace=True)
        x_test.drop(['Id'], axis=1, inplace=True)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train['y_train'].values

    def blending(self):
        mete_reg = LinearRegression()
        reg1 = model.svm_regressor()
        reg2 = model.dt_regressor()
        reg3 = model.xgb_regressor()
        self.blend = StackingRegressor(regressors=[reg1, reg2, reg3], meta_regressor=mete_reg)
        self.blend.fit(self.x_train, self.y_train)
        return self.blend

    def score(self):
        scores = cross_val_score(self.blend, X=self.x_train, y=self.y_train, cv=10,
                                 verbose=2)  # scoring='neg_mean_squared_error'
        return scores

    def prediction(self):
        y_pred = self.blend.predict(self.x_test)
        y_pred = np.expm1(y_pred)
        return y_pred


if __name__ == '__main__':
    blender = Blend(x_train, x_test, y_train)
    blender.blending()
    scores = blender.score()
    print(scores)
    prediction = pd.DataFrame()
    prediction['house_price_pred'] = blender.prediction()
    prediction.to_csv('output/prediction.csv')

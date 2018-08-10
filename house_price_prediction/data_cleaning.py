# 该文件主要作用是用于对房价数据和特征进行清洗，通过定义数据清洗类来完成数据清洗操作

#---------------------------------定义数据清洗类-------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataClean:
    def __init__(self,train_df,test_df):
        self.train_df = train_df
        self.test_df = test_df

    def y_train_to_log1p(self):
        self.y_train = pd.DataFrame()
        self.y_train['y_train'] = np.log1p(self.train_df.pop('SalePrice'))
        return self.y_train

    def concat_df(self):
        self.all_df = pd.concat((self.train_df,self.test_df))
        return self.all_df

    def one_hot(self):
        self.all_dummy_df = pd.get_dummies(self.all_df)
        return self.all_dummy_df

    def standard(self):
        self.noise_cols = ['Id']
        self.all_dummy_df = self.all_dummy_df.fillna(0)
        self.numerical_cols = self.all_df.columns[self.all_df.dtypes != 'object']
        scaler = StandardScaler()
        for col in set(self.numerical_cols)-set(self.noise_cols):
            scaler.fit(self.all_dummy_df[col].reshape(-1,1))
            self.all_dummy_df[col] = scaler.transform(self.all_dummy_df[col].reshape(-1,1))
        return self.all_dummy_df

    def split(self):
        self.dummy_train_df = self.all_dummy_df.loc[self.train_df.index]
        self.dummy_test_df = self.all_dummy_df.loc[self.test_df.index]







#---------------------------------读取数据--------------------------------------

train_df = pd.read_csv('input/train.csv',index_col=0)  #一定要设置Id为index,否则在后面的concat部分会出现index重复的情况
test_df = pd.read_csv('input/test.csv',index_col=0)

#创建cleaner对象
cleaner =DataClean(train_df,test_df,)
cleaner.y_train_to_log1p()
cleaner.concat_df()
cleaner.one_hot()
cleaner.standard()
cleaner.split()

#将cleaner对象中封装的dummy_train_df,dummy_test_df和y_train保存到文件中
cleaner.dummy_train_df.to_csv('cleaned_data/cleaned_x_train')
cleaner.dummy_test_df.to_csv('cleaned_data/cleaned_x_test')
cleaner.y_train.to_csv('cleaned_data/cleaned_y_train')








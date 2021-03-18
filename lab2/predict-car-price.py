from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


class PriceoftheCar:

    def __init__(self):
        self.dataframe = pd.read_csv('data/data.csv')

    def Trimming(self):
        self.dataframe.columns = self.dataframe.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.dataframe.dtypes[self.dataframe.dtypes == 'object'].index)
        for i in string_columns:
            self.dataframe[i] = self.dataframe[i].str.lower().str.replace(' ', '_')
            
    def Validation(self):
        np.random.seed(2)
        n = len(self.dataframe)
        nvalue = int(0.2 * n)
        ntest = int(0.2 * n)
        ntrain = n - (nvalue + ntest)
        index = np.arange(n)
        np.random.shuffle(index)
        Shuffleddataframe = self.dataframe.iloc[index]
        dataframe_train = Shuffleddataframe.iloc[:ntrain].copy()
        dataframe_value = Shuffleddataframe.iloc[ntrain:ntrain+nvalue].copy()
        dataframe_test = Shuffleddataframe.iloc[ntrain+nvalue:].copy()
        ytrain_initial= dataframe_train['msrp'].values
        yvalue_initial = dataframe_value['msrp'].values
        ytest_initial = dataframe_test['msrp'].values
        ytrain = np.log1p(dataframe_train.msrp.values)
        yvalue = np.log1p(dataframe_value.msrp.values)
        ytest = np.log1p(dataframe_test.msrp.values)
        
        del dataframe_train['msrp']
        del dataframe_value['msrp']
        del dataframe_test['msrp']

        return dataframe_train, dataframe_value, dataframe_test, ytrain_initial, yvalue_initial, ytest_initial, ytrain, yvalue, ytest

    def Linearregression(self, A, y):
        ones = np.ones(A.shape[0])
        A = np.column_stack([ones, A])
        k = A.T.dot(A)
        kin = np.linalg.inv(k)
        s = kin.dot(A.T).dot(y)

        return s[0], s[1:]

    def Prepare(self, dataframe):
        base = ['engine_hp', 'engine_cylinders',
                'highway_mpg', 'city_mpg', 'popularity']
        dataframe_number = dataframe[base]
        dataframe_number = dataframe_number.fillna(0)
        A = dataframe_number.values
        return A

    def RMSE(self, y, ypred):
        e = ypred - y
        Mse = (e ** 2).mean()
        return np.sqrt(Mse)


c = PriceoftheCar()
c.Trimming()
dataframe_train, dataframe_value, dataframe_test, ytrain_initial, yvalue_initial, ytest_initial, ytrain, yvalue, ytest = c.Validation()
Xtrain = c.Prepare(dataframe_train)
a_0, a = c.Linearregression(Xtrain, ytrain)
ypred = a_0 + Xtrain.dot(a)

print(a_0)

Xvalue = c.Prepare(dataframe_value)
ypred = a_0 + Xvalue.dot(a)
Prediction = np.expm1(ypred)

dataframe_value['msrp'] = yvalue_initial

dataframe_value['msrp_pred'] = Prediction

print(dataframe_value[['make', 'model', 'engine_cylinders', 'transmission_type', 'driven_wheels', 'number_of_doors',
              'market_category', 'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity', 'msrp', 'msrp_pred']].head().to_markdown())
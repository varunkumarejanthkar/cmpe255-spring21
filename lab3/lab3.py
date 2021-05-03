import pandas as pd
from inspect import CO_NEWLOCALS
import numbers

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import SelectKBest

class DiabetesClassifier:
        def __init__(self, length, breadth, unit_cost=0):
            print("Inside init");
            self.length = length
            self.breadth = breadth
            self.unit_cost = unit_cost
            col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
            self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
            print("Inside self");
            print(self.pima.head())
            self.X_test = None
            self.y_test = None        

        def define_feature(self, scale):
            global X
            if scale == 1:
                columns =  ['skin', 'insulin', 'bmi', 'age']
                X = self.pima[columns]
            elif scale == 2:
                columns = ['glucose', 'insulin', 'bmi', 'age']
                X = self.pima[columns]        
            elif scale == 3:
                columns = ['pregnant', 'glucose', 'insulin', 'bmi', 'bp']
                X = self.pima[columns]
            elif scale == 4:
                columns = ['pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'age', 'pedigree']
                X = self.pima[columns]            
            y = self.pima.label
            return X, y
    
        def train(self, scale = 0):
        # split X and y into training and testing sets
            X, y = self.define_feature(scale)
            X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
            logreg = LogisticRegression(max_iter = 1000)
            logreg.fit(X_train, y_train)
            return logreg
    
        def predict(self, scale=0):
            model = self.train( scale)
            y_pred_class = model.predict(self.X_test)
            return y_pred_class

        def calculate_accuracy(self, result):
            return metrics.accuracy_score(self.y_test, result)

        def examine(self):
            dist = self.y_test.value_counts()
            print(dist)
            percent_of_ones = self.y_test.mean()
            percent_of_zeros = 1 - self.y_test.mean()
            return self.y_test.mean()
    
        def confusion_matrix(self, result):
            return metrics.confusion_matrix(self.y_test, result)
        
        def use_skb(self):
            select = SelectKBest(k=4)
            select.fit(self.pima.iloc[:,:-1], self.pima.label)
            c = selector.get_support(indices=True)
            f = self.pima.iloc[:,c]
            return f

        def get_metrics(self, scale = 0):
            global s, cmatrix
            r = classifier.predict(scale)
            s = classifier.calculate_accuracy(r)
            cmatrix = classifier.confusion_matrix(r)
            #print(cmatrix)
            return cmatrix
            pass

classifier = DiabetesClassifier(160, 120, 2000)
print('| Experiement | Accuracy | Confusion Matrix | Comment |')
print('|-------------|----------|------------------|---------|')
print('| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |  |')
for scale in range(1,5):
    classifier.get_metrics( scale)
    print(f'| Solution %d | {score} | {con_matrix.tolist()} | Used features:  {X.columns.tolist()} ' %(scale) )

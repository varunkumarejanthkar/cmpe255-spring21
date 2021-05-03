import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class DiabetesClassifier:
    def __init__(self):
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        print(self.pima.head())
        print(self.pima.describe())
        self.X_test = None
        self.y_test = None
        self.X = None
        self.y = None
        
    def box_plot(self):
        # TODO: Draw a box plot with 'age' on y-axis and 'label' on x-axis using Seaborn plotting library. 
        # Then, save the box plot to 'diabetes_by_age.png' file.
        #
        
        plt.xlabel('label')
        plt.ylabel('age')
        plt.boxplot(self.pima)
        plt.title('Diabetes by Age')
        plt.savefig('diabetes_by_age.png')
        plt.show()

    def corr_matrix(self):
        # TODO: Calculate correlation matrix for each feature.
        corr_matrix = self.pima.corr()
        print(corr_matrix)
        sns.heatmap(corr_matrix)

    def create_new_feature(self):
        # TODO: create a new synthetic feature called 'bmi_skin' by multiplying 'bmi' * 'skin'
        # and set the new feature 'bmi_skin' into self.pima DataFrame.
        self.pima['bmi_skin'] = self.pima['bmi']* self.pima['skin']
        print(self.pima.head())

    def define_feature(self, feature_cols):
        self.X = self.pima[feature_cols]
        self.y = self.pima.label

    def train(self):
        # TODO: set test size to the 80/20 rule for training and testing in below train_test_split(...) function parameter.
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(max_iter=150)
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self):
        model = self.train()
        y_pred_class = model.predict(self.X_test)
        return y_pred_class

    def calculate_accuracy(self, pred):
        # TODO: compute accuracy_score using metrics library and return the score.
        accuracy = metrics.accuracy_score(self.y_test, pred)
        return accuracy

    def confusion_matrix(self, pred):
        # TODO: compute confusion_matrix using metrics library and return the score.
        cm = metrics.confusion_matrix(self.y_test, pred)
        return cm

    def precision_score(self, pred):
        # TODO: compute precision_score using metrics library and return the score.
        precision = metrics.precision_score(self.y_test, pred, average='macro')
        return precision


    def recall_score(self, pred):
        # TODO: compute recall_score using metrics library and return the score.
        recall = metrics.recall_score(self.y_test, pred, average='macro')
        return recall

    
if __name__ == "__main__":
    # Feel free to change any code here! Especially to answer the Diabetes Classifier related MCQ questions.
    classifer = DiabetesClassifier()
    classifer.box_plot()
    classifer.corr_matrix()

    # TODO: Based on the correlation matrix, select top 5 features.
    feature_cols = ['pregnant', 'age', 'label', 'glucose', 'bmi']

    classifer.create_new_feature()
    # TODO: add new synthetic feature 'bmi_skin' to the feature_cols list.

    feature_cols.append('bmi_skin')
    print(feature_cols)

    # Now, the feature cols should have 6 features.
    # Now, the feature cols should have 6 features.
    classifer.define_feature(feature_cols)
    result = classifer.predict()
    print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")

    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix={con_matrix}")

    precision = classifer.precision_score(result)
    print(f"precision_score={precision}")
    recall = classifer.recall_score(result)
    print(f"recall_score={recall}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
import parfit.parfit as pf
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
import pickle


# Gets the data and drops irrelevant columns
def get_training_and_testing_data():
    scaler = StandardScaler()

    print("Retreiving Datasets")
    df_train = pd.read_csv('./datasets/train_val.csv')  # , delim_whitespace=True)
    df_train.columns = df_train.columns.str.replace(' ', '')

    X_training = df_train.drop('MDD', axis=1)
    y_training = df_train['MDD']
    print(X_training.columns)

    scaler.fit(X_training)
    trainX = scaler.transform(X_training)

    df_test = pd.read_csv('./datasets/test.csv')  # ,  delim_whitespace=True)
    X_test = df_test.drop('MDD', axis=1)
    y_test = df_test['MDD']

    testX = scaler.transform(X_test)

    return trainX, y_training, testX, y_test

def SVM():
    X_train, y_train, X_test, y_test = get_training_and_testing_data()

    svm_model  = svm.SVC(random_state=0, class_weight='balanced', probability=True)

    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']} 

    clf = GridSearchCV(svm_model, param_grid, refit = True, scoring = 'roc_auc', verbose = 3)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    SVM_test_results = {'UKBB Test SVM Prob': y_pred_prob, 'UKBB Test SVM Label': y_pred, 'UKBB SVM Actual': y_test}
    df_SVM_test_results = pd.DataFrame.from_dict(SVM_test_results)
    df_SVM_test_results.to_csv('./Model-Comparison-Results/SVM_results.csv', index=False, header=True)

 
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print(confusion_matrix(y_test, y_pred))

    with open('./Models/SVM-model.pkl','wb') as f:
        pickle.dump(clf,f)

SVM()
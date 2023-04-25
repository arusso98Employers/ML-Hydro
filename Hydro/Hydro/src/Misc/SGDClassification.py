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
from time import sleep
from tqdm import tqdm
import parfit.parfit as pf
import re
import pickle
from sklearn.model_selection import GridSearchCV

def feature_list():
    df_train = pd.read_csv('Datasets/train.csv')  # , delim_whitespace=True)
    df_train.columns = df_train.columns.str.replace(' ', '')

    X_training = df_train.drop('MDD', axis=1)
    return X_training.columns

# Gets the data and drops irrelevant columns
def get_training_and_testing_data():
    scaler = StandardScaler()

    print("Retreiving Datasets")
    df_train = pd.read_csv('Datasets/train_val.csv')  # , delim_whitespace=True)
    df_train.columns = df_train.columns.str.replace(' ', '')

    X_training = df_train.drop('MDD', axis=1)
    y_training = df_train['MDD']
    #print(X_training.columns)

    scaler.fit(X_training)
    trainX = scaler.transform(X_training)

    df_test = pd.read_csv('Datasets/test.csv')  # ,  delim_whitespace=True)
    X_test = df_test.drop('MDD', axis=1)
    y_test = df_test['MDD']

    testX = scaler.transform(X_test)

    return trainX, y_training, testX, y_test


def sgd_model():
    X_train, y_train, X_test, y_test = get_training_and_testing_data()

    estimator = SGDClassifier(random_state=0, loss="log_loss", class_weight="balanced",
                        max_iter=5, alpha=0.01, penalty='l2')
    parameters = {
        'max_iter': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'alpha':  [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}   

    clf = GridSearchCV(estimator=estimator,param_grid=parameters,scoring = 'roc_auc',n_jobs = 10,cv = 10,verbose=True)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    XGB_test_results = {'UKBB Test SGD Prob': y_pred_prob, 'UKBB Test SGD Label': y_pred, 'UKBB SGD Actual': y_test}
    df_XGB_test_results = pd.DataFrame.from_dict(XGB_test_results)
    df_XGB_test_results.to_csv('Model-Comparison-Results/SGD_results.csv', index=False, header=True)

 
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print(confusion_matrix(y_test, y_pred))

    with open('./Models/SGD-model.pkl','wb') as f:
        pickle.dump(clf,f)

sgd_model()
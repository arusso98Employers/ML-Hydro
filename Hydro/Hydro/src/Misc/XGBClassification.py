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
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pickle

# Gets the data and drops irrelevant columns
def get_training_and_testing_data():
    scaler = StandardScaler()

    print("Retreiving Datasets")
    df_train = pd.read_csv('./datasets/train_val.csv')  # , delim_whitespace=True)
    df_train.columns = df_train.columns.str.replace(' ', '')

    X_training = df_train.drop('MDD', axis=1)
    y_training = df_train['MDD']

    scaler.fit(X_training)
    trainX = scaler.transform(X_training)

    df_test = pd.read_csv('Datasets/test.csv')  # ,  delim_whitespace=True)
    X_test = df_test.drop('MDD', axis=1)
    y_test = df_test['MDD']

    testX = scaler.transform(X_test)

    return trainX, y_training, testX, y_test

def XGBModel():
    X_train, y_train, X_val, y_val, X_test, y_test = get_training_and_testing_data()

    estimator = xgb.XGBClassifier(random_state=0, scale_pos_weight=8.6, objective = 'binary:logistic', tree_method = 'hist')

    parameters = {
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05]}   

    clf = GridSearchCV(estimator=estimator,param_grid=parameters,scoring = 'roc_auc',n_jobs = 10,cv = 10,verbose=True)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    XGB_test_results = {'UKBB Test XGB Prob': y_pred_prob, 'UKBB Test XGB Label': y_pred, 'UKBB XGB Actual': y_test}
    df_XGB_test_results = pd.DataFrame.from_dict(XGB_test_results)
    df_XGB_test_results.to_csv('Model-Comparison/XGB_results.csv', index=False, header=True)

 
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print(confusion_matrix(y_test, y_pred))

    with open('./Models/XGB-model.pkl','wb') as f:
        pickle.dump(clf,f)

XGBModel()
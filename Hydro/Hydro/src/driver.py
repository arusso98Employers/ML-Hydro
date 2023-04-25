import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from time import sleep
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from xgboost import XGBClassifier
import os

def feature_list():
    df_train = pd.read_csv('Datasets/train.csv')  # , delim_whitespace=True)
    list_of_column_names = list(df_train.columns)
    return list_of_column_names

def feature_list_retest():
    df_train = pd.read_csv('Datasets/retest.csv')  # , delim_whitespace=True)
    list_of_column_names = list(df_train.columns)
    return list_of_column_names

# Gets the data and drops irrelevant columns
def get_training_and_testing_data(inp):
    scaler = StandardScaler()

    print("Retreiving Datasets")
    df_train = pd.read_csv('Datasets/train.csv')  # , delim_whitespace=True)
    df_train.columns = df_train.columns.str.replace(' ', '')

    X_training = df_train.drop(inp, axis=1)
    y_training = df_train[inp]
    #print(X_training.columns)
    print()

    scaler.fit(X_training)
    trainX = scaler.transform(X_training)

    df_test = pd.read_csv('Datasets/test.csv')  # ,  delim_whitespace=True)
    X_test = df_test.drop(inp, axis=1)
    y_test = df_test[inp]

    testX = scaler.transform(X_test)

    return trainX, y_training, testX, y_test

def save_model_stats(y_pred_prob, y_pred, y_test, clf, i):

    if i == 1:
        folder_name = 'SGD'
    elif i == 2:
        folder_name = 'SVM'
    elif i == 3:
        folder_name = 'XGB'
    
    print("Specify the name of the file that will hold the predicted probabilities, predicted labels and actual labels of the testing set")
    inp = str(input(""))

    SGD_test_results = {folder_name + ' Probability': y_pred_prob, folder_name + ' Predicted Label': y_pred, 'Actual Label': y_test}
    df_SGD_test_results = pd.DataFrame.from_dict(SGD_test_results)
    df_SGD_test_results.to_csv(folder_name + '-Results/'+inp+'.csv', index=False, header=True)

    print('Accuracy Score (AUC): {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))

    print("Would you like to save your model? [Y/N]")
    while True:
        try:
            inp = str(input(''))
            if inp != 'Y' and inp != 'N':
                raise ValueError #this will send it to the print message and back to the input option
            elif(inp == 'Y'):
                print("Specify name of the new model")
                inp2 = str(input(''))
                with open('./Models/'+inp2+'.pkl','wb') as f:
                    pickle.dump(clf,f)

            elif(inp == 'N'):
                break
            break
        except ValueError:
            print("Invalid option. Type Y or N")

    pick_model()




def SGD_untuned():
    print("Specify target, ensure the name matches exactly with the header in the train.csv file\n")
    print("You may choose a value from the list below\n")
    print(feature_list())

    while True:
        try:
            inp = str(input(''))
            if inp not in feature_list():
                raise ValueError #this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid option. Choose a Value from the list above")


    X_train, y_train, X_test, y_test = get_training_and_testing_data(inp)

    clf = SGDClassifier(random_state=0, loss="log_loss", class_weight="balanced",
                        max_iter=10, alpha=0.01, penalty='l2')

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    save_model_stats(y_pred_prob, y_pred, y_test, clf, 1)

def SGD_tuned():
    print("Specify target, ensure the name matches exactly with the header in the train.csv file\n")
    print("You may choose a value from the list below\n")
    print(feature_list())

    while True:
        try:
            inp = str(input(''))
            if inp not in feature_list():
                raise ValueError #this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid option. Choose a Value from the list above")

    print("c")

    X_train, y_train, X_test, y_test = get_training_and_testing_data(inp)

    estimator = SGDClassifier(random_state=0, loss="log_loss", class_weight="balanced",
                        max_iter=10, alpha=0.01, penalty='l2')
    parameters = {
        'max_iter': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'alpha':  [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}   

    clf = GridSearchCV(estimator=estimator,param_grid=parameters,scoring = 'roc_auc',n_jobs = 10,cv = 10,verbose=10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    save_model_stats(y_pred_prob, y_pred, y_test, clf, 1)


def SGD():
    print("Would you like your model tuned? [Y/N]")
    while True:
        try:
            inp = str(input(''))
            if inp != 'Y' and inp != 'N':
                raise ValueError #this will send it to the print message and back to the input option
            elif(inp == 'Y'):
                tuned = 1
            elif(inp == 'N'):
                tuned = 0
            break
        except ValueError:
            print("Invalid option. Type Y or N")

    if tuned == 1:
        SGD_tuned()
    elif tuned == 0:
        SGD_untuned()

def SVM_tuned():
    print("Specify target, ensure the name matches exactly with the header in the train.csv file\n")
    print("You may choose a value from the list below\n")
    print(feature_list())

    while True:
        try:
            inp = str(input(''))
            if inp not in feature_list():
                raise ValueError #this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid option. Choose a Value from the list above")


    X_train, y_train, X_test, y_test = get_training_and_testing_data(inp)

    svm_model  = svm.SVC(random_state=0, class_weight='balanced', probability=True)

    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']} 

    clf = GridSearchCV(svm_model, param_grid, refit = True, scoring = 'roc_auc', verbose = 3)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    save_model_stats(y_pred_prob, y_pred, y_test, clf, 2)

def SVM_untuned():
    print("Specify target, ensure the name matches exactly with the header in the train.csv file\n")
    print("You may choose a value from the list below\n")
    print(feature_list())

    while True:
        try:
            inp = str(input(''))
            if inp not in feature_list():
                raise ValueError #this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid option. Choose a Value from the list above")


    X_train, y_train, X_test, y_test = get_training_and_testing_data(inp)

    clf  = svm.SVC(random_state=0, class_weight='balanced', probability=True)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    save_model_stats(y_pred_prob, y_pred, y_test, clf, 1)
    

def SVM():
    print("Would you like your model tuned? [Y/N]")
    while True:
        try:
            inp = str(input(''))
            if inp != 'Y' and inp != 'N':
                raise ValueError #this will send it to the print message and back to the input option
            elif(inp == 'Y'):
                tuned = 1
            elif(inp == 'N'):
                tuned = 0
            break
        except ValueError:
            print("Invalid option. Type Y or N")

    if tuned == 1:
        SVM_tuned()
    elif tuned == 0:
        SVM_untuned()

def xgb_tuned():
    print("Specify target, ensure the name matches exactly with the header in the train.csv file\n")
    print("You may choose a value from the list below\n")
    print(feature_list())

    while True:
        try:
            inp = str(input(''))
            if inp not in feature_list():
                raise ValueError #this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid option. Choose a Value from the list above")


    X_train, y_train, X_test, y_test = get_training_and_testing_data(inp)

    print("(EXTREMELY IMPORTANT) Specify the weight of the positive class e.g. total instances divided by positive instances")
    inp = float(input(''))

    estimator = XGBClassifier(random_state=0, scale_pos_weight=inp, objective = 'binary:logistic', tree_method = 'hist')

    parameters = {
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05]}   

    clf = GridSearchCV(estimator=estimator,param_grid=parameters,scoring = 'roc_auc',n_jobs = 10,cv = 10,verbose=10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    save_model_stats(y_pred_prob, y_pred, y_test, clf, 3)

def xgb_untuned():
    print("Specify target, ensure the name matches exactly with the header in the train.csv file\n")
    print("You may choose a value from the list below\n")
    print(feature_list())

    while True:
        try:
            inp = str(input(''))
            if inp not in feature_list():
                raise ValueError #this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid option. Choose a Value from the list above")


    X_train, y_train, X_test, y_test = get_training_and_testing_data(inp)

    print("(EXTREMELY IMPORTANT) Specify the weight of the positive class e.g. total instances divided by positive instances")
    inp = float(input(''))
    clf = XGBClassifier(random_state=0, scale_pos_weight=inp, objective = 'binary:logistic', tree_method = 'hist')


    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    save_model_stats(y_pred_prob, y_pred, y_test, clf, 3)


def xgb():
    print("Would you like your model tuned? [Y/N]")
    while True:
        try:
            inp = str(input(''))
            if inp != 'Y' and inp != 'N':
                raise ValueError #this will send it to the print message and back to the input option
            elif(inp == 'Y'):
                tuned = 1
            elif(inp == 'N'):
                tuned = 0
            break
        except ValueError:
            print("Invalid option. Type Y or N")

    if tuned == 1:
        xgb_tuned()
    elif tuned == 0:
        xgb_untuned()

def all_models():

    # Get the list of all files and directories
    path = "./Models"
    dir_list = os.listdir(path)
 
    print("Specify Model you wish to retest")
 
    # prints all files
    print(dir_list)

    while True:
        try:
            inp = str(input(''))
            if inp not in dir_list:
                raise ValueError #this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid option. Choose a Value from the list above")

    print("Does your retest Dataset have the Actual labels? [Y/N]")
    while True:
        try:
            lab = str(input(''))
            if lab != 'Y' and lab != 'N':
                raise ValueError #this will send it to the print message and back to the input option
            elif(lab == 'Y'):
                labels = 1
            elif(lab == 'N'):
                labels = 0
            break
        except ValueError:
            print("Invalid option. Type Y or N")

    print("Specify dataset you wish to retest")
    path = "./datasets"
    dir_list = os.listdir(path)
    print(dir_list)
    while True:
        try:
            retest_inp = str(input(''))
            if retest_inp not in dir_list:
                raise ValueError #this will send it to the print message and back to the input option
            break
        except ValueError:
            print("Invalid option. Choose a Value from the list above")

    scaler = StandardScaler()

    X_retest = pd.read_csv('Datasets/'+retest_inp)

    if labels == 1:
        print("Specify Actual label header name")
        target = str(input(''))
        y_true = X_retest[target]
        X_retest = X_retest.drop(target, axis=1)


    print("Does your retest set have ID's? [Y/N]")
    while True:
        try:
            ids = str(input(''))
            if ids != 'Y' and ids != 'N':
                raise ValueError #this will send it to the print message and back to the input option
            elif(ids == 'Y'):
                id_label = 1
            elif(ids == 'N'):
                id_label = 0
            break
        except ValueError:
            print("Invalid option. Type Y or N")

    if id_label == 1:
        print("Specify ID header name")
        id_header = str(input(''))
        x_ids = X_retest[id_header]
        X_retest = X_retest.drop(id_header, axis=1)


        

    scaler.fit(X_retest)
    X_retest = scaler.transform(X_retest)

    pickled_model = pickle.load(open('./Models/'+inp, 'rb'))
    y_pred = pickled_model.predict(X_retest)
    y_pred_prob = pickled_model.predict_proba(X_retest)[:, 1]

    if id_label == 1:

        print('Accuracy Score (AUC): {:.2f}'.format(accuracy_score(y_true, y_pred)))
        print("Confusion matrix")
        print(confusion_matrix(y_true, y_pred))

    print("Does you want to save your data? [Y/N]")
    while True:
        try:
            save = str(input(''))
            if save != 'Y' and save != 'N':
                raise ValueError #this will send it to the print message and back to the input option
            elif(save == 'Y'):
                saved = 1
            elif(lab == 'N'):
                saved = 0
            break
        except ValueError:
            print("Invalid option. Type Y or N")

    if saved == 1:

        print("Specify the name of the file that will hold the predicted probabilities and predicted labels of this retest set")
        inp2 = str(input(""))

        if id_label == 1 and labels == 1:
            retest_results = {'ID': x_ids, 'Model Probability': y_pred_prob, 'Model Predicted Label': y_pred, 'Data Actual': y_true}
        
        if id_label == 1 and labels == 0:
            retest_results = {'ID': x_ids, 'Model Probability': y_pred_prob, 'Model Predicted Label': y_pred}

        elif id_label == 0 and labels == 0:
            retest_results = {'Model Probability': y_pred_prob, 'Model Predicted Label': y_pred}

        elif id_label == 0 and labels == 1:
            retest_results = {'Model Probability': y_pred_prob, 'Model Predicted Label': y_pred, 'Data Actual': y_true}

        test_results = pd.DataFrame.from_dict(retest_results)
        test_results.to_csv('Saved-Model-Results/'+inp2+'.csv', index=False, header=True)

    pick_model()


def intro():
    print("Welcome to Hydro - A Machine Learning application developed by the University of Maryland, Baltimore")
    print("Maryland Psychiatric Research Center")
    print("You will be given the choice of 3 Models which you can run your data on")
    print("Ensure the training and testing set are in the \'datasets\' folder labeled train.csv and test.csv\n")

def pick_model():
    print("Pick an option simply by typing the number next to the option:")
    print("1 - Schastic Gradient Descent linear Model")
    print("2 - Support Vector Machines")
    print("3 - eXtreme Gradient Boosting")
    print("4 - Test previously saved models")
    print("5 - Exit")

    while True:
        try:
            number1 = int(input(''))
            if number1 < 1 or number1 > 5:
                raise ValueError #this will send it to the print message and back to the input option
            elif(number1 == 1):
                model = 1
            elif(number1 == 2):
                model = 2
            elif(number1 == 3):
                model = 3
            elif(number1 == 4):
                model = 4
            elif(number1 == 5):
                model = 5
            break
        except ValueError:
            print("Invalid integer. The number must be in the range of 1-4.")

    if model == 1:
        SGD()
    elif model == 2:
        SVM()
    elif model == 3:
        xgb()
    elif model == 4:
        all_models()
    elif model == 5:
        print("Thank you for using - Press Control C to quit")


intro()
pick_model()
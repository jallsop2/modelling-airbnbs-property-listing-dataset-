from tabular_data import load_airbnb
from classification_hyperparmeter_tuning import tune_classification_model_hyperparameters
from save_models import save_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib

def visualise_confusion_matrix(true_values, pred_values, normalise=False):

    cf = confusion_matrix(true_values, pred_values)

    if normalise == True:
        cf = cf / cf.sum()

    display = ConfusionMatrixDisplay(cf)
    display.plot()
    plt.show()


def tune_classification_models(data):

    X, y = data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    
    models = []

    models.append(["logistic_regression", LogisticRegression(), {
        "C" : [0.01, 0.1, 0.5, 1],
        "max_iter" : [100, 500],
        "random_state" : [1]
    }])

    """ models.append(["decision_tree", DecisionTreeClassifier(), {
        "max_depth" : [2, 5, 10, 100, None],
        "min_samples_split" : [1, 2, 5, 10],
        "min_samples_leaf" : [1, 2, 5, 10, 15, 20, 25],
        "random_state" : [1]
    } ])
 """
    models.append(["random_forest", RandomForestClassifier(), {
        "n_estimators" : [10, 100, 200],
        "max_depth" : [2, 5, 10],
        "min_samples_leaf" : [5, 10],
        "min_samples_split" : [5, 10, 15],
        "min_weight_fraction_leaf" : [0, 0.01, 0.05, 0.1],
        "random_state" : [1]
    } ]) 

    """ models.append(["gradient_boosting", GradientBoostingClassifier(), {
        "subsample" : [0.5, 1],
        "min_samples_leaf" : [1, 2, 5, 10],
        "max_depth" : [1, 2, 3, 4],
        "n_iter_no_change" : [5, 20, None],
        "random_state" : [1]
    }]) """



    for i in range(len(models)):

        best_model , best_params, best_metrics = tune_classification_model_hyperparameters(models[i][1], X_train, X_test, y_train, y_test, models[i][2])

        print(f"\n {models[i][0]} \n")
        print(best_params)
        #print(best_model.coef_)

        print(f"\nThe train accuracy is {best_metrics['train_accuracy']}")
        print(f"\nThe validation accuracy is {best_metrics['validation_accuracy']}")
        print(f"\nThe test accuracy is {best_metrics['test_accuracy']}")

        path = f"models/classification/{models[i][0]}"

        test_pred = best_model.predict(X_test)

        visualise_confusion_matrix(y_test, test_pred)

        #save_model(best_model, best_params, best_metrics, path)


def evaluate_classification_models(models, X, y, n):

    num_models = len(models)

    train_accuracy = np.zeros([num_models])
    test_accuracy = np.zeros([num_models])

    #counter = np.zeros([num_models])

    for i in range(n):

        print(f" {i}", end = "\r")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        current_train_accuracy = []
        current_test_accuracy = []

        for j in range(num_models):
            params = models[j][2]
            model = models[j][1](**params)
            model.fit(X_train,y_train)
            current_train_accuracy.append(model.score(X_train, y_train))
            current_test_accuracy.append(model.score(X_test, y_test))

        train_accuracy += np.array(current_train_accuracy)
        test_accuracy += np.array(current_test_accuracy)

    train_accuracy = train_accuracy/n
    test_accuracy = test_accuracy/n


    for j in range(num_models):
        print(train_accuracy[j], test_accuracy[j], models[j][0])


    

def test_and_save_classification_models(models, num_tests, X, y):

    num_models = len(models)

    avg_train_accuracy = np.zeros([num_models])
    avg_test_accuracy = np.zeros([num_models])

    for i in range(num_tests):

        print(f"Test {i+1}", end = "\r")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        for j in range(num_models):
            params = models[j][2]
            model = models[j][1](**params)
            model.fit(X_train,y_train)
            avg_train_accuracy[j] += model.score(X_train, y_train)
            avg_test_accuracy[j] += model.score(X_test, y_test)


    avg_train_accuracy = avg_train_accuracy/num_tests
    avg_test_accuracy = avg_test_accuracy/num_tests

    for j in range(num_models):
        params = models[j][2]
        model = models[j][1](**params)
        model.fit(X, y)

        metrics = {
        "average training accuracy" : avg_train_accuracy[j],
        "average test accuracy" : avg_test_accuracy[j]
        }

        path = f"models/classification/{models[j][0]}"

        print(models[j][0])
        print(metrics)

        save_model(model, params, metrics, path)




def find_best_saved_classification_model():
    
    dirs = os.listdir("models/classification")
    best_accuracy = 0

    for dir in dirs:

        metrics_file = open(f"models/classification/{dir}/metrics.json")
        metrics = json.loads(metrics_file.read())
        accuracy = metrics["test_accuracy"]
        print(accuracy)
        
        if accuracy > best_accuracy:
            hyperparameter_file = open(f"models/classification/{dir}/hyperparameters.json")
            hyperparameters = json.loads(hyperparameter_file.read())
            best_model_info = [dir, joblib.load(f"models/classification/{dir}/model.joblib"), hyperparameters, metrics]
            best_accuracy = accuracy

    return best_model_info


if __name__ == "__main__":
    data, labels = load_airbnb(category=True)

    y = data[:,-1]
    #print(y)
    X = data[:,:-1]

    scaler = StandardScaler()

    scaler.fit(X)
    X = scaler.transform(X)



    models = []

    models.append(["logistic_regression", LogisticRegression, {
        "random_state" : 1
    }])

    models.append(["logistic_regression", LogisticRegression, {
        "C" : 0.25,
        "random_state" : 1
    }])

    models.append(["decision_tree", DecisionTreeClassifier, {
        "max_depth" : None,
        "min_samples_split" : 150,
        "min_samples_leaf" : 25,
        "random_state" : 1
    } ])

    models.append(["random_forest", RandomForestClassifier, {
        "n_estimators" : 100,
        "max_depth" : None,
        "min_samples_leaf" : 3,
        "min_samples_split" : 20,
        "random_state" : 1
    } ]) 


    models.append(["gradient_boosting", GradientBoostingClassifier, {
        "subsample" : 0.75,
        "learning_rate" : 0.01,
        "n_estimators" : 1000,
        "min_samples_leaf" : 5,
        "min_samples_split" : 4,
        "max_depth" : 2,
        "n_iter_no_change" : None,
        "random_state" : 1
    }])


 
    #evaluate_classification_models(models, X, y, 100)

    test_and_save_classification_models(models, 1000, X, y)

    #print(find_best_classification_model())





    
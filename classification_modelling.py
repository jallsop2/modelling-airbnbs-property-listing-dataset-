from tabular_data import load_airbnb
from classification_hyperparmeter_tuning import tune_classification_model_hyperparameters
from save import save_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import numpy as np
import matplotlib.pyplot as plt

def visualise_confusion_matrix(true_values, pred_values, normalise=False):

    cf = confusion_matrix(true_values, pred_values)

    if normalise == True:
        cf = cf / cf.sum()

    display = ConfusionMatrixDisplay(cf)
    display.plot()
    plt.show()


def evaluate_classification_models(data):

    X, y = data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    
    models = []

    models.append(["logistic_regression", LogisticRegression(), {
        "C" : [0.01, 0.1, 0.5, 1],
        "max_iter" : [100,500, 1000, 5000, 10000],
        "random_state" : [1]
    }])

    models.append(["decision_tree", DecisionTreeClassifier(), {
        "max_depth" : [2, 5, 10, 100, 1000, None],
        "min_samples_split" : [1, 2, 5, 10],
        "min_samples_leaf" : [1, 2, 5, 10, 15, 20, 25],
        "random_state" : [1]
    } ])

    """ models.append(["random_forest", RandomForestClassifier(), {
        "n_estimators" : [10, 100, 500],
        "max_depth" : [2, 5, 10, 100, None],
        "min_samples_leaf" : [1, 5, 10, 15, 20],
        "min_samples_split" : [1, 2, 5, 10],
        "random_state" : [1]
    } ])  """

    models.append(["gradient_boosting", GradientBoostingClassifier(), {
        "subsample" : [0.5, 1],
        "min_samples_leaf" : [1, 2, 5, 10],
        "max_depth" : [1, 2, 3, 4],
        "n_iter_no_change" : [5, 20, None],
        "random_state" : [1]
    }])



    for i in range(len(models)):

        best_model , best_params, best_metrics = tune_classification_model_hyperparameters(models[i][1], X_train, X_test, y_train, y_test, models[i][2])

        print(f"\n {models[i][0]} \n")
        print(best_params)
        #print(best_model.coef_)

        print(f"\nThe train accuracy is {best_metrics['train_accuracy']}")
        print(f"\nThe validation accuracy is {best_metrics['validation_accuracy']}")
        print(f"\nThe test accuracy is {best_metrics['test_accuracy']}")

        path = f"models/classification/{models[i][0]}"

        save_model(best_model, best_params, best_metrics, path)

if __name__ == "__main__":
    data, labels = load_airbnb(category=True)

    y = data[:,-1]
    #print(y)
    X = data[:,:-1]

    scaler = StandardScaler()

    scaler.fit(X)
    X = scaler.transform(X)

    evaluate_classification_models([X,y])





    
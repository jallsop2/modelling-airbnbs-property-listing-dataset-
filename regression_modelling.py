from tabular_data import load_airbnb
from regression_hyperparameter_tuning import custom_tune_regression_model_hyperparameters, tune_regression_model_hyperparameters
from save import save_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import os
import json
import joblib


def evaluate_regression_models(data):

    X, y = data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    
    models = []

    models.append(["linear_regression", SGDRegressor(), {
        "alpha" : [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1, 0.5],
        "max_iter" : [100, 500, 1000, 5000, 10000, 100000],
        "eta0" : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        "random_state" : [1]
    }])

    models.append(["regression_tree", DecisionTreeRegressor(), {
        "max_depth" : [5, 10, 100, 1000, None],
        "min_samples_split" : [1, 2, 5, 10],
        "min_samples_leaf" : [1, 2, 5, 10, 15, 20, 25],
        "random_state" : [1]
    } ])

    models.append(["random_forest", RandomForestRegressor(), {
        "n_estimators" : [100, 500],
        "max_depth" : [10, 100, None],
        "min_samples_leaf" : [1, 5, 10, 15, 20],
        "random_state" : [1]
    } ]) 

    models.append(["gradient_boosting", GradientBoostingRegressor(), {
        "subsample" : [0.25, 0.5, 0.66, 0.75],
        "min_samples_leaf" : [1, 5, 10, 15, 20],
        "max_depth" : [1, 2, 3, 4, 5, 10],
        "n_iter_no_change" : [5, 10, 20],
        "random_state" : [1]
    }])



    for i in range(len(models)):

        best_model , best_params, best_score = tune_regression_model_hyperparameters(models[i][1], X_train, y_train, models[i][2])

        print(f"\n {models[i][0]} \n")
        print(best_params)
        #print(best_model.coef_)

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        print(f"\nThe score was {best_score}")

        training_RMSE = mean_squared_error(y_train,y_train_pred,squared=False)
        training_r2_score = r2_score(y_train,y_train_pred)
        test_RMSE = mean_squared_error(y_test,y_test_pred,squared=False)
        test_r2_score = r2_score(y_test,y_test_pred)

        print(f"Training set: RMSE = {training_RMSE}, r2 score = {training_r2_score}")
        print(f"Test set: RMSE = {test_RMSE}, r2 score = {test_r2_score}\n\n")

        metrics = {
            "Training RMSE" : training_RMSE,
            "Training r2 score" : training_r2_score,
            "Test RMSE" : test_RMSE,
            "Test r2 score" : test_r2_score
        }

        path = f"models/regression/{models[i][0]}"

        save_model(best_model, best_params, metrics, path)


def find_best_model():
    
    dirs = os.listdir("models/regression")
    best_RMSE = np.infty

    for dir in dirs:

        metrics_file = open(f"models/regression/{dir}/metrics.json")
        metrics = json.loads(metrics_file.read())
        RMSE = metrics["Test RMSE"]
        print(RMSE)
        
        if RMSE < best_RMSE:
            hyperparameter_file = open(f"models/regression/{dir}/hyperparameters.json")
            hyperparameters = json.loads(hyperparameter_file.read())
            best_model_info = [dir, joblib.load(f"models/regression/{dir}/model.joblib"), hyperparameters, metrics]
            best_RMSE = RMSE

    return best_model_info


if __name__ == "__main__":

    data, labels = load_airbnb()

    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    y = data[:,3]
    X = np.concatenate((data[:,:3], data[:,4:]),1)
    #X = data[:,:3]
    #X = np.concatenate((data[:,:3], data[:,10:]),1)

    evaluate_regression_models([X,y])


    print(find_best_model())

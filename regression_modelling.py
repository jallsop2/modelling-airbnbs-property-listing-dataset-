from tabular_data import load_airbnb
from regression_hyperparameter_tuning import custom_tune_regression_model_hyperparameters, tune_regression_model_hyperparameters
from save_models import save_model

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


def tune_regression_models(data):

    X, y = data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    
    models = []

    """ models.append(["linear_regression", SGDRegressor(), {
        "alpha" : [0.001, 0.005, 0.01, 0.1, 0.2, 0.5],
        "max_iter" : [50, 100, 500],
        "eta0" : [0.005, 0.01, 0.05],
        "random_state" : [1]
    }]) """

    """ models.append(["regression_tree", DecisionTreeRegressor(), {
        "max_depth" : [5, 10, 100, 1000, None],
        "min_samples_split" : [1, 2, 5, 10],
        "min_samples_leaf" : [1, 2, 5, 10, 15, 20, 25],
        "random_state" : [1]
    } ]) """

    """ models.append(["random_forest", RandomForestRegressor(), {
        "n_estimators" : [100, 500],
        "max_depth" : [10, 100, None],
        "min_samples_leaf" : [1, 5, 10, 15, 20],
        "random_state" : [1]
    } ])  """

    models.append(["gradient_boosting", GradientBoostingRegressor(), {
        "subsample" : [0.25, 0.5, 0.75, 1],
        "min_samples_leaf" : [1, 5, 10, 15, 20],
        "max_depth" : [1, 2, 3, 4, 5, 10],
        "n_iter_no_change" : [5, 10, 20, None],
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



def evaluate_regression_models(models, X, y, n):

    num_models = len(models)

    train_RMSE = np.zeros([num_models])
    test_RMSE = np.zeros([num_models])

    #counter = np.zeros([num_models])

    for i in range(n):
        #print(test_RMSE)

        print(f" {i}", end = "\r")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        current_train_RMSE = []
        current_test_RMSE = []

        for j in range(num_models):
            params = models[j][2]
            model = models[j][1](**params)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            current_train_RMSE.append(mean_squared_error(y_train,y_train_pred,squared=False))
            current_test_RMSE.append(mean_squared_error(y_test,y_test_pred,squared=False))

        train_RMSE += np.array(current_train_RMSE)
        test_RMSE += np.array(current_test_RMSE)

    train_RMSE = train_RMSE/n
    test_RMSE = test_RMSE/n


    for i in range(num_models):
        print(train_RMSE[i], test_RMSE[i])
    
    #print(counter)

def test_and_save_regression_model(models, num_tests, X, y):

    num_models = len(models)

    avg_train_RMSE = np.zeros([num_models])
    avg_test_RMSE = np.zeros([num_models])

    for i in range(num_tests):

        print(f"Test {i+1}", end = "\r")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        for j in range(num_models):
            params = models[j][2]
            model = models[j][1](**params)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            avg_train_RMSE[j] += mean_squared_error(y_train,y_train_pred,squared=False)
            avg_test_RMSE[j] += mean_squared_error(y_test,y_test_pred,squared=False)


    avg_train_RMSE = avg_train_RMSE/num_tests
    avg_test_RMSE = avg_test_RMSE/num_tests

    for j in range(num_models):
        params = models[j][2]
        model = models[j][1](**params)
        model.fit(X, y)

        metrics = {
        "average training RMSE" : avg_train_RMSE[j],
        "average test RMSE" : avg_test_RMSE[j]
        }

        path = f"models/regression/{models[j][0]}"

        save_model(model, params, metrics, path)

        counter += 1


def find_best_saved_regression_model():
    
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


    models = {}

    models["linear_regression", SGDRegressor] = {
        "random_state" : 1
    }

    """ models["linear_regression", SGDRegressor] = {
        "alpha" : 0.1,
        "max_iter" : 10000,
        "learning_rate" : "adaptive",
        "eta0" : 0.01,
        "n_iter_no_change" : 100,
        "random_state" : 1
    }

    models["regression_tree", DecisionTreeRegressor] =  {
        "max_depth" : None,
        "min_samples_split" : 75,
        "min_samples_leaf" : 25,
        "random_state" : 1
    }

    models["random_forest", RandomForestRegressor] = {
        "n_estimators" : 1000,
        "max_depth" : 10,
        "min_samples_leaf" : 15,
        "min_samples_split" : 30,
        "random_state" : 1
    }

    models["gradient_boosting", GradientBoostingRegressor] = {
        "n_estimators" : 100,
        "subsample" : 0.5,
        "min_samples_leaf" : 15,
        "min_samples_split" : 60,
        "max_depth" : 2,
        "random_state" : 1
    } """

   

    evaluate_regression_models(models, X, y, 100000)

    #test_and_save_model(models, 10000, X, y)

    #print(find_best_saved_regression_model())

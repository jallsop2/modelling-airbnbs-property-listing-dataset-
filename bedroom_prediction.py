from tabular_data import load_airbnb
from neural_network_modelling import AirbnbNightlyPriceImageDataset, train_test_NN_model
from regression_modelling import train_test_regression_model

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib
import json
import os
import numpy as np



def evaluate_regression_models(models, data, prediction_index, n, train_size=0.7):

    num_models = len(models)

    train_RMSE = np.zeros([num_models])
    test_RMSE = np.zeros([num_models])


    for i in range(n):

        print(f"Test {i}", end = "\r")

        num_data_points = data.shape[0]

        indices = range(num_data_points)

        train_indices, test_indices = train_test_split(indices, train_size= train_size)

        train_pytorch_dataset = AirbnbNightlyPriceImageDataset(train_indices, prediction_index)
        test_pytorch_dataset = AirbnbNightlyPriceImageDataset(test_indices, prediction_index)

        train_numpy_dataset = data[train_indices,:]
        test_numpy_dataset = data[test_indices,:] 

        current_train_RMSE = []
        current_test_RMSE = []

        for j in range(num_models):
            params = models[j][2]
            model_class = models[j][1]

            if model_class == "NN":
                metrics = train_test_NN_model(params, train_pytorch_dataset, test_pytorch_dataset)
            else:
                metrics = train_test_regression_model(model_class, params, train_numpy_dataset, test_numpy_dataset, prediction_index)

            current_train_RMSE.append(metrics[0])
            current_test_RMSE.append(metrics[1])

        train_RMSE += np.array(current_train_RMSE)
        test_RMSE += np.array(current_test_RMSE)

    print("Testing Complete.")

    train_RMSE = train_RMSE/n
    test_RMSE = test_RMSE/n


    for j in range(num_models):
        print(models[j][0])
        print(train_RMSE[j], test_RMSE[j])
        print()

    return train_RMSE, test_RMSE


data, labels = load_airbnb()

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)


models = []

models.append(["linear_regression", SGDRegressor, {
    "alpha" : 0.1,
    "max_iter" : 10000,
    "learning_rate" : "adaptive",
    "eta0" : 0.01,
    "n_iter_no_change" : 100,
    "random_state" : 1
}])

models.append(["regression_tree", DecisionTreeRegressor, {
    "max_depth" : None,
    "min_samples_split" : 75,
    "min_samples_leaf" : 25,
    "random_state" : 1
}])

models.append(["random_forest", RandomForestRegressor, {
    "n_estimators" : 1000,
    "max_depth" : 10,
    "min_samples_leaf" : 15,
    "min_samples_split" : 30,
    "random_state" : 1
}])

models.append(["gradient_boosting", GradientBoostingRegressor, {
    "n_estimators" : 100,
    "subsample" : 0.5,
    "min_samples_leaf" : 15,
    "min_samples_split" : 60,
    "max_depth" : 2,
    "random_state" : 1
}])

models.append(["neural_network", "NN", {
    "optimiser": "torch.optim.Adam",
    "optimiser_hyperparameters": { 
        "lr": 0.01,
        "weight_decay": 0.05},
    "epochs": 50,
    "hidden_layer_width": 8,
    "depth": 2,
    "batch_size": 100
}])


models.append(["neural_network", "NN", {
    "optimiser": "torch.optim.Adam",
    "optimiser_hyperparameters": { 
        "lr": 0.005,
        "weight_decay": 0.05},
    "epochs": 50,
    "hidden_layer_width": 32,
    "depth": 4,
    "batch_size": 100
}])




prediction_index = 2
n = 100
train_size = 0.7

evaluate_regression_models(models, data, prediction_index, n, train_size)
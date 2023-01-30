import joblib
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def save_model(model, params, metrics, path):
    joblib.dump(model, path+"/model.joblib")

    json_params = json.dumps(params, indent=4)
    with open(f"{path}/hyperparameters.json","w") as file:
        file.write(json_params)

    json_metrics = json.dumps(metrics, indent=4)
    with open(f"{path}/metrics.json","w") as file:
        file.write(json_metrics)



def test_and_save_model(models, num_tests, type, X, y):

    num_models = len(models)

    avg_train_RMSE = np.zeros([num_models])
    avg_test_RMSE = np.zeros([num_models])

    for i in range(num_tests):

        print(f"Test {i+1}", end = "\r")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        counter = 0

        for model_type, params in models.items():
            model = model_type[1](**params)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            avg_train_RMSE[counter] += mean_squared_error(y_train,y_train_pred,squared=False)
            avg_test_RMSE[counter] += mean_squared_error(y_test,y_test_pred,squared=False)

            counter += 1

    avg_train_RMSE = avg_train_RMSE/num_tests
    avg_test_RMSE = avg_test_RMSE/num_tests

    counter = 0
    for model_type, params in models.items():
        model = model_type[1](**params)
        model.fit(X, y)

        metrics = {
        "average training RMSE" : avg_train_RMSE[counter],
        "average test RMSE" : avg_test_RMSE[counter]
        }

        path = f"models/{type}/{model_type[0]}"

        save_model(model, params, metrics, path)

        counter += 1
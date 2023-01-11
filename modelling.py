from tabular_data import load_airbnb
from regression_hyperparameter_tuning import custom_tune_regression_model_hyperparameters, tune_regression_model_hyperparameters
from save import save_model

from sklearn import model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np


data, labels = load_airbnb()

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

y = data[:,3]
#X = data[:,:3]
X = np.concatenate((data[:,:3], data[:,4:]),1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

#X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

hyperparameters = {
    "alpha" : [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1, 0.5],
    "max_iter" : [100, 500, 1000, 5000, 10000, 100000],
    "eta0" : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
}

""" model, arg_dict, metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters)

print(arg_dict)
print(metrics)

y_test_pred = model.predict(X_test)
print(f"Test set: RMSE = {mean_squared_error(y_test,y_test_pred,squared=False)}, r2 score = {r2_score(y_test,y_test_pred)}") """

model , params, score = tune_regression_model_hyperparameters(X_train, y_train, hyperparameters)

print(params)
print(model.coef_)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print(f"\nThe score was {score}")

training_RMSE = mean_squared_error(y_train,y_train_pred,squared=False)
training_r2_score = r2_score(y_train,y_train_pred)
test_RMSE = mean_squared_error(y_test,y_test_pred,squared=False)
test_r2_score = r2_score(y_test,y_test_pred)
print(f"Training set: RMSE = {training_RMSE}, r2 score = {training_r2_score}")
print(f"Test set: RMSE = {test_RMSE}, r2 score = {test_r2_score}")

metrics = {
    "Training RMSE" : training_RMSE,
    "Training r2 score" : training_r2_score,
    "Test RMSE" : test_RMSE,
    "Test r2 score" : test_r2_score
}

path = "models/regression/linear_regression"

save_model(model, params, metrics, path)

#print(model.coef_)
#print(X_train.shape, X_validation.shape, X_test.shape)
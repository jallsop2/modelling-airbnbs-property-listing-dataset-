from tabular_data import load_airbnb

from sklearn import model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


import numpy as np
from itertools import product


def custom_tune_regression_model_hyperparameters(model_type, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters):

    range_list = []

    best_RMSE = np.infty


    for list in hyperparameters.values():
        range_list.append(len(list))
    
    for i in product(*map(range, range_list)):
        arg_dict = {}

        counter = 0
        for key, value in hyperparameters.items():
            arg_dict[key] = value[i[counter]]
            counter += 1
        #print(arg_dict)
        model = SGDRegressor(**arg_dict)

        model.fit(X_train,y_train)
        y_validation_pred = model.predict(X_validation)
        validation_RMSE = mean_squared_error(y_validation, y_validation_pred, squared=False)
        print(validation_RMSE, model)

        if validation_RMSE < best_RMSE:

            y_train_pred = model.predict(X_train)
            train_RMSE = mean_squared_error(y_train, y_train_pred, squared=False)
            train_r2_score = r2_score(y_train, y_train_pred)
            validation_r2_score = r2_score(y_validation, y_validation_pred)

            best_RMSE = validation_RMSE
            best_model = model
            best_hyperparameters = arg_dict
            best_metrics = {
                "train_RMSE" : train_RMSE,
                "validation_RMSE" : validation_RMSE,
                "train_r2_score" : train_r2_score,
                "validation_r2_score" : validation_r2_score
            }
        
        model = None

        #model = SGDRegressor(alpha=1e-05, max_iter=100, eta0=0.001)

        #print(i)

    return best_model, best_hyperparameters, best_metrics

data, labels = load_airbnb()

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

y = data[:,3]
#X = data[:,:3]
X = np.concatenate((data[:,:3], data[:,4:]),1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

hyperparameters = {
    "alpha" : [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1, 0.5],
    "max_iter" : [100, 500, 1000, 5000, 10000, 1000000],
    "eta0" : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
}

model, arg_dict, metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters)

print(arg_dict)
print(metrics)

y_test_pred = model.predict(X_test)
print(f"Test set: RMSE = {mean_squared_error(y_test,y_test_pred,squared=False)}, r2 score = {r2_score(y_test,y_test_pred)}")


""" model = SGDRegressor(max_iter=1000000)
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print(f"Training set: RMSE = {mean_squared_error(y_train,y_train_pred,squared=False)}, r2 score = {r2_score(y_train,y_train_pred)}")
print(f"Test set: RMSE = {mean_squared_error(y_test,y_test_pred,squared=False)}, r2 score = {r2_score(y_test,y_test_pred)}") """

#print(model.coef_)
#print(X_train.shape, X_validation.shape, X_test.shape)
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
from itertools import product
import numpy as np


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
        model = model_type(**arg_dict, random_state=1)

        model.fit(X_train,y_train)
        y_validation_pred = model.predict(X_validation)
        validation_RMSE = mean_squared_error(y_validation, y_validation_pred, squared=False)
        print(validation_RMSE, model)

        if validation_RMSE < best_RMSE:

            best_RMSE = validation_RMSE
            best_model = model

                    
        model = None

        #model = SGDRegressor(alpha=1e-05, max_iter=100, eta0=0.001)

        #print(i)

    y_train_pred = best_model.predict(X_train)
    y_validation_pred = best_model.predict(X_validation)

    train_RMSE = mean_squared_error(y_train, y_train_pred, squared=False)
    train_r2_score = r2_score(y_train, y_train_pred)
    validation_r2_score = r2_score(y_validation, y_validation_pred)

    best_hyperparameters = arg_dict
    best_metrics = {
        "train_RMSE" : train_RMSE,
        "validation_RMSE" : validation_RMSE,
        "train_r2_score" : train_r2_score,
        "validation_r2_score" : validation_r2_score
    }

    return best_model, best_hyperparameters, best_metrics


def tune_regression_model_hyperparameters(model_class, X_train, y_train, param_grid):

    #scorer = make_scorer(mean_squared_error,greater_is_better=False)

    #param_grid["random_state"] = [1]
    grid_search = GridSearchCV(estimator=model_class , param_grid=param_grid, scoring= 'neg_root_mean_squared_error')
    
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


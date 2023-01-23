from sklearn.model_selection import GridSearchCV
from itertools import product


def custom_tune_classification_model_hyperparameters(model_type, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters):

    range_list = []

    best_validation_score = 0


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

        validation_score = model.score(X_validation, y_validation)

        if validation_score > best_validation_score:

            best_validation_score = validation_score
            best_model = model

                    
        model = None


    train_score = best_model.score(X_train, y_train)
    validation_score = best_validation_score
    test_score = best_model.score(X_test, y_test)

    best_hyperparameters = arg_dict
    best_metrics = {
        "train_score" : train_score,
        "validation_score" : validation_score,
        "test_score" : test_score,
    }

    return best_model, best_hyperparameters, best_metrics


def tune_classification_model_hyperparameters(model_class, X_train, X_test, y_train, y_test, param_grid):

    #scorer = make_scorer(mean_squared_error,greater_is_better=False)

    #param_grid["random_state"] = [1]
    grid_search = GridSearchCV(estimator=model_class , param_grid=param_grid, scoring= 'accuracy')
    
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    train_score = best_model.score(X_train, y_train)
    validation_score = grid_search.best_score_
    test_score = best_model.score(X_test, y_test)

    best_metrics = {
        "train_accuracy" : train_score,
        "validation_accuracy" : validation_score,
        "test_accuracy" : test_score,
    }

    return best_model, grid_search.best_params_, best_metrics


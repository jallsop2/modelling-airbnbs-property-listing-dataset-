import joblib
import json

from sklearn.model_selection import train_test_split


def save_model(model, params, metrics, path):
    joblib.dump(model, path+"/model.joblib")

    json_params = json.dumps(params, indent=4)
    with open(f"{path}/hyperparameters.json","w") as file:
        file.write(json_params)

    json_metrics = json.dumps(metrics, indent=4)
    with open(f"{path}/metrics.json","w") as file:
        file.write(json_metrics)







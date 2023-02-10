import joblib
import json

from torch import save

from sklearn.model_selection import train_test_split


def save_model(model, params, metrics, path, pytorch=False):

    if pytorch == False:
        joblib.dump(model, f"{path}/model.joblib")

    elif pytorch == True:
        save(model.state_dict(), f"{path}/model.pt")

    json_params = json.dumps(params, indent=4)
    with open(f"{path}/hyperparameters.json","w") as file:
        file.write(json_params)

    json_metrics = json.dumps(metrics, indent=4)
    with open(f"{path}/metrics.json","w") as file:
        file.write(json_metrics)







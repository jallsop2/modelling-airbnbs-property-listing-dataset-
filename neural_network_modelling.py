import torch
import numpy as np
import yaml
import os
import datetime
from time import sleep
from time import perf_counter as timer

from ast import literal_eval

from tabular_data import load_airbnb
from save_models import save_model

from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as f

from torchmetrics.functional import r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class TorchStandardScaler:

    def fit(self, tensor):
        self.mean = tensor.mean(0, keepdim=True)
        self.std = tensor.std(0, unbiased=False, keepdim=True)

    def transform(self, tensor):
        tensor -= self.mean
        tensor /= (self.std + 1e-7)
        return tensor

    def inverse_transform_predictions(self, predictions):
        predictions *= self.std[:,-1]
        predictions += self.mean[:,-1]


class AirbnbNightlyPriceImageDataset(Dataset):

    def __init__(self, indices = None):
        super().__init__()
        numerical_data, labels = load_airbnb()

        numerical_data = np.concatenate((numerical_data[:,:3], numerical_data[:,4:], numerical_data[:,3].reshape([-1,1])), 1)

        scaler = StandardScaler()
        scaler.fit(numerical_data)
        data = scaler.transform(numerical_data)

        if indices != None:
            data = data[indices,:]
            
        self.data_tensor = torch.from_numpy(data).to(torch.float32)
       

    def __getitem__(self, index):
        #print(index)
        #return (self.data_tensor[index, :-1], self.data_tensor[index,-1].view(-1,1))
        return (self.data_tensor[index, :-1], self.data_tensor[index,-1])

    def __len__(self):
        return self.data_tensor.size(0)   

def create_train_test_datasets(Dataset, train_split = 0, validation_split = 0):

    if train_split == 0:    
        return Dataset()
    
    else:
        num_data_points = load_airbnb()[0].shape[0]

        indices = range(num_data_points)

        train_indices, test_indices = train_test_split(indices, train_size= train_split)

        if validation_split == 0:
            return (Dataset(train_indices), Dataset(test_indices))

        elif validation_split != 0:
            test_indices, validation_indices = train_test_split(test_indices, test_size=validation_split)
            return (Dataset(train_indices), Dataset(validation_indices), Dataset(test_indices))


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(11,1)

    def forward(self, features):
        return self.linear_layer(features)


class NN(torch.nn.Module):
    def __init__(self, width=16, depth=2):
        super().__init__()

        self.width = width
        self.depth = depth

        setup_list = [torch.nn.Linear(11, width), torch.nn.ReLU()]

        for i in range(depth-2):
            setup_list.append(torch.nn.Linear(width, width))
            setup_list.append(torch.nn.ReLU())

        setup_list.append(torch.nn.Linear(width, 1))

        self.layers = torch.nn.Sequential(*setup_list)
        
    def forward(self, features):
        return self.layers(features)


def get_nn_config():
    with open("nn_config.yaml", "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


def train(model, data_loader, optimiser_class,  optimiser_hyperparameters, epochs=10, track_performance=False, tensorboard=False):

    optimiser = optimiser_class(model.parameters(), **optimiser_hyperparameters)

    if tensorboard == True:
        writer = SummaryWriter()

    batch_idx = 0

    train_tic = timer()
    avg_pred_time = 0

    for epoch in range(epochs):
        for batch in data_loader:
            
            features, labels = batch

            pred_tic = timer()            
            predictions = model(features)
            avg_pred_time += timer() - pred_tic

            labels = labels.view(-1,1)

            loss = f.mse_loss(predictions, labels)
            loss.backward()

            optimiser.step()
            optimiser.zero_grad()

            if tensorboard == True:
                writer.add_scalar("loss", loss.item(), batch_idx)
            
            batch_idx += 1
    
    if track_performance == True:

        training_time = timer() - train_tic

        avg_pred_time /= batch_idx

        return (training_time, avg_pred_time)



def test(model, list_of_data_sets, print_metrics = False):

    num_data_sets = len(list_of_data_sets)

    RMSE_list = [0]*num_data_sets
    r2_list = [0]*num_data_sets

    for i in range(num_data_sets):

        X = list_of_data_sets[i].data_tensor[:,:-1]
        y = list_of_data_sets[i].data_tensor[:,-1].view(-1,1)

        predictions = model(X)
        RMSE_list[i] = torch.sqrt(f.mse_loss(predictions, y))
        r2_list[i] = r2_score(predictions, y)

        if print_metrics == True:
            print(f"\nDataset {i}")
            print(f"RMSE = {RMSE_list[i]}")
            print(f"r2 score = {r2_list[i]}")
        
    return (RMSE_list, r2_list)


def save_configs_to_file(hyperparameter_list, metrics_list):

    num_configs = len(hyperparameter_list)

    for config_num in range(num_configs):

            hyperparameters = hyperparameter_list[config_num]

            full_dataset = AirbnbNightlyPriceImageDataset()
            data_loader = DataLoader(full_dataset, batch_size, shuffle=True)

            optimiser_str = hyperparameters["optimiser"]
            optimiser_class = eval(optimiser_str)
            optimiser_hyperparameters = hyperparameters["optimiser_hyperparameters"]
            epochs = hyperparameters["epochs"]
            


            model = NN(hyperparameters["hidden_layer_width"], hyperparameters["depth"])

            train(model, data_loader, optimiser_class, optimiser_hyperparameters, epochs, track_performance=False, tensorboard=False)

            metrics = metrics_list[config_num]


            sleep(1)
            datetime_str = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
            path = f"models/regression/neural_networks/{datetime_str}"
            os.makedirs(path)
            save_model(model, hyperparameters, metrics, path, pytorch=True)


def get_average_model_performance(hyperparameter_list, Dataset, num_tests=10, batch_size=50, train_split_size=0.7, print_metrics=True, save_to_file=False, tensorboard=False ):

    num_configs = len(hyperparameter_list)

    train_RMSE = np.zeros([num_configs])
    test_RMSE = np.zeros([num_configs])

    train_r2_score = np.zeros([num_configs])
    test_r2_score = np.zeros([num_configs])

    training_time = np.zeros([num_configs])
    pred_time = np.zeros([num_configs])

    optimiser_whitelist = ["torch.optim.SGD", "torch.optim.Adam"]

    for test_num in range(num_tests):

        print(f"Test: {test_num}", end = "\r")

        train_data, test_data = create_train_test_datasets(Dataset, train_split=train_split_size)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

        config_num = 0
        for hyperparameters in hyperparameter_list:

            optimiser_str = hyperparameters["optimiser"]
            optimiser_hyperparameters = hyperparameters["optimiser_hyperparameters"]
            epochs = hyperparameters["epochs"]

            if optimiser_str in optimiser_whitelist:
                optimiser_class = eval(optimiser_str)
            else:
                print("That optimiser is not allowed.")
                exit()

            model = NN(hyperparameters["hidden_layer_width"], hyperparameters["depth"])

            current_training_time, current_pred_time = train(model, train_loader, optimiser_class, optimiser_hyperparameters, epochs, track_performance=True, tensorboard=tensorboard)
            
            current_RMSE, current_r2_score = test(model, [train_data, test_data])

            train_RMSE[config_num] += current_RMSE[0]/num_tests
            test_RMSE[config_num] += current_RMSE[1]/num_tests

            train_r2_score[config_num] += current_r2_score[0]/num_tests
            test_r2_score[config_num] += current_r2_score[1]/num_tests

            training_time[config_num] += current_training_time/num_tests
            pred_time[config_num] += current_pred_time/num_tests

            config_num += 1
    
    print("Testing complete. \n")

    metrics_list = []
    for config_num in range(num_configs):

        metrics_list.append({
            "RMSE" : {
                "Training data" : train_RMSE[config_num].item(),
                "Test data" : test_RMSE[config_num].item()
            },
            "r2_score" : {
                "Training data" : train_r2_score[config_num].item(),
                "Test data" : test_r2_score[config_num].item()
            },
            "average_training_duration" : training_time[config_num],
            "average_interference_latency" : pred_time[config_num]
        })

        if print_metrics == True:
            print(metrics_list[config_num]["RMSE"])

    
    if save_to_file == True:
        save_configs_to_file(hyperparameter_list, metrics_list)
   

if __name__ == "__main__":

    dataset = AirbnbNightlyPriceImageDataset
    num_tests = 1000
    batch_size = 100
    train_split_size = 0.7

    #hyperparameters = get_nn_config()

    hyperparameter_list = []

    hyperparameter_list.append({
        "optimiser": "torch.optim.SGD",
        "optimiser_hyperparameters": { 
            "lr": 0.1,
            "weight_decay": 0.05},
        "epochs": 50,
        "hidden_layer_width": 8,
        "depth": 2
    })

    hyperparameter_list.append({
        "optimiser": "torch.optim.Adam",
        "optimiser_hyperparameters": { 
            "lr": 0.01,
            "weight_decay": 0.05},
        "epochs": 50,
        "hidden_layer_width": 8,
        "depth": 2
    })

    hyperparameter_list.append({
        "optimiser": "torch.optim.SGD",
        "optimiser_hyperparameters": { 
            "lr": 0.05,
            "weight_decay": 0.05},
        "epochs": 50,
        "hidden_layer_width": 32,
        "depth": 4
    })

    hyperparameter_list.append({
        "optimiser": "torch.optim.Adam",
        "optimiser_hyperparameters": { 
            "lr": 0.005,
            "weight_decay": 0.05},
        "epochs": 50,
        "hidden_layer_width": 32,
        "depth": 4
    })





    get_average_model_performance(hyperparameter_list, dataset, num_tests, batch_size, train_split_size, print_metrics=True, save_to_file=True, tensorboard=False)

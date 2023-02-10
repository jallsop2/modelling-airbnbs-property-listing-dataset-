import torch
import numpy as np
import yaml
import os
import datetime
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


def train(model, data_loader, epochs=10, learning_rate=0.001, optimiser_class=torch.optim.SGD, track_performance=False, tensorboard=False):

    optimiser = optimiser_class(model.parameters(), lr = learning_rate)

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



def get_average_model_performance(hyperparams, Dataset, num_tests=10, batch_size=50, train_split_size=0.7):

    metrics = []

    train_RMSE = 0
    test_RMSE = 0

    train_r2_score = 0
    test_r2_score = 0

    training_time = 0
    pred_time = 0

    optimiser_whitelist = ["torch.optim.SGD"]

    learning_rate = hyperparams["learning_rate"]

    optimiser_str = hyperparams["optimiser"]

    epochs = hyperparams["epochs"]
    
    if optimiser_str in optimiser_whitelist:
        optimiser_class = eval(optimiser_str)
    else:
        print("That optimiser is not allowed.")
        exit()

    for test_num in range(num_tests):

        print(f"Test: {test_num}", end = "\r")

        train_data, test_data = create_train_test_datasets(Dataset, train_split=train_split_size)

        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

        model = NN(hyperparams["hidden_layer_width"], hyperparams["depth"])

        current_training_time, current_pred_time = train(model, train_loader, epochs, learning_rate, optimiser_class, track_performance=True)

        current_RMSE, current_r2_score = test(model, [train_data, test_data])

        train_RMSE += current_RMSE[0]/num_tests
        test_RMSE += current_RMSE[1]/num_tests

        train_r2_score += current_r2_score[0]/num_tests
        test_r2_score += current_r2_score[1]/num_tests

        training_time += current_training_time/num_tests
        pred_time += current_pred_time/num_tests


    metrics = {
        "RMSE" : {
            "Training data" : train_RMSE.item(),
            "Test data" : test_RMSE.item()
        },
        "r2_score" : {
            "Training data" : train_r2_score.item(),
            "Test data" : test_r2_score.item()
        },
        "average_training_duration" : training_time,
        "average_interference_latency" : pred_time
    }

    return (model, metrics)



def test_average_performance(print_metrics=True, save_to_file=False):

    hyperparameters = get_nn_config()
    dataset = AirbnbNightlyPriceImageDataset
    num_tests = 100
    batch_size = 50
    train_split_size = 0.7

    model, metrics = get_average_model_performance(hyperparameters, dataset, num_tests, batch_size, train_split_size)

    if print_metrics == True:
        print(metrics)
    
    if save_to_file == True:
        datetime_str = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        path = f"models/regression/neural_networks/{datetime_str}"
        os.makedirs(path)

        save_model(model, hyperparameters, metrics, path, pytorch=True)

if __name__ == "__main__":

    test_average_performance(save_to_file = True)


    """ train_data, test_data = create_train_test_datasets(AirbnbNightlyPriceImageDataset, train_split=0.7)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=50)

    #model = LinearRegression()
    model = NN()

    train(model, train_loader, True, 10, 0.01)

    test(model, train_data, test_data, print_metrics=True) """

            

            





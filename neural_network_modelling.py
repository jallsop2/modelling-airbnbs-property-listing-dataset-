import torch
import numpy as np

from tabular_data import load_airbnb

from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as f

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


def train(model, data_loader, track_performance=False, epochs=10, learning_rate = 0.001):

    optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate)

    if track_performance == True:
        writer = SummaryWriter()


    batch_idx = 0

    for epoch in range(epochs):
        for batch in data_loader:
            
            features, labels = batch
            predictions = model(features)

            labels = labels.view(-1,1)

            loss = f.mse_loss(predictions, labels)
            loss.backward()

            optimiser.step()
            optimiser.zero_grad()

            if track_performance == True:
                writer.add_scalar("loss", loss.item(), batch_idx)
                batch_idx += 1

def test(model, train_data, test_data=None, validation_data=None, print_metrics = False):

    train_RMSE = None
    validation_RMSE = None
    test_RMSE = None


    X_train = train_data.data_tensor[:,:-1]
    y_train = train_data.data_tensor[:,-1].view(-1,1)

    train_predictions = model(X_train)
    train_RMSE = torch.sqrt(f.mse_loss(train_predictions, y_train))

    if print_metrics == True:
        print(f"Training loss = {train_RMSE}")


    if validation_data != None:
        X_validation = validation_data.data_tensor[:,:-1]
        y_validation = validation_data.data_tensor[:,-1].view(-1,1)

        validation_predictions = model(X_validation)
        validation_RMSE = torch.sqrt(f.mse_loss(validation_predictions, y_validation))

        if print_metrics == True:
            print(f"Validation loss = {validation_RMSE}")

    if test_data != None:
        X_test = test_data.data_tensor[:,:-1]
        y_test = test_data.data_tensor[:,-1].view(-1,1)

        test_predictions = model(X_test)
        test_RMSE = torch.sqrt(f.mse_loss(test_predictions, y_test))


        if print_metrics == True:
            print(f"Test loss = {test_RMSE}")

    return [train_RMSE, validation_RMSE, test_RMSE]


def get_average_model_performance(model_class, hyperparams, Dataset, num_tests=10, batch_size=50, epochs=100, train_split_size=0.7):


    train_RMSE = 0
    test_RMSE = 2

    for test_num in range(num_tests):

        print(f"Test: {test_num}", end = "\r")

        train_data, test_data = create_train_test_datasets(Dataset, train_split=train_split_size)

        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

        model = model_class()

        train(model, train_loader, **hyperparams)

        metrics = test(model, train_data, test_data)

        train_RMSE += metrics[0]
        test_RMSE += metrics[2]

    train_RMSE = train_RMSE/num_tests
    test_RMSE = test_RMSE/num_tests

    return (train_RMSE, test_RMSE)



def test_average_performance():
    model_class = LinearRegression
    hyperparams = {
        "epochs" : 10,
        "learning_rate" : 0.01
        }
    dataset = AirbnbNightlyPriceImageDataset
    num_tests = 100
    batch_size = 50
    train_split_size = 0.7

    train_RMSE, test_RMSE = get_average_model_performance(model_class, hyperparams, dataset, num_tests, batch_size, train_split_size)

    print(f"Average training RMSE = {train_RMSE}")
    print(f"Average test RMSE = {test_RMSE}")


if __name__ == "__main__":

    #test_average_performance()

    train_data, test_data = create_train_test_datasets(AirbnbNightlyPriceImageDataset, train_split=0.7)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=50)

    model = LinearRegression()

    train(model, train_loader, True, 100, 0.01)

    test(model, train_data, test_data, print_metrics=True)

            

            





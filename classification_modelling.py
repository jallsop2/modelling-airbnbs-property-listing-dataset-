from tabular_data import load_airbnb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

if __name__ == "__main__":
    data, labels = load_airbnb(category=True)

    #print(data)

    y = data[:,-1]
    X = data[:,:-1]

    scaler = StandardScaler()

    scaler.fit(X)
    X = scaler.transform(X)

    print(X.shape,len(y))

    
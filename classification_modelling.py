from tabular_data import load_airbnb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression

import numpy as np

if __name__ == "__main__":
    data, labels = load_airbnb(category=True)

    #print(data)

    y = data[:,-1]
    X = data[:,:-1]

    scaler = StandardScaler()

    scaler.fit(X)
    X = scaler.transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LogisticRegression()

    model.fit(X_train,y_train)

    print(model.score(X_test,y_test))

    test_predictions = model.predict(X_test)




    
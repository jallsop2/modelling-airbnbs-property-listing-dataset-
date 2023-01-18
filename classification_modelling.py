from tabular_data import load_airbnb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt

def visualise_confusion_matrix(cf, normalise=False):
    if normalise == True:
        cf = cf / cf.sum()

    display = ConfusionMatrixDisplay(cf)
    display.plot()
    plt.show()

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

    print(model.score(X_train, y_train))

    print(model.score(X_test,y_test))

    test_predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)

    #print(precision_score(y_test, test_predictions, average="micro"))
    #print(recall_score(y_test, test_predictions, average="micro"))

    cf_train = confusion_matrix(y_train, train_predictions)
    cf_test = confusion_matrix(y_test, test_predictions)

    print("\n\n")
    print(model.classes_)

    visualise_confusion_matrix(cf_train)
    visualise_confusion_matrix(cf_test)
    




    
from tabular_data import load_airbnb
import numpy as np
import matplotlib.pyplot as plt

data, labels = load_airbnb()

y = data[:,3]
X = np.concatenate((data[:,:3], data[:,4:]),1)


print(X[:,0].shape, y.shape)

plt.scatter(X[:,1],y)
plt.show()
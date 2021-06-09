import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense      #every node is connected to the preeceeding
from keras.optimizers import Adam   #one of many optimaztion algorithms, it's adabtive algorithm
#it's from the stochastic gradient descent family which uses 1 sample to perfrom
#gradient descent.. it's also from Adagrad and RMSprop family
#it also calculates an adaptive learning rate, alpha, for each parameter

n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T

# print("Xa : ")
# print(Xa)
# print("Xb: ")
# print(Xb)

X = np.vstack((Xa, Xb))
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

# print("X :")
# print(X)






model = Sequential()
model.add(Dense(units = 1, input_shape = (2,), activation = 'sigmoid'))   #adds layers/ units = 1 because we have only one ouput node producing one output
#input_shape = 2 because we have two inputs
adam = Adam(lr = 0.1)
model.compile(adam, loss='binary_crossentropy', metrics = ['accuracy'])
h = model.fit(x=X, y=y, verbose = 1, batch_size = 50, epochs = 100, shuffle = 'true') #epoch wwill go through our data 500 times

# plt.scatter(X[:n_pts,0], X[:n_pts,1], color = 'r')
# plt.scatter(X[n_pts:,0], X[n_pts:,1], color = 'b')

plt.plot(h.history['acc'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.show()

plt.plot(h.history['loss'])
plt.legend(['loss'])
plt.title('loss')
plt.xlabel('epoch')

plt.show()

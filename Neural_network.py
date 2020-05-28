import numpy as np
import matplotlib.pyplot as plots
class Neural_Network():
    def forward(X, W1, b1, W2, b2):
        Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))
        activation = Z.dot(W2) + b2
        Y = 1 / (1 + np.exp(-activation))
        return Y,Z

    def predict(X, W1, b1, W2, b2):
        Y, _ = Neural_Network.forward(X, W1, b1, W2, b2)
        return np.round(Y)

    def _w2(Z, T, Y):
        return Z.T.dot(T - Y)

    def _b2(T, Y):
        return (T - Y).sum(axis = 0)

    def _w1(X, Z, T, Y, W2):
        dZ = (T-Y).dot(W2.T) * Z * (1 - Z) # this is for sigmoid activation
        return X.T.dot(dZ)

    def _b1(Z, T, Y, W2):
        dZ = (T-Y).dot(W2.T) * Z * (1 - Z) # this is for sigmoid activation
        return dZ.sum(axis=0)

    def cost(T, Y):
        return np.sum(T*np.log(Y))

def fit():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 1]).reshape(4,1)
    W1 = np.random.randn(2, 8)
    b1 = np.zeros(8)
    W2 = np.random.randn(8,2)
    b2 = np.zeros(2)
    plot = []
    learning_rate = 0.0001
    for i in range(100000):
        pY, Z = Neural_Network.forward(X, W1, b1, W2, b2)
        plt = Neural_Network.cost(Y, pY)
        prediction = Neural_Network.predict(X, W1, b1, W2, b2)

        plot.append(plt)
        W2 += learning_rate * (Neural_Network._w2(Z, Y, pY))
        b2 += learning_rate * (Neural_Network._b2(Y, pY)) 
        W1 += learning_rate * (Neural_Network._w1(X, Z, Y, pY, W2))
        b1 += learning_rate * (Neural_Network._b1(Z, Y, pY, W2)) 
        if i % 10000 == 0:
            print(plt)
            print("actual",Y)
            print("predicted",prediction)

    print("Accuracy:", np.mean(prediction == Y) * 100)
    plots.plot(plot)
    plots.show()

if __name__ == '__main__':
     fit()

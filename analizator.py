import numpy as np

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Tworzenie historii (okien czasowych) - łączenie chwil t, t-1, t-2 w jeden wektor
def create_time_windows(X_data, y_data, window_size):
    X_windows, y_windows = [], []
    for i in range(len(X_data) - window_size):
        window = X_data[i:(i+window_size)].flatten()
        X_windows.append(window)
        y_windows.append(y_data[i+window_size])
    return np.array(X_windows), np.array(y_windows)

#Skalowanie danych (odjęcie średniej i podział przez odchylenie standardowe)
# Dzięki temu sieć uczy się stabilnie i nie "wybucha" matematycznie
def scale_data(X_train, X_test):
    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    return (X_train-mean) / std, (X_test - mean)/std
class SiecNeuronowa:
    def __init__(self, input_dim):
        # W np.random.randn(input_dim, 32) 32 to liczba neuronów na warstwę a input_dim to liczba cech wejściowych
        self.W1 = np.random.randn(input_dim, 32)*np.sqrt(2.0/input_dim)
        self.b1 = np.zeros((1, 32))

        self.W2 = np.random.randn(32,16) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros((1,16))

        self.W3 = np.random.randn(16,2) * np.sqrt(2.0 / 16)
        self.b3 = np.zeros((1,2))

    def ruch(self,X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = np.dot(self.A1,self.W2) + self.b2
        self.A2 = relu(self.Z2)

        self.Z3 = np.dot(self.A2,self.W3) + self.b3
        self.A3 = self.Z3
        return self.A3

    def backward(self, X, y, y_pred, lr):
        #Wsteczna propagacja błędu (Backpropagation) i nauka
        m = X.shape[0]

        # 1. błędy warstwy wyjściowej
        dZ3 = (y_pred - y) / m
        dW3 = np.dot(self.A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        # 2. Cofanie błędu do warstwy 2
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        # 3. Cofanie błędu do warstwy 1
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        # 4. Aktualizacja wag
        self.W3 -= lr * dW3;  self.b3 -= lr * db3
        self.W2 -= lr * dW2;  self.b2 -= lr * db2
        self.W1 -= lr * dW1;  self.b1 -= lr * db1

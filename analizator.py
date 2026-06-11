import numpy as np
import pandas as pd

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def calculate_errors(y_true, y_pred):
    diff = y_true - y_pred
    return np.sqrt(np.sum(diff**2, axis=1))

def create_time_windows(X_data, y_data, window_size):
    X_windows, y_windows = [], []
    for i in range(len(X_data) - window_size):
        window = X_data[i:(i+window_size)].flatten()
        X_windows.append(window)
        y_windows.append(y_data[i+window_size])
    return np.array(X_windows), np.array(y_windows)

def scale_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    std[std < 0.1] = 1.0 
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    X_train_scaled = np.clip(X_train_scaled, -5.0, 5.0)
    X_test_scaled = np.clip(X_test_scaled, -5.0, 5.0)
    
    return X_train_scaled, X_test_scaled

def usun_grube_bledy(df, kolumny):
    df_clean = df.copy()
    for col in kolumny:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        skok = df_clean[col].diff().abs()
        
        prog = max(skok.mean() + 4 * skok.std(), 20.0)
        maska_bledow = skok > prog
        df_clean.loc[maska_bledow, col] = np.nan
        
    df_clean[kolumny] = df_clean[kolumny].interpolate(method='linear').ffill().bfill()
    return df_clean

class SiecNeuronowa:
    def __init__(self, input_dim):
        self.W1 = np.random.randn(input_dim, 32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, 32))
        self.W2 = np.random.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros((1, 16))
        self.W3 = np.random.randn(16, 2) * np.sqrt(2.0 / 16)
        self.b3 = np.zeros((1, 2))

    def ruch(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.Z3
        return self.A3

    def backward(self, X, y_true, y_pred, lr):
        m = X.shape[0] 
        dZ3 = (y_pred - y_true) / m
        dW3 = np.dot(self.A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
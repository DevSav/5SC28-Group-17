import copy

import numpy as np
import torch
from torch import nn


def make_lstm_data(u, y, history=15):
    """Make sequence data for the LSTM.

    Each sample is:
    [[u[k-15], y[k-15]], ..., [u[k-1], y[k-1]]] -> y[k]
    """
    u = np.asarray(u).reshape(-1)
    y = np.asarray(y).reshape(-1)

    X = []
    Y = []

    for k in range(history, len(y)):
        one_sequence = []

        for j in range(k - history, k):
            one_sequence.append([u[j], y[j]])

        X.append(one_sequence)
        Y.append(y[k])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32).reshape(-1, 1)
    return X, Y


class LSTMNetwork(nn.Module):
    """Small LSTM model, based on the Lecture 3 recurrent examples."""

    def __init__(self, hidden_size=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.lay_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hiddens, _ = self.lstm(x)
        last_hidden = hiddens[:, -1, :]
        y = self.lay_out(last_hidden)
        return y


class LSTMANNModel:
    """Advanced ANN model for one-step prediction and simulation."""

    def __init__(self, hidden_size=20, learning_rate=0.001):
        self.model = LSTMNetwork(hidden_size=hidden_size)
        self.learning_rate = learning_rate
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def _scale_x(self, x):
        return (x - self.x_mean) / self.x_std

    def _scale_y(self, y):
        return (y - self.y_mean) / self.y_std

    def _unscale_y(self, y_scaled):
        return y_scaled * self.y_std + self.y_mean

    def fit(self, Xtrain, Ytrain, Xval, Yval, epochs=80, batch_size=256, patience=10):
        Xtrain = np.asarray(Xtrain, dtype=np.float32)
        Ytrain = np.asarray(Ytrain, dtype=np.float32)
        Xval = np.asarray(Xval, dtype=np.float32)
        Yval = np.asarray(Yval, dtype=np.float32)

        self.x_mean = Xtrain.mean(axis=(0, 1), keepdims=True)
        self.x_std = Xtrain.std(axis=(0, 1), keepdims=True) + 1e-8
        self.y_mean = Ytrain.mean(axis=0, keepdims=True)
        self.y_std = Ytrain.std(axis=0, keepdims=True) + 1e-8

        Xtrain = torch.as_tensor(self._scale_x(Xtrain))
        Ytrain = torch.as_tensor(self._scale_y(Ytrain))
        Xval = torch.as_tensor(self._scale_x(Xval))
        Yval = torch.as_tensor(self._scale_y(Yval))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        best_state = None
        no_improvement = 0
        ids = np.arange(len(Xtrain))

        for epoch in range(epochs):
            np.random.shuffle(ids)

            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start : start + batch_size]
                Xbatch = Xtrain[batch_ids]
                Ybatch = Ytrain[batch_ids]

                Ypred = self.model(Xbatch)
                Loss = torch.mean((Ypred - Ybatch) ** 2)

                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

            with torch.no_grad():
                Yval_pred = self.model(Xval)
                val_loss = torch.mean((Yval_pred - Yval) ** 2).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1

            print("epoch", epoch, "validation loss", round(val_loss, 5))

            if no_improvement >= patience:
                print("Stopped early at epoch", epoch)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = torch.as_tensor(self._scale_x(X))

        self.model.eval()
        with torch.no_grad():
            prediction_scaled = self.model(X).numpy()

        return self._unscale_y(prediction_scaled).reshape(-1)

    def simulate(self, u, y_start, history=15):
        u = np.asarray(u).reshape(-1)
        y_start = list(np.asarray(y_start).reshape(-1))

        Ysim = y_start.copy()
        start_index = len(y_start)

        for k in range(start_index, len(u)):
            one_sequence = []

            for j in range(k - history, k):
                one_sequence.append([u[j], Ysim[j]])

            next_y = self.predict(np.array([one_sequence], dtype=np.float32))[0]
            Ysim.append(next_y)

        return np.array(Ysim)

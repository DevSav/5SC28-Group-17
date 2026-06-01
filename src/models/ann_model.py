import copy

import numpy as np
import torch
from torch import nn


def make_narx_data(u, y, input_delay=3, output_delay=3):
    """Make the NARX input-output data like in the practical session."""
    u = np.asarray(u).reshape(-1)
    y = np.asarray(y).reshape(-1)

    first_index = max(input_delay, output_delay)
    X = []
    Y = []

    for k in range(first_index, len(y)):
        # One row is [old u values, old y values] and the target is y[k].
        row = []

        for delay in range(input_delay, 0, -1):
            row.append(u[k - delay])

        for delay in range(output_delay, 0, -1):
            row.append(y[k - delay])

        X.append(row)
        Y.append(y[k])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32).reshape(-1, 1)
    return X, Y


class Network(nn.Module):
    """Small feedforward ANN, based on the Lecture 2 practical."""

    def __init__(self, n_inputs, n_hidden_nodes=20):
        super().__init__()
        self.lay1 = nn.Linear(n_inputs, n_hidden_nodes)
        self.lay2 = nn.Linear(n_hidden_nodes, 1)

    def forward(self, x):
        x1 = torch.sigmoid(self.lay1(x))
        y = self.lay2(x1)
        return y


class ANNModel:
    """Small wrapper around the ANN so prediction and simulation are easy."""

    def __init__(self, number_of_inputs, hidden_neurons=20, learning_rate=0.001, l2_weight=0.0001):
        self.model = Network(number_of_inputs, hidden_neurons)
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
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

    def fit(self, Xtrain, Ytrain, Xval, Yval, epochs=1000, patience=40):
        """Train the ANN. This follows the practical notebook training loop."""
        Xtrain = np.asarray(Xtrain, dtype=np.float32)
        Ytrain = np.asarray(Ytrain, dtype=np.float32)
        Xval = np.asarray(Xval, dtype=np.float32)
        Yval = np.asarray(Yval, dtype=np.float32)

        # Normalize using only the training data.
        self.x_mean = Xtrain.mean(axis=0, keepdims=True)
        self.x_std = Xtrain.std(axis=0, keepdims=True) + 1e-8
        self.y_mean = Ytrain.mean(axis=0, keepdims=True)
        self.y_std = Ytrain.std(axis=0, keepdims=True) + 1e-8

        Xtrain = torch.as_tensor(self._scale_x(Xtrain))
        Ytrain = torch.as_tensor(self._scale_y(Ytrain))
        Xval = torch.as_tensor(self._scale_x(Xval))
        Yval = torch.as_tensor(self._scale_y(Yval))

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_weight,
        )

        best_val_loss = float("inf")
        best_state = None
        no_improvement = 0

        for epoch in range(epochs):
            self.model.train()
            Ypred = self.model(Xtrain)
            Loss = torch.mean((Ypred - Ytrain) ** 2)

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

            if epoch % 100 == 0:
                print("epoch", epoch, "train loss", round(Loss.item(), 5), "validation loss", round(val_loss, 5))

            if no_improvement >= patience:
                print("Stopped early at epoch", epoch)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, inputs):
        """Predict outputs for prepared NARX inputs."""
        inputs = np.asarray(inputs, dtype=np.float32)
        x = torch.tensor(self._scale_x(inputs))

        self.model.eval()
        with torch.no_grad():
            prediction_scaled = self.model(x).numpy()

        return self._unscale_y(prediction_scaled).reshape(-1)

    def simulate(self, u, y_start, input_delay=3, output_delay=3):
        """Simulate the model by feeding back its own previous outputs."""
        u = np.asarray(u).reshape(-1)
        y_start = list(np.asarray(y_start).reshape(-1))

        Ysim = y_start.copy()
        start_index = len(y_start)

        for k in range(start_index, len(u)):
            row = []

            for delay in range(input_delay, 0, -1):
                row.append(u[k - delay])

            for delay in range(output_delay, 0, -1):
                row.append(Ysim[k - delay])

            next_y = self.predict(np.array([row], dtype=np.float32))[0]
            Ysim.append(next_y)

        return np.array(Ysim)

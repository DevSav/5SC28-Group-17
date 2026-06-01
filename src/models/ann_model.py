import copy

import numpy as np
import torch
from torch import nn


def make_narx_data(u, y, input_delay=3, output_delay=3):
    """Make delayed inputs for a simple NARX model.

    We predict y[k] from old inputs and old measured outputs:
    [u[k-3], u[k-2], u[k-1], y[k-3], y[k-2], y[k-1]] -> y[k]
    """
    u = np.asarray(u).reshape(-1)
    y = np.asarray(y).reshape(-1)

    first_index = max(input_delay, output_delay)
    inputs = []
    targets = []

    for k in range(first_index, len(y)):
        row = []

        for delay in range(input_delay, 0, -1):
            row.append(u[k - delay])

        for delay in range(output_delay, 0, -1):
            row.append(y[k - delay])

        inputs.append(row)
        targets.append(y[k])

    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32).reshape(-1, 1)


class SmallNeuralNetwork(nn.Module):
    """A small feedforward neural network."""

    def __init__(self, number_of_inputs, hidden_neurons=20):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(number_of_inputs, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, 1),
        )

    def forward(self, x):
        return self.layers(x)


class ANNModel:
    """Simple ANN model for the unbalanced disc angle."""

    def __init__(self, number_of_inputs, hidden_neurons=20, learning_rate=0.001, l2_weight=0.0001):
        self.model = SmallNeuralNetwork(number_of_inputs, hidden_neurons)
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

    def fit(self, train_inputs, train_outputs, validation_inputs, validation_outputs, epochs=1000, patience=40):
        """Train the ANN with early stopping."""
        train_inputs = np.asarray(train_inputs, dtype=np.float32)
        train_outputs = np.asarray(train_outputs, dtype=np.float32)
        validation_inputs = np.asarray(validation_inputs, dtype=np.float32)
        validation_outputs = np.asarray(validation_outputs, dtype=np.float32)

        self.x_mean = train_inputs.mean(axis=0, keepdims=True)
        self.x_std = train_inputs.std(axis=0, keepdims=True) + 1e-8
        self.y_mean = train_outputs.mean(axis=0, keepdims=True)
        self.y_std = train_outputs.std(axis=0, keepdims=True) + 1e-8

        x_train = torch.tensor(self._scale_x(train_inputs))
        y_train = torch.tensor(self._scale_y(train_outputs))
        x_val = torch.tensor(self._scale_x(validation_inputs))
        y_val = torch.tensor(self._scale_y(validation_outputs))

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_weight,
        )
        loss_function = nn.MSELoss()

        best_validation_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.model.train()
            prediction = self.model(x_train)
            train_loss = loss_function(prediction, y_train)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                validation_prediction = self.model(x_val)
                validation_loss = loss_function(validation_prediction, y_val).item()

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % 100 == 0:
                print("epoch", epoch, "train loss", round(train_loss.item(), 5), "validation loss", round(validation_loss, 5))

            if epochs_without_improvement >= patience:
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

        y_simulated = y_start.copy()
        first_index = len(y_start)

        for k in range(first_index, len(u)):
            row = []

            for delay in range(input_delay, 0, -1):
                row.append(u[k - delay])

            for delay in range(output_delay, 0, -1):
                row.append(y_simulated[k - delay])

            next_y = self.predict(np.array([row], dtype=np.float32))[0]
            y_simulated.append(next_y)

        return np.array(y_simulated)

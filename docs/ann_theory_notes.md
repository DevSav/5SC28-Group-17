# ANN Theory Notes - Group 17

These notes explain the ANN part in simple words.

## What The ANN Is Used For

In this project the ANN is used as a model of the unbalanced disc.

The model tries to predict the next measured angle of the disc from old measurements.

The assignment says that, for the modelling task, we should use:

- input: motor voltage;
- output: measured disc angle;
- not the angular velocity.

## Model Structure

I started with a NARX model.

NARX means Nonlinear AutoRegressive with eXogenous input.

In simple words:

- exogenous input means old input voltages;
- autoregressive means old measured output angles;
- nonlinear means the relation is not just a straight line.

The model has this form:

```text
y[k] = f(u[k-1], u[k-2], u[k-3], y[k-1], y[k-2], y[k-3])
```

Here:

- `u` is the voltage;
- `y` is the measured angle;
- `f` is learned by the neural network.

This matches the lecture idea that a feedforward ANN can represent the static nonlinear function inside a dynamic model.

## Neural Network

The network is deliberately small:

```text
input layer -> tanh hidden layer -> output layer
```

I used `tanh` because it is a standard smooth nonlinear activation function and it was also used in the lecture examples.

The network is not very deep. This is a conscious choice:

- it is easier to explain;
- it trains faster;
- it is less likely to overfit;
- the rubric asks for clear design, not only complicated code.

## Training

The loss function is mean squared error:

```text
MSE = mean((measured angle - predicted angle)^2)
```

This follows the lecture idea that the cost function measures how well the model fits the data.

The optimizer is Adam. Adam is a practical gradient-based method. PyTorch calculates the gradients using backpropagation.

## Validation And Overfitting

The data is split into:

- training data: used to change the network weights;
- validation data: used to check if the model still works on data it did not train on.

I used early stopping. This means training stops when the validation loss stops improving.

This is based on the lecture theory about overfitting:

- if we train too long, the model may learn noise;
- then the training error is low, but validation error becomes worse.

I also used a small L2 regularization term. This discourages very large weights.

## Prediction And Simulation

The assignment asks us to test both prediction and simulation.

Prediction:

- the model uses past measured outputs;
- this is easier because the correct past angle is known.

Simulation:

- the model feeds back its own previous predicted outputs;
- this is harder because mistakes can build up over time.

This difference is important in system identification and is mentioned in the lectures.

## What To Improve Later

This first ANN is a simple baseline. Later we can try:

- different delays;
- more hidden neurons;
- a second hidden layer;
- a recurrent neural network;
- comparing NARX and NOE-style simulation.

For the report, we should not only show the best result. We should also explain what we tried and why.

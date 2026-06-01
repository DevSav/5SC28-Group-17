# Experiment Log - Group 17

Use this file to record what we tried.

## Template

Date:

Person:

Goal:

Method:

Settings:

Result:

What went well:

What did not work:

Next step:

## 2026-06-01 ANN NARX Baseline

Date: 2026-06-01

Person: Group 17 ANN part

Goal: Train a first simple ANN model for the unbalanced disc dynamics.

Method: NARX model with a small feedforward neural network. The model uses old voltages and old measured angles to predict the next angle.

Settings: 15 input delays, 15 output delays, 50 hidden neurons, 1500 epochs, early stopping available, L2 regularization.

Result: On a held-out test split, prediction RMSE was about 0.0204 rad, which is about 1.17 degrees. Simulation RMSE was about 0.852 rad, which is about 48.8 degrees.

What went well: The model learned one-step prediction much better than the first quick baseline.

What did not work: Long simulation is still poor. This makes sense because NARX uses measured past outputs during prediction, but in simulation it must use its own previous predictions. Small errors can build up.

Next step: Try a model that is more suitable for simulation, for example tuning delays/neurons more carefully or trying an NOE/recurrent-style model.

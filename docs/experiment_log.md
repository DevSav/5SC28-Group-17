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

## 2026-06-01 ANN NARX Practical-Style Version

Date: 2026-06-01

Person: Group 17 ANN part

Goal: Make the ANN code look closer to the Lecture 2 practical session code.

Method: Changed the network to a practical-style `Network` class with `lay1`, `lay2`, sigmoid activation, direct MSE loss, Adam optimizer, and the standard PyTorch loop: `zero_grad`, `backward`, `step`.

Settings: 15 input delays, 15 output delays, 50 hidden neurons, 1500 epochs, early stopping available, L2 regularization.

Result: On a held-out test split, prediction RMSE was about 0.0525 rad, which is about 3.01 degrees. Simulation RMSE was about 0.261 rad, which is about 15.0 degrees.

What went well: The code now matches the practical session style better, and the simulation result improved compared with the previous run.

What did not work: One-step prediction became worse than the previous tanh version.

Next step: Try both activation functions again later and report the trade-off. Also add an advanced ANN architecture, for example RNN/LSTM or NOE-style recurrent model.

## 2026-06-01 Advanced ANN LSTM

Date: 2026-06-01

Person: Group 17 ANN part

Goal: Add the advanced ANN architecture required by section 4.1.

Method: LSTM model. Each training sample is a short sequence of old voltage and angle values, and the target is the next angle.

Settings: 15 history samples, LSTM hidden size 20, 80 epochs, Adam optimizer.

Result: On a held-out test split, prediction RMSE was about 0.00715 rad, which is about 0.410 degrees. Simulation RMSE was about 0.0367 rad, which is about 2.10 degrees.

What went well: The LSTM performed much better than the simple NARX model on both prediction and simulation.

What did not work: This is still only tested on our own held-out split. The hidden-test true outputs are not available, so the checker only confirms file format.

Next step: Use the LSTM `.npz` outputs as the current ANN submission files unless a better model is trained later.

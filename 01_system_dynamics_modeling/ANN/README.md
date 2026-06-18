# ANN Model Work

This folder contains the ANN part of section 4.1.

## Files

```text
models/ann_model.py
  Simple feedforward NARX ANN.

models/lstm_ann_model.py
  Advanced recurrent LSTM ANN.

train_ann_assignment.py
  Trains only the simple NARX ANN.

train_lstm_ann_assignment.py
  Trains only the LSTM ANN.

compare_ann_models.py
  Trains both models and makes comparison plots.

tune_lstm_ann.py
  Tries a few LSTM settings and saves the tuning CSV.

plot_all_ann_results.py
  Makes the final summary plots from the CSV files.
```

## Models

We use two ANN models:

```text
Simple ANN:
  NARX feedforward neural network.
  Uses old voltages and old measured angles to predict the next angle.

Advanced ANN:
  LSTM recurrent neural network.
  Uses a short sequence of voltage and angle values to predict the next angle.
```

Both models use only:

```text
u  = applied motor voltage
th = measured disk angle
```

They do not use angular velocity.

## How To Run

From the repository root:

```bash
python 01_system_dynamics_modeling/ANN/compare_ann_models.py
```

This trains/compares the simple and advanced ANN models and creates plots in:

```text
results/plots/
```

The final generated ANN submission files are saved in:

```text
results/test_outputs/
```

These output files are ignored by Git because they can be regenerated.

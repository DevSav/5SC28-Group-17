# ANN Model Work

This folder explains the ANN part of section 4.1.

The actual ANN code is in:

```text
../../src/models/ann_model.py
../../src/models/lstm_ann_model.py
../../src/train_ann_assignment.py
../../src/train_lstm_ann_assignment.py
../../src/compare_ann_models.py
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
python -m src.compare_ann_models
```

This trains/compares the simple and advanced ANN models and creates plots in:

```text
../../results/plots/
```

The final generated ANN submission files are saved in:

```text
../../results/test_outputs/
```

These output files are ignored by Git because they can be regenerated.

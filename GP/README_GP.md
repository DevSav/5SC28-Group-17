# Final GP Model for Unbalanced Disk System Identification

This folder contains the final Gaussian Process (GP) modeling results for the 5SC28 Design Assignment: Model and Policy Learning for an Unbalanced Disc.

The GP model is used for the system identification part of the assignment. The model uses only the applied motor voltage and the measured disk angle. The angular velocity is not used.

## Final Model Summary

Model type:

```text
GPy Sparse Delta-output GP-NARX
```

Model structure:

```text
Input  = past motor voltage u and past measured disk angle theta
Output = delta theta
```

Final selected hyperparameters:

```text
na = 10
nb = 8
M  = 250 inducing points
Kernel = Matern 5/2
```

Final validation performance:

```text
Prediction RMSE = 0.002859 rad = 0.1638 deg
Simulation RMSE = 0.025508 rad = 1.4615 deg
Simulation NRMS = 5.04 %
```

## Files in This Folder

### `best_gpy_sparse_delta_sim_model.joblib`

This is the final saved GP model selected based on the best validation simulation performance.

It contains the trained sparse GP model and the required preprocessing information for simulation and prediction.

### `gpy_sparse_delta_na10_nb8_M250.joblib`

This is the trained GPy sparse delta-output GP model with:

```text
na = 10
nb = 8
M = 250
```

This file is kept as the direct saved model from the sparse GP training process.

### `gp_delta_gpy_sparse_tuning.py`

This is the main training and tuning script for the final GPy sparse delta-output GP model.

It trains sparse GP models using different settings and evaluates them on validation prediction and simulation performance.

The final selected model is the configuration with:

```text
na = 10
nb = 8
M = 250
```

### `gp_generate_hidden_submissions_gpy_sparse_delta.py`

This script loads the final trained GP model and generates the hidden-test submission files for both tasks:

```text
prediction task
simulation task
```

The generated outputs are saved as `.npz` files.

### `gp_model_interface.py`

This file provides an interface for loading and using the final GP model.

It can be used by other scripts or group members to perform GP prediction or simulation without rewriting the model-loading code.

### `gp_sparse_delta_hidden_prediction_submission.npz`

This is the final hidden-test output file for the prediction task.

It was generated using the final sparse delta-output GP model.

### `gp_sparse_delta_hidden_simulation_submission.npz`

This is the final hidden-test output file for the simulation task.

It was generated using the final sparse delta-output GP model.

### `best_gpy_sparse_delta_validation_simulation.png`

This figure shows the validation simulation result of the final selected GPy sparse delta-output GP model.

It compares the measured validation output with the simulated GP output.

### `best_gpy_sparse_delta_validation_simulation_error.png`

This figure shows the validation simulation error of the final selected GPy sparse delta-output GP model.

It is used to analyze the simulation accuracy and error distribution over the validation data.

### `gpy_sparse_delta_sim_tuning_results.csv`

This file contains the tuning results for the GPy sparse delta-output GP models.

It records the tested model settings and their validation prediction/simulation performance.


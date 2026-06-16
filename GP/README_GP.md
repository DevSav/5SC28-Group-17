# GP Modeling Results

This folder contains the Gaussian Process (GP) modeling files for the 5SC28 Design Assignment.

The GP model was developed for the system identification part of the assignment. The model uses the applied motor voltage and the measured disk angle. Angular velocity was not used in the official GP modeling task.

## Final GP Model Summary

Model type:

```text
GPy Sparse Delta-output GP-NARX
```

Model structure:

```text
Input  = past motor voltage u and past measured disk angle theta
Output = delta theta
```

Final selected setting:

```text
na = 10
nb = 8
M  = 250 inducing points
Kernel = Matern 5/2
```

Validation performance:

```text
Prediction RMSE = 0.002859 rad = 0.1638 deg
Simulation RMSE = 0.025508 rad = 1.4615 deg
Simulation NRMS = 5.04 %
```

## Uploaded Files

### `gp_delta_gpy_sparse_tuning.py`

Main training and tuning script for the final sparse delta-output GP model.

This script was used to train and compare sparse GP models with different settings. The final selected configuration was:

```text
na = 10
nb = 8
M = 250
```

### `gp_generate_hidden_submissions_gpy_sparse_delta.py`

Script used to generate the hidden-test submission files for the GP model.

It generates outputs for both:

```text
prediction task
simulation task
```

### `gp_model_interface.py`

Interface script for loading and using the trained GP model.

Note: the trained `.joblib` model files are not included in this GitHub folder because they exceed the GitHub browser upload size limit. Therefore, this interface requires the model file to be provided separately if someone wants to run prediction or simulation locally.

### `gp_sparse_delta_hidden_prediction_submission.npz`

Final hidden-test output file for the prediction task.

This is one of the final GP submission files.

### `gp_sparse_delta_hidden_simulation_submission.npz`

Final hidden-test output file for the simulation task.

This is one of the final GP submission files.

### `best_gpy_sparse_delta_validation_simulation.png`

Validation simulation plot for the final selected sparse delta-output GP model.

It compares the measured validation output with the GP simulated output.

### `best_gpy_sparse_delta_validation_simulation_error.png`

Validation simulation error plot for the final selected sparse delta-output GP model.

It shows the simulation error over the validation samples.

### `gpy_sparse_delta_sim_tuning_results.csv`

CSV file containing the tuning results for the GPy sparse delta-output GP models.

It records the tested configurations and their corresponding prediction and simulation performance.

## Files Not Included

The following trained model files were not uploaded because they are too large for GitHub browser upload:

```text
best_gpy_sparse_delta_sim_model.joblib
gpy_sparse_delta_na10_nb8_M250.joblib
```

These files are stored locally and can be shared separately if needed.

## Notes

This folder contains the official GP modeling results for the system identification task.

A separate control-oriented GP model may exist for model-based control experiments, but that model is not the official GP benchmark model because it uses angular velocity.


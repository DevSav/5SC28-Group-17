# 03 Single Policy Reference Tracking

This folder contains the reference-tracking controller work from the ZIP file.

This matches the third main assignment step: extend the controller so one policy can swing up and track a reference near the top position.

## Folder Structure

```text
03_single_policy_reference_tracking/
  reference_tracking/
    DQN reference-tracking controller and result files.
```

## Important Results

```text
reference_tracking/results_dqn_reference_tracking/
```

This folder already contains result files, plots, trained parameters, and logs from the controller teammate.

## Run Again If Needed

Probably not needed for the report, because the result files are already present.

To reproduce it:

```bash
cd 03_single_policy_reference_tracking/reference_tracking
python main_train_dqn_reference_tracking.py
```

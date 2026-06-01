# 5SC28 Design Project - Group 17

This repository contains the work for the 5SC28 Machine Learning for Systems and Control design project.

Project topic: model and policy learning for an unbalanced disc.

## What We Need To Do

The assignment has three main parts:

1. Learn models of the system dynamics.
   - Gaussian Process model.
   - Artificial Neural Network model.
   - Test both prediction and simulation performance.

2. Learn a swing-up policy.
   - Q-learning based method.
   - Actor-critic or model-internalization method.
   - Start in simulation before trying the real setup.

3. Extend the policy.
   - One single policy should swing up the disc and track a reference around the top position.

## Grading Focus

The rubric mainly checks:

- quality and quantity of obtained results;
- design choices and systematic approach;
- use of course theory;
- critical discussion of results and limitations;
- clarity of report and presentation;
- individual contribution.

## Repository Structure

```text
assignment_files/
  gym-unbalanced-disk/  Required benchmark data, simulator, and checker files.
data/
  raw/                 Original downloaded data. Do not edit these files.
  processed/           Cleaned data used by our scripts.
docs/                  Assignment notes, planning, and useful references.
notebooks/             Experiments and plots.
reports/               Report drafts and final report material.
results/
  models/              Saved trained models.
  plots/               Figures for report and presentation.
  test_outputs/        Output files that must be submitted.
src/
  models/              GP and ANN model code.
  policies/            Reinforcement learning policy code.
  utils/               Small helper functions.
```

## Simple Workflow

1. Use the files in `assignment_files/gym-unbalanced-disk/`.
2. Use notebooks only for quick experiments.
3. Put working code in `src/`.
4. Save final plots in `results/plots/`.
5. Save final test outputs in `results/test_outputs/`.
6. Write down decisions and failed attempts in `docs/experiment_log.md`.

## Section 4.1 Modelling Checklist

Section 4.1 of the assignment is about modelling the system dynamics.

Required:

- GP model: one model structure is enough, for example NARX or NOE.
- ANN model: at least one simple model, for example NFIR, NARX, or NOE.
- ANN model: at least one more advanced learning architecture.
- Input: applied motor voltage.
- Output: measured disk angle.
- Do not use angular velocity for this modelling part.
- Test the final models on the provided prediction and simulation tasks.
- Save the final prediction and simulation output files for submission.
- Explain the model choices, hyperparameters, validation, and limitations.

Current ANN status:

- Done: simple ANN NARX baseline.
- Done: prediction and simulation output files are generated.
- Done: held-out validation/test split is used before creating hidden-test outputs.
- Not done yet: advanced ANN architecture. A simple next option is an RNN/LSTM or a small NOE-style recurrent model.

## Run The Simple ANN Model

The first ANN model is a small NARX neural network.

NARX means the network predicts the next angle from old voltages and old measured angles:

```text
old voltages + old measured angles -> next measured angle
```

This follows section 4.1 because it uses only:

- `u`: motor voltage;
- `th`: measured disk angle.

It does not use angular velocity.

Run the assignment ANN script:

```bash
python -m src.train_ann_assignment
```

This uses:

```text
assignment_files/gym-unbalanced-disk/disc-benchmark-files/training-val-test-data.npz
```

It creates:

```text
results/models/ann_narx_assignment_model.pt
results/test_outputs/ann_hidden_prediction_submission.npz
results/test_outputs/ann_hidden_simulation_submission.npz
```

These result files are ignored by Git because they can be regenerated.

Optional settings:

```bash
python -m src.train_ann_assignment --input-delay 15 --output-delay 15 --hidden-neurons 50 --epochs 1500
```

The older generic script also exists. It expects a CSV file with an input column and an output column.

Accepted input column names:

- `u`
- `input`
- `voltage`

Accepted output column names:

- `y`
- `theta`
- `angle`

Example:

```bash
python -m src.train_ann --data data/raw/my_training_data.csv
```

## Group

Group 17

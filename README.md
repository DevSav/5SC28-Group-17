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

1. Put the downloaded benchmark data in `data/raw/`.
2. Use notebooks for quick experiments.
3. Move working code into `src/`.
4. Save final plots in `results/plots/`.
5. Save final test outputs in `results/test_outputs/`.
6. Write down decisions and failed attempts in `docs/experiment_log.md`.

## Run The Simple ANN Model

The first ANN model is a small NARX neural network. It expects a CSV file with an input column and an output column.

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

For the downloaded assignment data, use:

```bash
python -m src.train_ann_assignment
```

## Group

Group 17

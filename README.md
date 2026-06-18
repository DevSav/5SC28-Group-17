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

## Assignment Flow Structure

```text
01_system_dynamics_modeling/
  ANN/                 Short guide to our ANN work.
  GP/                  Gaussian Process work from the GP teammate.

02_policy_learning_swingup/
  01_q_learning_swingup/
  02_hybrid_policy_model_internalization/

03_single_policy_reference_tracking/
  reference_tracking/

04_reports_and_submission/
  Notes about report files and final submission outputs.

assignment_files/      Required benchmark data, simulator, and checker files.
docs/                  Experiment log and planning notes.
reports/               Report drafts and final report material.
```

This structure follows the assignment flow:

1. First we learn system dynamics models: ANN and GP.
2. Then we learn swing-up policies.
3. Then we extend the control work to reference tracking.
4. Finally we collect plots, outputs, and report material.

## Simple Workflow

1. Use the files in `assignment_files/gym-unbalanced-disk/`.
2. Read `01_system_dynamics_modeling/ANN/README.md` for the ANN part.
3. Read `01_system_dynamics_modeling/GP/README_GP.md` for the GP part.
4. Read the READMEs in `02_policy_learning_swingup/` and `03_single_policy_reference_tracking/` for the control part.
5. Save ANN plots in `01_system_dynamics_modeling/ANN/results/plots/`.
6. Save ANN test outputs in `01_system_dynamics_modeling/ANN/results/test_outputs/`.
7. Write down decisions and failed attempts in `docs/experiment_log.md`.

The experiment log is here:

```text
docs/experiment_log.md
```

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
- Done: advanced ANN LSTM model.
- Done: prediction and simulation output files are generated.
- Done: held-out validation/test split is used before creating hidden-test outputs.
- Done: provided submission checker runs on the generated files.

Practical session check:

- Lecture 2 ANN practical uses PyTorch for neural networks and backpropagation.
- Lecture 2 also has an ANN for NARX exercise.
- That NARX exercise builds rows like `[old inputs, old outputs] -> next output`.
- Lecture 3 Deep Learning practical uses train/validation splitting, normalization, Adam, and recurrent models.
- Our simple ANN follows the Lecture 2 NARX style, but with the real unbalanced-disc data.
- Our advanced ANN follows the Lecture 3 recurrent model style, but with the real unbalanced-disc data.

## Run The Simple ANN Model

The first ANN model is a small NARX neural network. This is the simple ANN result required in section 4.1.

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
python 01_system_dynamics_modeling/ANN/train_ann_assignment.py
```

This uses:

```text
assignment_files/gym-unbalanced-disk/disc-benchmark-files/training-val-test-data.npz
```

It creates:

```text
01_system_dynamics_modeling/ANN/results/models/ann_narx_assignment_model.pt
01_system_dynamics_modeling/ANN/results/test_outputs/ann_hidden_prediction_submission.npz
01_system_dynamics_modeling/ANN/results/test_outputs/ann_hidden_simulation_submission.npz
```

These result files are ignored by Git because they can be regenerated.

Optional settings:

```bash
python 01_system_dynamics_modeling/ANN/train_ann_assignment.py --input-delay 15 --output-delay 15 --hidden-neurons 50 --epochs 1500
```

## Run The Advanced ANN Model

The second ANN model is a small LSTM model. This is the advanced ANN result required in section 4.1.

The LSTM uses short sequences:

```text
[[old voltage, old angle], ...] -> next measured angle
```

Run it with:

```bash
python 01_system_dynamics_modeling/ANN/train_lstm_ann_assignment.py
```

It creates:

```text
01_system_dynamics_modeling/ANN/results/models/ann_lstm_assignment_model.pt
01_system_dynamics_modeling/ANN/results/test_outputs/ann_lstm_hidden_prediction_submission.npz
01_system_dynamics_modeling/ANN/results/test_outputs/ann_lstm_hidden_simulation_submission.npz
```

The default LSTM setting is the tuned setting:

```text
history = 15
hidden size = 60
epochs = 120
```

Latest standalone held-out split result:

```text
Prediction RMSE: 0.004049 rad = 0.232 degrees
Simulation RMSE: 0.041892 rad = 2.40 degrees
```

These are not the official hidden-test scores. They are our own train/validation/test split scores from `training-val-test-data.npz`.

The provided checker can be used to check the output file format:

```bash
python assignment_files/gym-unbalanced-disk/disc-benchmark-files/submission-file-checker.py 01_system_dynamics_modeling/ANN/results/test_outputs/ann_lstm_hidden_prediction_submission.npz assignment_files/gym-unbalanced-disk/disc-benchmark-files/hidden-test-prediction-submission-file.npz

python assignment_files/gym-unbalanced-disk/disc-benchmark-files/submission-file-checker.py 01_system_dynamics_modeling/ANN/results/test_outputs/ann_lstm_hidden_simulation_submission.npz assignment_files/gym-unbalanced-disk/disc-benchmark-files/hidden-test-simulation-submission-file.npz
```

For final submission, the LSTM files are currently the better ANN outputs to share because they perform better on our held-out split.

The best tuned LSTM output files are generated by the comparison script:

```text
01_system_dynamics_modeling/ANN/results/models/ann_lstm_tuned_assignment_model.pt
01_system_dynamics_modeling/ANN/results/test_outputs/ann_lstm_tuned_hidden_prediction_submission.npz
01_system_dynamics_modeling/ANN/results/test_outputs/ann_lstm_tuned_hidden_simulation_submission.npz
```

## Compare The ANN Models

To make plots and CSV files comparing the simple NARX ANN with the advanced LSTM ANN, run:

```bash
python 01_system_dynamics_modeling/ANN/compare_ann_models.py
```

This creates:

```text
01_system_dynamics_modeling/ANN/results/plots/ann_model_scores.csv
01_system_dynamics_modeling/ANN/results/plots/lstm_tuning_results.csv
01_system_dynamics_modeling/ANN/results/plots/ann_simulation_comparison.csv
01_system_dynamics_modeling/ANN/results/plots/ann_prediction_comparison.png
01_system_dynamics_modeling/ANN/results/plots/ann_simulation_comparison.png
01_system_dynamics_modeling/ANN/results/plots/ann_error_barplot.png
01_system_dynamics_modeling/ANN/results/test_outputs/ann_lstm_tuned_hidden_prediction_submission.npz
01_system_dynamics_modeling/ANN/results/test_outputs/ann_lstm_tuned_hidden_simulation_submission.npz
```

The current comparison run gave:

```text
simple NARX ANN:
  prediction RMSE = 0.0426 rad = 2.44 degrees
  simulation RMSE = 0.2815 rad = 16.13 degrees

advanced LSTM ANN:
  prediction RMSE = 0.00366 rad = 0.210 degrees
  simulation RMSE = 0.0291 rad = 1.67 degrees
```

This is useful for the report because it clearly shows why the LSTM is the stronger ANN result.

The tuning CSV keeps the progress so far. It compares several LSTM settings, so we can show that the final model was selected systematically instead of guessed.

To make plots comparing all ANN methods and all LSTM tuning runs, use:

```bash
python 01_system_dynamics_modeling/ANN/plot_all_ann_results.py
```

This creates:

```text
01_system_dynamics_modeling/ANN/results/plots/ann_all_methods_scores.csv
01_system_dynamics_modeling/ANN/results/plots/ann_all_methods_barplot.png
01_system_dynamics_modeling/ANN/results/plots/ann_lstm_tuning_plot.png
```

These plots are useful for the report because they show the full path from the simple NARX baseline to the tuned LSTM model.

## Group

Group 17

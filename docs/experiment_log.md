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

## 2026-06-01 ANN Comparison Plots

Date: 2026-06-01

Person: Group 17 ANN part

Goal: Make figures and CSV files comparing the simple ANN and advanced ANN.

Method: Trained the simple NARX ANN and advanced LSTM ANN on the same train/validation/test split, then plotted prediction, simulation, and RMSE bars.

Result: The comparison files are saved in `results/plots/`. In this run, the LSTM had prediction RMSE about 0.477 degrees and simulation RMSE about 2.50 degrees. The simple NARX ANN had prediction RMSE about 2.44 degrees and simulation RMSE about 16.13 degrees.

What went well: The plots clearly show that the LSTM is better for both prediction and simulation.

What did not work: These are still held-out split results, not official hidden-test scores.

Next step: Use the plots in the report and explain why the recurrent model handles simulation better.

## 2026-06-01 LSTM Tuning

Date: 2026-06-01

Person: Group 17 ANN part

Goal: Improve the advanced ANN model while keeping the old results for comparison.

Method: Tried several LSTM settings and saved all scores in `results/plots/lstm_tuning_results.csv`.

Settings tried: history 10 or 15, hidden size 20, 40, or 60, and 80 or 120 epochs.

Result: Best tuning result was history 15, hidden size 60, 120 epochs. It reached prediction RMSE about 0.218 degrees and simulation RMSE about 1.77 degrees in the tuning run. The regenerated comparison plot run gave prediction RMSE about 0.210 degrees and simulation RMSE about 1.67 degrees.

What went well: The tuned LSTM is close to the benchmark good NN simulation value of 1.55 degrees mentioned in the assignment repository README.

What did not work: Increasing model size does not always improve simulation. The hidden size 40 run had better prediction than smaller models, but worse simulation than the hidden size 60 model.

Next step: Use the tuned LSTM as the final ANN model and use the tuning CSV to support the systematic approach in the report.

## 2026-06-01 All ANN Method Plots

Date: 2026-06-01

Person: Group 17 ANN part

Goal: Make plots that compare every ANN result, not only the final two models.

Method: Used `results/plots/ann_model_scores.csv` and `results/plots/lstm_tuning_results.csv` to make an all-method bar plot and an LSTM tuning plot.

Result: Saved `ann_all_methods_barplot.png`, `ann_lstm_tuning_plot.png`, and `ann_all_methods_scores.csv` in `results/plots/`.

What went well: The plots show the improvement path clearly: simple NARX is the weakest, and the tuned LSTM with history 15 and hidden size 60 is the strongest.

What did not work: The plots compare held-out split results only, not official hidden-test scores.

Next step: Use these plots in the report to support the systematic approach and model-selection discussion.

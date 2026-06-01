# Assignment Summary - Group 17

## System

The system is an unbalanced disc that behaves like an inverted pendulum.

The input is the motor voltage. The voltage must stay between -3 V and +3 V.

For the modelling task, only the measured angle should be used as output. The angular velocity should not be used there.

For the control task, both angle and angular velocity may be used.

## Part 1: Modelling The Dynamics

This part is 50 percent of the project grade.

We need:

- one Gaussian Process model;
- at least one simple ANN model, for example NFIR, NARX, or NOE;
- at least one more advanced ANN model;
- prediction and simulation tests;
- clear motivation for all model choices.

Current ANN branch status:

- simple ANN model: NARX feedforward neural network;
- advanced ANN model: LSTM recurrent neural network;
- input used: motor voltage `u`;
- output used: measured angle `th`;
- angular velocity: not used;
- validation/test: chronological train, validation, and test split;
- generated files: hidden prediction and simulation `.npz` output files;
- provided checker: runs on the generated hidden prediction and simulation files;
- best current ANN result: LSTM, because it has better held-out prediction and simulation results.

## Part 2: Swing-Up Policy

This part is 40 percent of the project grade.

We need:

- one Q-learning based method;
- one actor-critic or model-internalization method;
- training in simulation first;
- discussion of advantages and disadvantages.

## Part 3: Single Policy For Swing-Up And Tracking

This part is 10 percent of the project grade.

We need one policy that:

- swings the pendulum from the bottom to the top;
- tracks a reference around the top position;
- does not use a switching controller made from separate single-target policies.

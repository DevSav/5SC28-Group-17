# Control Work

This folder contains the controller work from the ZIP file.

The original controller ZIP is kept in `99_original_zip/`.

## Folder Structure

```text
control/
  01_q_learning_swingup/
    DQN / Q-learning swing-up controller.

  02_hybrid_policy_model_internalization/
    Hybrid policy / model-internalization style controller.
    This includes teacher data, RBF imitation, hybrid policy optimization,
    and robustness evaluation results.

  03_reference_tracking/
    DQN reference-tracking controller.

  99_original_zip/
    Original uploaded ZIP file, kept unchanged.
```

## What Is Already Included

The ZIP already contains trained/evaluated result folders.

Important result folders:

```text
01_q_learning_swingup/results_dqn_swingup/
02_hybrid_policy_model_internalization/results_hybrid_policy_optimization/
02_hybrid_policy_model_internalization/results_optimized_policy_robustness/
03_reference_tracking/results_dqn_reference_tracking/
```

## Do We Need To Run Anything?

Probably not for the report.

The result folders already contain metrics, plots, trained parameters, and logs.

Run scripts only if we want to reproduce or improve the control results.

## Main Scripts

Q-learning swing-up:

```bash
cd control/01_q_learning_swingup
python main_train_dpn_swingup.py
```

Hybrid/model-internalization style controller:

```bash
cd control/02_hybrid_policy_model_internalization
python main_optimize_hybrid_policy_simulator.py
python main_evaluate_optimized_policy_robustness.py
```

Reference tracking:

```bash
cd control/03_reference_tracking
python main_train_dqn_reference_tracking.py
```

## Notes

The controller folders include their own copy of the simulator package and result files.

The controller READMEs inside each folder are mostly the original simulator README files, not project-specific explanations.

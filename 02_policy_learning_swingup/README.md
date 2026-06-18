# 02 Policy Learning: Swing-Up

This folder contains the swing-up controller work from the ZIP file.

This matches the second main assignment step: learn a policy that can swing the disk up.

The original controller ZIP is kept in `../99_original_archives/control_zip/`.

## Folder Structure

```text
02_policy_learning_swingup/
  01_q_learning_swingup/
    DQN / Q-learning swing-up controller.

  02_hybrid_policy_model_internalization/
    Hybrid policy / model-internalization style controller.
    This includes teacher data, RBF imitation, hybrid policy optimization,
    and robustness evaluation results.
```

## What Is Already Included

The ZIP already contains trained/evaluated result folders.

Important result folders:

```text
01_q_learning_swingup/results_dqn_swingup/
02_hybrid_policy_model_internalization/results_hybrid_policy_optimization/
02_hybrid_policy_model_internalization/results_optimized_policy_robustness/
```

## Do We Need To Run Anything?

Probably not for the report.

The result folders already contain metrics, plots, trained parameters, and logs.

Run scripts only if we want to reproduce or improve the control results.

## Main Scripts

Q-learning swing-up:

```bash
cd 02_policy_learning_swingup/01_q_learning_swingup
python main_train_dpn_swingup.py
```

Hybrid/model-internalization style controller:

```bash
cd 02_policy_learning_swingup/02_hybrid_policy_model_internalization
python main_optimize_hybrid_policy_simulator.py
python main_evaluate_optimized_policy_robustness.py
```

## Notes

The controller folders include their own copy of the simulator package and result files.

The controller READMEs inside each folder are mostly the original simulator README files, not project-specific explanations.

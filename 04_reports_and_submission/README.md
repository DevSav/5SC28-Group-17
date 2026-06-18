# 04 Reports And Submission

This folder explains where the final project material should be collected.

## Professor Announcement Requirements

Deadline: June 22.

Each student submits an individual reflection report separately.

Individual reflection:

```text
maximum 500 words
use the provided template
include own contribution
include team contribution
include the contribution ranking table
```

The group submits one zip file.

The group zip must contain:

```text
1 report PDF
2 folders
```

The report PDF must be:

```text
2-column IEEE format
6-12 pages
include names and student numbers
```

The first folder must contain the requested prediction and simulation test files:

```text
at least 1 ANN prediction result
at least 1 ANN simulation result
at least 1 GP prediction result
at least 1 GP simulation result
```

The second folder must contain:

```text
relevant code used during the project
possible additional material
```

The assessment starts with a 10-minute group presentation and then a group discussion/Q&A.
Everyone should be able to answer questions about all project parts and course theory.

## Zip Template

Use this folder as the draft structure for the final group zip:

```text
submission_zip_template/
  Group17_design_report.pdf
  01_requested_test_files/
  02_code_and_additional_material/
```

The PDF is not generated yet.
When the report is finished, export it as `Group17_design_report.pdf` and place it at the root of the zip.

## Report Material

Report drafts and written material should go in:

```text
../reports/
```

Useful plots are mainly in:

```text
../01_system_dynamics_modeling/ANN/results/plots/
```

The experiment log is:

```text
../docs/experiment_log.md
```

## Final Output Files

ANN hidden-test output files are generated in:

```text
../01_system_dynamics_modeling/ANN/results/test_outputs/
```

GP hidden-test output files are in:

```text
../01_system_dynamics_modeling/GP/
```

Controller result files are in:

```text
../02_policy_learning_swingup/
../03_single_policy_reference_tracking/
```

Before submission, we should check that the final report clearly states which files are the final outputs from ANN, GP, and control.

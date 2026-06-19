from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANN_SCORES = ROOT / "01_system_dynamics_modeling" / "ANN" / "results" / "plots" / "ann_model_scores.csv"
GP_SCORES = ROOT / "01_system_dynamics_modeling" / "GP" / "gpy_sparse_delta_sim_tuning_results.csv"
OUTPUT = ROOT / "reports" / "figures" / "ann_gp_model_comparison.png"


GRU_COLOR = "#2ca02c"
LSTM_COLOR = "#ff7f0e"
GP_COLOR = "#1f77b4"


def add_labels(axis, bars):
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.03,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    ann = pd.read_csv(ANN_SCORES)
    gp = pd.read_csv(GP_SCORES).iloc[0]

    gru = ann[ann["model"] == "advanced GRU ANN"].iloc[0]
    lstm = ann[ann["model"] == "advanced LSTM ANN"].iloc[0]

    prediction_values = [
        gru["prediction_rmse_deg"],
        gp["prediction_rmse_deg"],
    ]
    simulation_values = [
        lstm["simulation_rmse_deg"],
        gp["simulation_rmse_deg"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)

    pred_bars = axes[0].bar(
        ["ANN\n(GRU)", "GP"],
        prediction_values,
        color=[GRU_COLOR, GP_COLOR],
    )
    axes[0].set_title("Prediction")
    axes[0].set_ylabel("RMSE (degrees)")
    axes[0].set_ylim(0, 2.2)
    add_labels(axes[0], pred_bars)

    sim_bars = axes[1].bar(
        ["ANN\n(LSTM)", "GP"],
        simulation_values,
        color=[LSTM_COLOR, GP_COLOR],
    )
    axes[1].set_title("Simulation")
    axes[1].set_ylim(0, 2.2)
    add_labels(axes[1], sim_bars)

    fig.suptitle("Best ANN vs GP validation error", fontsize=13)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=200)
    plt.close(fig)

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()

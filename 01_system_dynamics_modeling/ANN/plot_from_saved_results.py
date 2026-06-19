import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import PLOTS_DIR, SAVED_ARRAYS_DIR


COLORS = {
    "measured": "black",
    "narx": "#7f7f7f",
    "gru": "#2ca02c",
    "lstm": "#ff7f0e",
}


def plot_prediction(data):
    samples = data["sample"].to_numpy()[:500]

    plt.figure(figsize=(10, 4))
    plt.plot(samples, data["measured_angle"][:500], label="measured", color=COLORS["measured"], linewidth=2)
    plt.plot(samples, data["simple_narx_prediction"][:500], label="NARX", color=COLORS["narx"])
    plt.plot(samples, data["advanced_gru_prediction"][:500], label="GRU", color=COLORS["gru"])
    plt.plot(samples, data["advanced_lstm_prediction"][:500], label="LSTM", color=COLORS["lstm"])
    plt.xlabel("sample")
    plt.ylabel("angle (rad)")
    plt.title("ANN prediction comparison from saved results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "saved_ann_prediction_comparison.png", dpi=150)
    plt.close()


def plot_simulation(data):
    samples = data["sample"].to_numpy()[:800]

    plt.figure(figsize=(10, 4))
    plt.plot(samples, data["measured_angle"][:800], label="measured", color=COLORS["measured"], linewidth=2)
    plt.plot(samples, data["simple_narx_simulation"][:800], label="NARX", color=COLORS["narx"])
    plt.plot(samples, data["advanced_gru_simulation"][:800], label="GRU", color=COLORS["gru"])
    plt.plot(samples, data["advanced_lstm_simulation"][:800], label="LSTM", color=COLORS["lstm"])
    plt.xlabel("sample")
    plt.ylabel("angle (rad)")
    plt.title("ANN simulation comparison from saved results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "saved_ann_simulation_comparison.png", dpi=150)
    plt.close()


def plot_scores(scores):
    names = scores["model"].str.replace("simple ", "", regex=False).str.replace("advanced ", "", regex=False)
    x = np.arange(len(names))
    width = 0.35
    colors = [COLORS["narx"], COLORS["gru"], COLORS["lstm"]]

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, scores["prediction_rmse_deg"], width, color=colors, alpha=0.55, label="prediction")
    plt.bar(x + width / 2, scores["simulation_rmse_deg"], width, color=colors, alpha=1.0, label="simulation")
    plt.yscale("log")
    plt.xticks(x, names, rotation=12, ha="right")
    plt.ylabel("RMSE (degrees)")
    plt.title("ANN model comparison from saved results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "saved_ann_score_comparison.png", dpi=150)
    plt.close()


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_data = pd.read_csv(SAVED_ARRAYS_DIR / "ann_prediction_results.csv")
    simulation_data = pd.read_csv(SAVED_ARRAYS_DIR / "ann_simulation_results.csv")
    scores = pd.read_csv(SAVED_ARRAYS_DIR / "ann_scores.csv")

    plot_prediction(prediction_data)
    plot_simulation(simulation_data)
    plot_scores(scores)

    print("Saved:", PLOTS_DIR / "saved_ann_prediction_comparison.png")
    print("Saved:", PLOTS_DIR / "saved_ann_simulation_comparison.png")
    print("Saved:", PLOTS_DIR / "saved_ann_score_comparison.png")


if __name__ == "__main__":
    main()

import pandas as pd
from matplotlib import pyplot as plt

from config import PLOTS_DIR


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_all_errors(data):
    x = range(len(data))
    width = 0.35

    plt.figure(figsize=(11, 5))
    prediction_bars = plt.bar(
        [i - width / 2 for i in x],
        data["prediction_rmse_deg"],
        width,
        label="prediction",
    )
    simulation_bars = plt.bar(
        [i + width / 2 for i in x],
        data["simulation_rmse_deg"],
        width,
        label="simulation",
    )

    add_labels(prediction_bars)
    add_labels(simulation_bars)

    plt.xticks(x, data["model"], rotation=25, ha="right")
    plt.yscale("log")
    plt.ylabel("RMSE (degrees)")
    plt.title("All ANN model results (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ann_all_methods_barplot.png", dpi=150)
    plt.close()


def plot_tuning_only(tuning_data):
    labels = []
    for _, row in tuning_data.iterrows():
        labels.append(f"h={int(row['history'])}, n={int(row['hidden_size'])}")

    plt.figure(figsize=(9, 5))
    plt.plot(labels, tuning_data["prediction_rmse_deg"], "o-", label="prediction")
    plt.plot(labels, tuning_data["simulation_rmse_deg"], "o-", label="simulation")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("RMSE (degrees)")
    plt.title("LSTM tuning comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ann_lstm_tuning_plot.png", dpi=150)
    plt.close()


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    simple_and_best = pd.read_csv(PLOTS_DIR / "ann_model_scores.csv")
    tuning_data = pd.read_csv(PLOTS_DIR / "lstm_tuning_results.csv")

    rows = []

    simple_row = simple_and_best[simple_and_best["model"] == "simple NARX ANN"].iloc[0]
    rows.append(
        {
            "model": "simple NARX ANN",
            "prediction_rmse_deg": simple_row["prediction_rmse_deg"],
            "simulation_rmse_deg": simple_row["simulation_rmse_deg"],
        }
    )

    if "advanced GRU ANN" in simple_and_best["model"].values:
        gru_row = simple_and_best[simple_and_best["model"] == "advanced GRU ANN"].iloc[0]
        rows.append(
            {
                "model": "advanced GRU ANN",
                "prediction_rmse_deg": gru_row["prediction_rmse_deg"],
                "simulation_rmse_deg": gru_row["simulation_rmse_deg"],
            }
        )

    if "advanced LSTM ANN" in simple_and_best["model"].values:
        lstm_row = simple_and_best[simple_and_best["model"] == "advanced LSTM ANN"].iloc[0]
        rows.append(
            {
                "model": "advanced LSTM ANN",
                "prediction_rmse_deg": lstm_row["prediction_rmse_deg"],
                "simulation_rmse_deg": lstm_row["simulation_rmse_deg"],
            }
        )

    for _, row in tuning_data.iterrows():
        rows.append(
            {
                "model": f"LSTM h={int(row['history'])}, n={int(row['hidden_size'])}",
                "prediction_rmse_deg": row["prediction_rmse_deg"],
                "simulation_rmse_deg": row["simulation_rmse_deg"],
            }
        )

    all_data = pd.DataFrame(rows)
    all_data.to_csv(PLOTS_DIR / "ann_all_methods_scores.csv", index=False)

    plot_all_errors(all_data)
    plot_tuning_only(tuning_data)

    print("Saved:", PLOTS_DIR / "ann_all_methods_scores.csv")
    print("Saved:", PLOTS_DIR / "ann_all_methods_barplot.png")
    print("Saved:", PLOTS_DIR / "ann_lstm_tuning_plot.png")


if __name__ == "__main__":
    main()

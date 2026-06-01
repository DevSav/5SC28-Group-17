import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from src.config import MODELS_DIR, PLOTS_DIR, RAW_DATA_DIR, ROOT, TEST_OUTPUTS_DIR
from src.models.ann_model import ANNModel, make_narx_data
from src.models.lstm_ann_model import LSTMANNModel, make_lstm_data


INPUT_DELAY = 15
OUTPUT_DELAY = 15
HISTORY = 15
SIMULATION_START_SAMPLES = 50


def rmse(real, predicted):
    return np.mean((real - predicted) ** 2) ** 0.5


def rmse_degrees(real, predicted):
    return rmse(real, predicted) / (2 * np.pi) * 360


def get_benchmark_folder():
    folder = ROOT / "assignment_files" / "gym-unbalanced-disk" / "disc-benchmark-files"

    if not folder.exists():
        folder = RAW_DATA_DIR / "gym-unbalanced-disk" / "disc-benchmark-files"

    return folder


def plot_prediction(th_real, narx_pred, lstm_pred):
    plt.figure(figsize=(10, 4))
    plt.plot(th_real[:500], label="measured")
    plt.plot(narx_pred[:500], label="simple NARX ANN")
    plt.plot(lstm_pred[:500], label="advanced LSTM ANN")
    plt.xlabel("sample")
    plt.ylabel("angle (rad)")
    plt.title("Prediction comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ann_prediction_comparison.png", dpi=150)
    plt.close()


def plot_simulation(th_real, narx_sim, lstm_sim):
    plt.figure(figsize=(10, 4))
    plt.plot(th_real[:800], label="measured")
    plt.plot(narx_sim[:800], label="simple NARX ANN")
    plt.plot(lstm_sim[:800], label="advanced LSTM ANN")
    plt.xlabel("sample")
    plt.ylabel("angle (rad)")
    plt.title("Simulation comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ann_simulation_comparison.png", dpi=150)
    plt.close()


def plot_scores(scores):
    names = [row["model"] for row in scores]
    prediction_scores = [row["prediction_rmse_deg"] for row in scores]
    simulation_scores = [row["simulation_rmse_deg"] for row in scores]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, prediction_scores, width, label="prediction")
    plt.bar(x + width / 2, simulation_scores, width, label="simulation")
    plt.xticks(x, names)
    plt.ylabel("RMSE (degrees)")
    plt.title("ANN model error comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ann_error_barplot.png", dpi=150)
    plt.close()


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    benchmark_folder = get_benchmark_folder()
    prediction_file = benchmark_folder / "hidden-test-prediction-submission-file.npz"
    simulation_file = benchmark_folder / "hidden-test-simulation-submission-file.npz"

    data = np.load(benchmark_folder / "training-val-test-data.npz")
    u = data["u"]
    th = data["th"]

    train_end = int(0.70 * len(th))
    validation_end = int(0.85 * len(th))

    u_train = u[:train_end]
    th_train = th[:train_end]
    u_val = u[train_end:validation_end]
    th_val = th[train_end:validation_end]
    u_test = u[validation_end:]
    th_test = th[validation_end:]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Training simple NARX ANN")
    Xtrain, Ytrain = make_narx_data(u_train, th_train, INPUT_DELAY, OUTPUT_DELAY)
    Xval, Yval = make_narx_data(u_val, th_val, INPUT_DELAY, OUTPUT_DELAY)
    Xtest, Ytest = make_narx_data(u_test, th_test, INPUT_DELAY, OUTPUT_DELAY)

    narx_model = ANNModel(number_of_inputs=Xtrain.shape[1], hidden_neurons=50)
    narx_model.fit(Xtrain, Ytrain, Xval, Yval, epochs=1500, patience=40)
    narx_prediction = narx_model.predict(Xtest)
    narx_simulation = narx_model.simulate(
        u_test,
        th_test[:SIMULATION_START_SAMPLES],
        INPUT_DELAY,
        OUTPUT_DELAY,
    )

    print("Training advanced LSTM ANN")
    Xtrain_lstm, Ytrain_lstm = make_lstm_data(u_train, th_train, history=HISTORY)
    Xval_lstm, Yval_lstm = make_lstm_data(u_val, th_val, history=HISTORY)
    Xtest_lstm, Ytest_lstm = make_lstm_data(u_test, th_test, history=HISTORY)

    lstm_model = LSTMANNModel(hidden_size=60)
    lstm_model.fit(Xtrain_lstm, Ytrain_lstm, Xval_lstm, Yval_lstm, epochs=120)
    lstm_prediction = lstm_model.predict(Xtest_lstm)
    lstm_simulation = lstm_model.simulate(
        u_test,
        th_test[:SIMULATION_START_SAMPLES],
        history=HISTORY,
    )

    hidden_prediction_data = np.load(prediction_file)
    upast = hidden_prediction_data["upast"]
    thpast = hidden_prediction_data["thpast"]
    hidden_prediction_input = np.stack(
        [
            upast[:, 15 - HISTORY :],
            thpast[:, 15 - HISTORY :],
        ],
        axis=2,
    )
    thnow = lstm_model.predict(hidden_prediction_input)

    hidden_simulation_data = np.load(simulation_file)
    u_hidden = hidden_simulation_data["u"]
    th_hidden = hidden_simulation_data["th"]
    th_simulated = lstm_model.simulate(
        u_hidden,
        th_hidden[:SIMULATION_START_SAMPLES],
        history=HISTORY,
    )

    skip = SIMULATION_START_SAMPLES

    scores = [
        {
            "model": "simple NARX ANN",
            "prediction_rmse_rad": rmse(Ytest.reshape(-1), narx_prediction),
            "prediction_rmse_deg": rmse_degrees(Ytest.reshape(-1), narx_prediction),
            "simulation_rmse_rad": rmse(th_test[skip:], narx_simulation[skip:]),
            "simulation_rmse_deg": rmse_degrees(th_test[skip:], narx_simulation[skip:]),
        },
        {
            "model": "advanced LSTM ANN",
            "prediction_rmse_rad": rmse(Ytest_lstm.reshape(-1), lstm_prediction),
            "prediction_rmse_deg": rmse_degrees(Ytest_lstm.reshape(-1), lstm_prediction),
            "simulation_rmse_rad": rmse(th_test[skip:], lstm_simulation[skip:]),
            "simulation_rmse_deg": rmse_degrees(th_test[skip:], lstm_simulation[skip:]),
        },
    ]

    pd.DataFrame(scores).to_csv(PLOTS_DIR / "ann_model_scores.csv", index=False)

    comparison = pd.DataFrame(
        {
            "sample": np.arange(len(th_test)),
            "measured_angle": th_test,
            "simple_narx_simulation": narx_simulation,
            "advanced_lstm_simulation": lstm_simulation,
        }
    )
    comparison.to_csv(PLOTS_DIR / "ann_simulation_comparison.csv", index=False)

    torch.save(lstm_model, MODELS_DIR / "ann_lstm_tuned_assignment_model.pt")
    np.savez(
        TEST_OUTPUTS_DIR / "ann_lstm_tuned_hidden_prediction_submission.npz",
        upast=upast,
        thpast=thpast,
        thnow=thnow,
    )
    np.savez(
        TEST_OUTPUTS_DIR / "ann_lstm_tuned_hidden_simulation_submission.npz",
        u=u_hidden,
        th=th_simulated,
    )

    plot_prediction(Ytest.reshape(-1), narx_prediction, lstm_prediction)
    plot_simulation(th_test, narx_simulation, lstm_simulation)
    plot_scores(scores)

    print("Saved:", PLOTS_DIR / "ann_model_scores.csv")
    print("Saved:", PLOTS_DIR / "ann_simulation_comparison.csv")
    print("Saved:", PLOTS_DIR / "ann_prediction_comparison.png")
    print("Saved:", PLOTS_DIR / "ann_simulation_comparison.png")
    print("Saved:", PLOTS_DIR / "ann_error_barplot.png")
    print("Saved:", MODELS_DIR / "ann_lstm_tuned_assignment_model.pt")
    print("Saved:", TEST_OUTPUTS_DIR / "ann_lstm_tuned_hidden_prediction_submission.npz")
    print("Saved:", TEST_OUTPUTS_DIR / "ann_lstm_tuned_hidden_simulation_submission.npz")


if __name__ == "__main__":
    main()

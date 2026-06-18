import numpy as np
import pandas as pd
import torch

from config import PLOTS_DIR, RAW_DATA_DIR, ROOT
from models.lstm_ann_model import LSTMANNModel, make_lstm_data


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


def train_and_score(u_train, th_train, u_val, th_val, u_test, th_test, history, hidden_size, epochs):
    Xtrain, Ytrain = make_lstm_data(u_train, th_train, history=history)
    Xval, Yval = make_lstm_data(u_val, th_val, history=history)
    Xtest, Ytest = make_lstm_data(u_test, th_test, history=history)

    model = LSTMANNModel(hidden_size=hidden_size)
    model.fit(Xtrain, Ytrain, Xval, Yval, epochs=epochs)

    prediction = model.predict(Xtest)
    simulation = model.simulate(
        u_test,
        th_test[:SIMULATION_START_SAMPLES],
        history=history,
    )

    skip = SIMULATION_START_SAMPLES

    return {
        "history": history,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "prediction_rmse_rad": rmse(Ytest.reshape(-1), prediction),
        "prediction_rmse_deg": rmse_degrees(Ytest.reshape(-1), prediction),
        "simulation_rmse_rad": rmse(th_test[skip:], simulation[skip:]),
        "simulation_rmse_deg": rmse_degrees(th_test[skip:], simulation[skip:]),
    }


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    benchmark_folder = get_benchmark_folder()
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

    settings = [
        {"history": 15, "hidden_size": 20, "epochs": 80},
        {"history": 10, "hidden_size": 20, "epochs": 80},
        {"history": 15, "hidden_size": 40, "epochs": 120},
        {"history": 15, "hidden_size": 60, "epochs": 120},
    ]

    results = []

    for setting in settings:
        print("Trying setting:", setting)
        result = train_and_score(
            u_train,
            th_train,
            u_val,
            th_val,
            u_test,
            th_test,
            setting["history"],
            setting["hidden_size"],
            setting["epochs"],
        )
        results.append(result)
        print(result)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PLOTS_DIR / "lstm_tuning_results.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print("Saved:", output_file)


if __name__ == "__main__":
    main()

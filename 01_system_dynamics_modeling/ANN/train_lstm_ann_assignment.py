import argparse

import numpy as np
import torch

from config import MODELS_DIR, RAW_DATA_DIR, ROOT, TEST_OUTPUTS_DIR
from models.lstm_ann_model import LSTMANNModel, make_lstm_data


HISTORY = 15
HIDDEN_SIZE = 60
EPOCHS = 120
SIMULATION_START_SAMPLES = 50


def rmse(real, predicted):
    error = real - predicted
    return np.mean(error ** 2) ** 0.5


def print_score(name, real, predicted):
    score_rad = rmse(real, predicted)
    score_deg = score_rad / (2 * np.pi) * 360
    print(name, "RMSE:", round(score_rad, 6), "radians")
    print(name, "RMSE:", round(score_deg, 4), "degrees")


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--simulation-start-samples", type=int, default=SIMULATION_START_SAMPLES)
    args = parser.parse_args()

    benchmark_folder = ROOT / "assignment_files" / "gym-unbalanced-disk" / "disc-benchmark-files"

    if not benchmark_folder.exists():
        benchmark_folder = RAW_DATA_DIR / "gym-unbalanced-disk" / "disc-benchmark-files"

    train_file = benchmark_folder / "training-val-test-data.npz"
    prediction_file = benchmark_folder / "hidden-test-prediction-submission-file.npz"
    simulation_file = benchmark_folder / "hidden-test-simulation-submission-file.npz"

    for file in [train_file, prediction_file, simulation_file]:
        if not file.exists():
            raise FileNotFoundError("Missing required assignment file: " + str(file))

    data = np.load(train_file)
    u = data["u"]
    th = data["th"]

    train_end = int(0.70 * len(th))
    validation_end = int(0.85 * len(th))

    u_train = u[:train_end]
    th_train = th[:train_end]

    u_validation = u[train_end:validation_end]
    th_validation = th[train_end:validation_end]

    u_test = u[validation_end:]
    th_test = th[validation_end:]

    Xtrain, Ytrain = make_lstm_data(u_train, th_train, history=args.history)
    Xval, Yval = make_lstm_data(u_validation, th_validation, history=args.history)
    Xtest, Ytest = make_lstm_data(u_test, th_test, history=args.history)

    model = LSTMANNModel(hidden_size=args.hidden_size)

    print("Training advanced ANN LSTM model")
    print("History:", args.history)
    print("Hidden size:", args.hidden_size)
    print("Training samples:", len(Xtrain))
    print("Validation samples:", len(Xval))
    print("Test samples:", len(Xtest))

    model.fit(Xtrain, Ytrain, Xval, Yval, epochs=args.epochs)

    test_prediction = model.predict(Xtest)
    print_score("LSTM test prediction", Ytest.reshape(-1), test_prediction)

    th_test_simulated = model.simulate(
        u_test,
        th_test[: args.simulation_start_samples],
        history=args.history,
    )
    skip = args.simulation_start_samples
    print_score("LSTM test simulation", th_test[skip:], th_test_simulated[skip:])

    hidden_prediction_data = np.load(prediction_file)
    upast = hidden_prediction_data["upast"]
    thpast = hidden_prediction_data["thpast"]
    prediction_input = np.stack(
        [
            upast[:, 15 - args.history :],
            thpast[:, 15 - args.history :],
        ],
        axis=2,
    )
    thnow = model.predict(prediction_input)

    hidden_simulation_data = np.load(simulation_file)
    u_hidden = hidden_simulation_data["u"]
    th_hidden = hidden_simulation_data["th"]
    th_simulated = model.simulate(
        u_hidden,
        th_hidden[: args.simulation_start_samples],
        history=args.history,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model_file = MODELS_DIR / "ann_lstm_assignment_model.pt"
    prediction_output_file = TEST_OUTPUTS_DIR / "ann_lstm_hidden_prediction_submission.npz"
    simulation_output_file = TEST_OUTPUTS_DIR / "ann_lstm_hidden_simulation_submission.npz"

    torch.save(model, model_file)
    np.savez(prediction_output_file, upast=upast, thpast=thpast, thnow=thnow)
    np.savez(simulation_output_file, u=u_hidden, th=th_simulated)

    print("Saved model:", model_file)
    print("Saved prediction submission:", prediction_output_file)
    print("Saved simulation submission:", simulation_output_file)


if __name__ == "__main__":
    main()

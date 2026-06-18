import argparse

import numpy as np
import torch

from config import MODELS_DIR, PLOTS_DIR, RAW_DATA_DIR, ROOT, TEST_OUTPUTS_DIR
from models.ann_model import ANNModel, make_narx_data


INPUT_DELAY = 15
OUTPUT_DELAY = 15
HIDDEN_NEURONS = 50
EPOCHS = 1500
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-delay", type=int, default=INPUT_DELAY)
    parser.add_argument("--output-delay", type=int, default=OUTPUT_DELAY)
    parser.add_argument("--hidden-neurons", type=int, default=HIDDEN_NEURONS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--simulation-start-samples", type=int, default=SIMULATION_START_SAMPLES)
    args = parser.parse_args()

    benchmark_folder = ROOT / "assignment_files" / "gym-unbalanced-disk" / "disc-benchmark-files"

    if not benchmark_folder.exists():
        benchmark_folder = RAW_DATA_DIR / "gym-unbalanced-disk" / "disc-benchmark-files"

    train_file = benchmark_folder / "training-val-test-data.npz"
    prediction_file = benchmark_folder / "hidden-test-prediction-submission-file.npz"
    simulation_file = benchmark_folder / "hidden-test-simulation-submission-file.npz"

    required_files = [train_file, prediction_file, simulation_file]
    for file in required_files:
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

    x_train, y_train = make_narx_data(u_train, th_train, args.input_delay, args.output_delay)
    x_validation, y_validation = make_narx_data(u_validation, th_validation, args.input_delay, args.output_delay)
    x_test, y_test = make_narx_data(u_test, th_test, args.input_delay, args.output_delay)

    model = ANNModel(
        number_of_inputs=x_train.shape[1],
        hidden_neurons=args.hidden_neurons,
    )

    print("Training ANN NARX model")
    print("Input delay:", args.input_delay)
    print("Output delay:", args.output_delay)
    print("Hidden neurons:", args.hidden_neurons)
    print("Training samples:", len(x_train))
    print("Validation samples:", len(x_validation))
    print("Test samples:", len(x_test))

    model.fit(
        x_train,
        y_train,
        x_validation,
        y_validation,
        epochs=args.epochs,
        patience=40,
    )

    test_prediction = model.predict(x_test)
    print_score("Test prediction", y_test.reshape(-1), test_prediction)

    th_test_simulated = model.simulate(
        u_test,
        th_test[: args.simulation_start_samples],
        args.input_delay,
        args.output_delay,
    )
    skip = args.simulation_start_samples
    print_score("Test simulation", th_test[skip:], th_test_simulated[skip:])

    hidden_prediction_data = np.load(prediction_file)
    upast = hidden_prediction_data["upast"]
    thpast = hidden_prediction_data["thpast"]

    prediction_input = np.concatenate(
        [
            upast[:, 15 - args.input_delay :],
            thpast[:, 15 - args.output_delay :],
        ],
        axis=1,
    )
    thnow = model.predict(prediction_input)

    hidden_simulation_data = np.load(simulation_file)
    u_hidden = hidden_simulation_data["u"]
    th_hidden = hidden_simulation_data["th"]
    th_simulated = model.simulate(
        u_hidden,
        th_hidden[: args.simulation_start_samples],
        args.input_delay,
        args.output_delay,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model_file = MODELS_DIR / "ann_narx_assignment_model.pt"
    torch.save(model, model_file)

    prediction_output_file = TEST_OUTPUTS_DIR / "ann_hidden_prediction_submission.npz"
    simulation_output_file = TEST_OUTPUTS_DIR / "ann_hidden_simulation_submission.npz"

    np.savez(prediction_output_file, upast=upast, thpast=thpast, thnow=thnow)
    np.savez(simulation_output_file, u=u_hidden, th=th_simulated)

    print("Saved model:", model_file)
    print("Saved prediction submission:", prediction_output_file)
    print("Saved simulation submission:", simulation_output_file)


if __name__ == "__main__":
    main()

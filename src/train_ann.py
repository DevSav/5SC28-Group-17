import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import MODELS_DIR, PLOTS_DIR
from src.data import split_train_validation
from src.models.ann_model import ANNModel, make_narx_data


def find_column(data, possible_names):
    """Find a column even if the file uses a slightly different name."""
    lower_names = {name.lower(): name for name in data.columns}

    for name in possible_names:
        if name.lower() in lower_names:
            return lower_names[name.lower()]

    raise ValueError("Could not find one of these columns: " + str(possible_names))


def mean_squared_error(real_y, predicted_y):
    error = real_y - predicted_y
    return np.mean(error ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file with input voltage and angle.")
    parser.add_argument("--input-delay", type=int, default=3)
    parser.add_argument("--output-delay", type=int, default=3)
    parser.add_argument("--hidden-neurons", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    data_path = Path(args.data)
    data = pd.read_csv(data_path)

    input_column = find_column(data, ["u", "input", "voltage"])
    output_column = find_column(data, ["y", "theta", "angle"])

    train_data, validation_data = split_train_validation(data, validation_fraction=0.2)

    x_train, y_train = make_narx_data(
        train_data[input_column],
        train_data[output_column],
        input_delay=args.input_delay,
        output_delay=args.output_delay,
    )
    x_val, y_val = make_narx_data(
        validation_data[input_column],
        validation_data[output_column],
        input_delay=args.input_delay,
        output_delay=args.output_delay,
    )

    model = ANNModel(
        number_of_inputs=x_train.shape[1],
        hidden_neurons=args.hidden_neurons,
    )
    model.fit(x_train, y_train, x_val, y_val, epochs=args.epochs)

    validation_prediction = model.predict(x_val)
    prediction_mse = mean_squared_error(y_val.reshape(-1), validation_prediction)

    first_index = max(args.input_delay, args.output_delay)
    y_start = validation_data[output_column].values[:first_index]
    y_simulated = model.simulate(
        validation_data[input_column].values,
        y_start,
        input_delay=args.input_delay,
        output_delay=args.output_delay,
    )
    simulation_mse = mean_squared_error(validation_data[output_column].values, y_simulated)

    print("Prediction MSE:", prediction_mse)
    print("Simulation MSE:", simulation_mse)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model_file = MODELS_DIR / "ann_narx_model.pt"
    torch.save(model, model_file)

    plt.figure(figsize=(10, 4))
    plt.plot(validation_data[output_column].values, label="measured angle")
    plt.plot(y_simulated, label="ANN simulation")
    plt.xlabel("sample")
    plt.ylabel("angle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ann_simulation_validation.png")

    print("Saved model to", model_file)
    print("Saved plot to", PLOTS_DIR / "ann_simulation_validation.png")


if __name__ == "__main__":
    main()

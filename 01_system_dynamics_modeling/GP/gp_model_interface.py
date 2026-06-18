import numpy as np
import joblib


class GPSparseDeltaNARX:
    def __init__(self, model_file="best_gpy_sparse_delta_sim_model.joblib"):
        bundle = joblib.load(model_file)

        self.model = bundle["model"]
        self.x_scaler = bundle["x_scaler"]
        self.y_scaler = bundle["y_scaler"]
        self.na = bundle["na"]
        self.nb = bundle["nb"]

    def predict_delta(self, u_past, theta_past):
        """
        Predict delta_theta(k) = theta(k) - theta(k-1).

        u_past:
            array-like, at least nb samples, ordered from old to new

        theta_past:
            array-like, at least na samples, ordered from old to new
        """
        u_past = np.asarray(u_past).reshape(-1)
        theta_past = np.asarray(theta_past).reshape(-1)

        if len(u_past) < self.nb:
            raise ValueError(f"u_past must contain at least {self.nb} samples.")

        if len(theta_past) < self.na:
            raise ValueError(f"theta_past must contain at least {self.na} samples.")

        x_raw = np.concatenate([
            u_past[-self.nb:],
            theta_past[-self.na:]
        ]).reshape(1, -1)

        x_scaled = self.x_scaler.transform(x_raw)

        delta_scaled, delta_var_scaled = self.model.predict(x_scaled)

        delta = self.y_scaler.inverse_transform(delta_scaled)[0, 0]

        # Approximate standard deviation in original output scale
        delta_std = np.sqrt(delta_var_scaled[0, 0]) * self.y_scaler.scale_[0]

        return delta, delta_std

    def predict_next_theta(self, u_past, theta_past):
        """
        Predict theta(k) using:
            theta_hat(k) = theta(k-1) + delta_theta_hat(k)
        """
        theta_past = np.asarray(theta_past).reshape(-1)

        delta, delta_std = self.predict_delta(u_past, theta_past)
        theta_next = theta_past[-1] + delta

        return theta_next, delta_std

    def simulate(self, u_sequence, theta_initial):
        """
        Free-run simulation.

        u_sequence:
            full input sequence

        theta_initial:
            initial measured angle sequence, at least max(na, nb) samples.
            After this, the model feeds back its own predictions.
        """
        u_sequence = np.asarray(u_sequence).reshape(-1)
        theta_initial = np.asarray(theta_initial).reshape(-1)

        skip = len(theta_initial)

        if skip < max(self.na, self.nb):
            raise ValueError(f"theta_initial must contain at least {max(self.na, self.nb)} samples.")

        if len(u_sequence) < skip:
            raise ValueError("u_sequence must be at least as long as theta_initial.")

        ysim = list(theta_initial)

        for k in range(skip, len(u_sequence)):
            u_past = u_sequence[k - self.nb:k]
            theta_past = np.array(ysim[-self.na:])

            theta_next, _ = self.predict_next_theta(u_past, theta_past)
            ysim.append(theta_next)

        return np.asarray(ysim)
import numpy as np
import warnings
import time
import csv
import joblib
import matplotlib.pyplot as plt

import GPy

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore")


# =====================================================
# 1. Load data
# =====================================================
data = np.load("training-val-test-data.npz")
th = data["th"].reshape(-1)
u = data["u"].reshape(-1)

print("Loaded data")
print("th shape:", th.shape)
print("u shape:", u.shape)


# =====================================================
# 2. Delta-output NARX data
# X(k) = [u(k-nb), ..., u(k-1), th(k-na), ..., th(k-1)]
# Target = delta_th(k) = th(k) - th(k-1)
# =====================================================
def make_delta_training_data(ulist, ylist, na, nb):
    Xdata = []
    Ydelta = []
    Ytheta = []

    for k in range(max(na, nb), len(ylist)):
        upast = ulist[k - nb:k]
        ypast = ylist[k - na:k]

        delta = ylist[k] - ylist[k - 1]

        Xdata.append(np.concatenate([upast, ypast]))
        Ydelta.append(delta)
        Ytheta.append(ylist[k])

    return np.asarray(Xdata), np.asarray(Ydelta).reshape(-1, 1), np.asarray(Ytheta)


# =====================================================
# 3. Free-run simulation for delta-output GP
# =====================================================
def simulate_delta_gpy_sparse(model, x_scaler, y_scaler, ulist, ylist, na, nb, skip=50):
    assert skip >= max(na, nb), "skip must be >= max(na, nb)"

    upast = list(ulist[skip - nb:skip])
    ypast = list(ylist[skip - na:skip])

    ysim = list(ylist[:skip])

    for k in range(skip, len(ylist)):
        x_raw = np.concatenate([upast, ypast]).reshape(1, -1)
        x_scaled = x_scaler.transform(x_raw)

        delta_scaled, _ = model.predict(x_scaled)
        delta_pred = y_scaler.inverse_transform(delta_scaled)[0, 0]

        ypred = ypast[-1] + delta_pred
        ysim.append(ypred)

        upast.append(ulist[k])
        upast.pop(0)

        ypast.append(ypred)
        ypast.pop(0)

    return np.asarray(ysim)


# =====================================================
# 4. Inducing point initialization
# =====================================================
def initialize_inducing_points(Xtrain, M, method="kmeans"):
    if method == "uniform":
        idx = np.linspace(0, len(Xtrain) - 1, M).astype(int)
        return Xtrain[idx].copy()

    if method == "kmeans":
        print(f"Initializing {M} inducing points by MiniBatchKMeans...")
        kmeans = MiniBatchKMeans(
            n_clusters=M,
            batch_size=2048,
            max_iter=100,
            n_init=3,
            random_state=0
        )
        kmeans.fit(Xtrain)
        return kmeans.cluster_centers_.copy()

    raise ValueError("Unknown inducing point initialization method.")


# =====================================================
# 5. Train / validation split
# =====================================================
split = 0.70
split_index = int(len(th) * split)

u_train_full = u[:split_index]
th_train_full = th[:split_index]

u_val_full = u[split_index:]
th_val_full = th[split_index:]


# =====================================================
# 6. Sparse GP candidates
# Start around the current best delta model: na=10, nb=8
# M = number of inducing points
# =====================================================
candidate_configs = [
    {"na": 10, "nb": 8, "M": 150, "max_iters": 600, "init": "kmeans"},
    {"na": 10, "nb": 8, "M": 250, "max_iters": 600, "init": "kmeans"},
]

# Later, if this improves or is close, try:
# candidate_configs += [
#     {"na": 12, "nb": 8,  "M": 250, "max_iters": 600, "init": "kmeans"},
#     {"na": 10, "nb": 10, "M": 250, "max_iters": 600, "init": "kmeans"},
#     {"na": 12, "nb": 10, "M": 250, "max_iters": 600, "init": "kmeans"},
# ]

skip = 50
results = []

best_bundle = None
best_sim_rmse = np.inf


# =====================================================
# 7. Train and evaluate sparse GP candidates
# =====================================================
for cfg in candidate_configs:
    na = cfg["na"]
    nb = cfg["nb"]
    M = cfg["M"]
    max_iters = cfg["max_iters"]
    init_method = cfg["init"]

    print("\n====================================================")
    print(f"GPy Sparse delta-output GP: na={na}, nb={nb}, M={M}")
    print("====================================================")

    start_time = time.time()

    Xtrain_raw, Ydelta_train, Ytheta_train = make_delta_training_data(
        u_train_full, th_train_full, na, nb
    )
    Xval_raw, Ydelta_val, Ytheta_val = make_delta_training_data(
        u_val_full, th_val_full, na, nb
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtrain = x_scaler.fit_transform(Xtrain_raw)
    Xval = x_scaler.transform(Xval_raw)

    Ytrain_scaled = y_scaler.fit_transform(Ydelta_train)
    Yval_scaled = y_scaler.transform(Ydelta_val)

    input_dim = Xtrain.shape[1]

    print("Full training samples:", Xtrain.shape[0])
    print("Validation samples:", Xval.shape[0])
    print("Input dimension:", input_dim)
    print("Inducing points M:", M)

    Z = initialize_inducing_points(Xtrain, M, method=init_method)

    # =====================================================
    # Matern 5/2 kernel = Matern nu=2.5
    # In GPy, Gaussian noise is represented by the likelihood.
    # ARD=True gives one lengthscale per regressor.
    # =====================================================
    kernel = GPy.kern.Matern52(
        input_dim=input_dim,
        variance=1.0,
        lengthscale=np.ones(input_dim),
        ARD=True
    )

    model = GPy.models.SparseGPRegression(
        Xtrain,
        Ytrain_scaled,
        kernel=kernel,
        Z=Z
    )

    # Noise variance in scaled delta-output units
    model.Gaussian_noise.variance = 1e-3
    model.Gaussian_noise.variance.constrain_bounded(1e-6, 1e-1)

    print("\nInitial sparse GP model:")
    print(model)

    print("\nOptimizing sparse GP...")
    model.optimize(messages=True, max_iters=max_iters)

    print("\nOptimized sparse GP model:")
    print(model)

    # =====================================================
    # 7.1 One-step theta prediction
    # Predict delta, then theta_hat(k) = theta(k-1) + delta_hat(k)
    # =====================================================
    Ydelta_val_pred_scaled, _ = model.predict(Xval)
    Ydelta_val_pred = y_scaler.inverse_transform(Ydelta_val_pred_scaled).reshape(-1)

    theta_last_val = Xval_raw[:, nb + na - 1]
    Ytheta_val_pred = theta_last_val + Ydelta_val_pred

    rmse_pred = mean_squared_error(Ytheta_val, Ytheta_val_pred) ** 0.5
    rmse_pred_deg = rmse_pred / (2 * np.pi) * 360
    nrms_pred = rmse_pred / np.std(Ytheta_val) * 100

    rmse_delta = mean_squared_error(Ydelta_val.reshape(-1), Ydelta_val_pred) ** 0.5

    # =====================================================
    # 7.2 Free-run validation simulation
    # =====================================================
    th_val_sim = simulate_delta_gpy_sparse(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        ulist=u_val_full,
        ylist=th_val_full,
        na=na,
        nb=nb,
        skip=skip
    )

    rmse_sim = mean_squared_error(th_val_full[skip:], th_val_sim[skip:]) ** 0.5
    rmse_sim_deg = rmse_sim / (2 * np.pi) * 360
    nrms_sim = rmse_sim / np.std(th_val_full[skip:]) * 100

    elapsed = time.time() - start_time

    print("\nResults:")
    print(f"One-step theta prediction RMSE = {rmse_pred:.6f} rad = {rmse_pred_deg:.4f} deg")
    print(f"Delta prediction RMSE          = {rmse_delta:.6f} rad")
    print(f"Free-run simulation RMSE       = {rmse_sim:.6f} rad = {rmse_sim_deg:.4f} deg")
    print(f"Simulation NRMS                = {nrms_sim:.2f} %")
    print(f"Elapsed time                   = {elapsed:.1f} sec")

    result = {
        "model_type": "gpy_sparse_delta_gp_narx",
        "na": na,
        "nb": nb,
        "M": M,
        "init_method": init_method,
        "input_dim": input_dim,
        "prediction_rmse_rad": rmse_pred,
        "prediction_rmse_deg": rmse_pred_deg,
        "prediction_nrms_percent": nrms_pred,
        "delta_rmse_rad": rmse_delta,
        "simulation_rmse_rad": rmse_sim,
        "simulation_rmse_deg": rmse_sim_deg,
        "simulation_nrms_percent": nrms_sim,
        "elapsed_sec": elapsed,
        "kernel": str(model.kern),
        "noise_variance": float(model.Gaussian_noise.variance.values[0])
    }

    results.append(result)

    model_name = f"gpy_sparse_delta_na{na}_nb{nb}_M{M}.joblib"
    joblib.dump(
        {
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "na": na,
            "nb": nb,
            "M": M,
            "split": split,
            "split_index": split_index,
            "prediction_rmse_rad": rmse_pred,
            "prediction_rmse_deg": rmse_pred_deg,
            "simulation_rmse_rad": rmse_sim,
            "simulation_rmse_deg": rmse_sim_deg,
            "simulation_nrms_percent": nrms_sim,
            "model_type": "gpy_sparse_delta_gp_narx",
            "kernel": str(model.kern),
            "noise_variance": float(model.Gaussian_noise.variance.values[0])
        },
        model_name
    )
    print("Saved model:", model_name)

    if rmse_sim < best_sim_rmse:
        best_sim_rmse = rmse_sim
        best_bundle = {
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "na": na,
            "nb": nb,
            "M": M,
            "split": split,
            "split_index": split_index,
            "prediction_rmse_rad": rmse_pred,
            "prediction_rmse_deg": rmse_pred_deg,
            "prediction_nrms_percent": nrms_pred,
            "delta_rmse_rad": rmse_delta,
            "simulation_rmse_rad": rmse_sim,
            "simulation_rmse_deg": rmse_sim_deg,
            "simulation_nrms_percent": nrms_sim,
            "model_type": "gpy_sparse_delta_gp_narx",
            "kernel": str(model.kern),
            "noise_variance": float(model.Gaussian_noise.variance.values[0]),
            "th_val_sim": th_val_sim
        }


# =====================================================
# 8. Summary sorted by simulation RMSE
# =====================================================
results_by_sim = sorted(results, key=lambda r: r["simulation_rmse_rad"])

print("\n\n====================================================")
print("GPy sparse delta-output candidates sorted by simulation RMSE")
print("====================================================")

for i, r in enumerate(results_by_sim):
    print(
        f"{i+1}. na={r['na']:2d}, nb={r['nb']:2d}, M={r['M']:4d} | "
        f"Pred RMSE={r['prediction_rmse_rad']:.6f} rad = {r['prediction_rmse_deg']:.4f} deg | "
        f"Sim RMSE={r['simulation_rmse_rad']:.6f} rad = {r['simulation_rmse_deg']:.4f} deg | "
        f"time={r['elapsed_sec']:.1f}s"
    )
    print("   kernel:", r["kernel"])
    print("   noise variance:", r["noise_variance"])
    print()


# =====================================================
# 9. Save CSV
# =====================================================
csv_filename = "gpy_sparse_delta_sim_tuning_results.csv"

if len(results_by_sim) > 0:
    fieldnames = list(results_by_sim[0].keys())

    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_by_sim)

    print("Saved sparse GP tuning results to:", csv_filename)


# =====================================================
# 10. Save best sparse model and plots
# =====================================================
if best_bundle is not None:
    th_val_sim_best = best_bundle.pop("th_val_sim")

    joblib.dump(best_bundle, "best_gpy_sparse_delta_sim_model.joblib")

    print("\nBest GPy sparse delta-output model saved to:")
    print("best_gpy_sparse_delta_sim_model.joblib")

    print("\nBest sparse delta-output model summary:")
    print("na =", best_bundle["na"])
    print("nb =", best_bundle["nb"])
    print("M =", best_bundle["M"])
    print(f"Prediction RMSE = {best_bundle['prediction_rmse_rad']:.6f} rad = {best_bundle['prediction_rmse_deg']:.4f} deg")
    print(f"Simulation RMSE = {best_bundle['simulation_rmse_rad']:.6f} rad = {best_bundle['simulation_rmse_deg']:.4f} deg")
    print(f"Simulation NRMS = {best_bundle['simulation_nrms_percent']:.2f} %")
    print("Kernel:", best_bundle["kernel"])
    print("Noise variance:", best_bundle["noise_variance"])

    n_plot = min(2000, len(th_val_full))

    plt.figure(figsize=(11, 4))
    plt.plot(th_val_full[:n_plot], label="measured validation output")
    plt.plot(th_val_sim_best[:n_plot], label="best GPy sparse delta-output GP simulation")
    plt.axvline(skip, linestyle="--", label="simulation start")
    plt.grid(True)
    plt.xlabel("sample")
    plt.ylabel("angle [rad]")
    plt.title("Best GPy sparse delta-output GP free-run simulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_gpy_sparse_delta_validation_simulation.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.plot(th_val_full[:n_plot] - th_val_sim_best[:n_plot])
    plt.axvline(skip, linestyle="--", label="simulation start")
    plt.grid(True)
    plt.xlabel("sample")
    plt.ylabel("simulation error [rad]")
    plt.title("Best GPy sparse delta-output GP simulation error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_gpy_sparse_delta_validation_simulation_error.png", dpi=300)
    plt.close()

    print("Saved plots:")
    print("best_gpy_sparse_delta_validation_simulation.png")
    print("best_gpy_sparse_delta_validation_simulation_error.png")
import numpy as np
import joblib


# =====================================================
# 1. Load final selected GPy sparse delta-output model
# =====================================================
bundle = joblib.load("best_gpy_sparse_delta_sim_model.joblib")

model = bundle["model"]
x_scaler = bundle["x_scaler"]
y_scaler = bundle["y_scaler"]
na = bundle["na"]
nb = bundle["nb"]

print("Loaded final GPy sparse delta-output GP-NARX model")
print("na =", na)
print("nb =", nb)
print("M =", bundle["M"])
print("Prediction RMSE =", bundle["prediction_rmse_rad"])
print("Simulation RMSE =", bundle["simulation_rmse_rad"])
print("Model type:", bundle["model_type"])
print("Kernel:", bundle["kernel"])


# =====================================================
# 2. Hidden prediction task
# Given:
#   upast:   u[k-15], ..., u[k-1]
#   thpast: th[k-15], ..., th[k-1]
#
# Our delta model predicts:
#   delta_th[k] = th[k] - th[k-1]
#
# Therefore:
#   thnow_pred = thpast_last + delta_pred
# =====================================================
pred_data = np.load("hidden-test-prediction-submission-file.npz")

upast_test = pred_data["upast"]
thpast_test = pred_data["thpast"]

print("\nHidden prediction data:")
print("upast_test shape:", upast_test.shape)
print("thpast_test shape:", thpast_test.shape)

Xpred_raw = np.concatenate(
    [
        upast_test[:, -nb:],
        thpast_test[:, -na:]
    ],
    axis=1
)

Xpred = x_scaler.transform(Xpred_raw)

delta_pred_scaled, _ = model.predict(Xpred)
delta_pred = y_scaler.inverse_transform(delta_pred_scaled).reshape(-1)

th_last = thpast_test[:, -1]
thnow_pred = th_last + delta_pred

assert len(thnow_pred) == len(upast_test), "number of prediction samples changed!"

prediction_filename = "gp_sparse_delta_hidden_prediction_submission.npz"

np.savez(
    prediction_filename,
    upast=upast_test,
    thpast=thpast_test,
    thnow=thnow_pred
)

print("\nSaved hidden prediction submission:")
print(prediction_filename)


# =====================================================
# 3. Hidden simulation task
# Given:
#   u: full input sequence
#   th: first 50 true outputs, rest zeros
#
# We keep first 50 true outputs, then simulate freely:
#   delta_hat[k] = GP(x[k])
#   th_hat[k] = th_hat[k-1] + delta_hat[k]
# =====================================================
sim_data = np.load("hidden-test-simulation-submission-file.npz")

u_test = sim_data["u"].reshape(-1)
th_test = sim_data["th"].reshape(-1)

print("\nHidden simulation data:")
print("u_test shape:", u_test.shape)
print("th_test shape:", th_test.shape)


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


skip = 50

th_test_sim = simulate_delta_gpy_sparse(
    model=model,
    x_scaler=x_scaler,
    y_scaler=y_scaler,
    ulist=u_test,
    ylist=th_test,
    na=na,
    nb=nb,
    skip=skip
)

assert len(th_test_sim) == len(th_test), "number of simulation samples changed!"

simulation_filename = "gp_sparse_delta_hidden_simulation_submission.npz"

np.savez(
    simulation_filename,
    th=th_test_sim,
    u=u_test
)

print("\nSaved hidden simulation submission:")
print(simulation_filename)

print("\nDone.")
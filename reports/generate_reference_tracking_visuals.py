import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "03_single_policy_reference_tracking" / "reference_tracking"
MODEL_FILE = REF_DIR / "results_dqn_reference_tracking" / "ref_dqn_model_best.pth"
OUTPUT_DIR = ROOT / "reports" / "figures"

sys.path.insert(0, str(REF_DIR))
import main_train_dqn_reference_tracking as ref_dqn  # noqa: E402


RED_TO_BLUE = LinearSegmentedColormap.from_list(
    "red_to_blue",
    ["#d62728", "#2f6fdd"],
)


def get_rollout():
    agent = ref_dqn.DQNAgent()
    agent.load(MODEL_FILE)

    env = ref_dqn.gym.make(ref_dqn.ENV_ID, dt=ref_dqn.DT, umax=ref_dqn.UMAX)
    result, metrics, _ = ref_dqn.rollout_episode(
        env=env,
        agent=agent,
        epsilon=0.0,
        seed=200,
        train=False,
    )
    env.close()
    return result, metrics


def closeness_to_reference(error):
    return 1.0 - min(abs(error) / np.pi, 1.0)


def disk_xy(theta, length=0.85):
    x = length * np.sin(theta)
    y = -length * np.cos(theta)
    return x, y


def draw_disk_axis(axis, theta, theta_ref, error, title):
    close = closeness_to_reference(error)
    color = RED_TO_BLUE(close)
    x, y = disk_xy(theta)
    xr, yr = disk_xy(theta_ref, length=0.92)

    axis.plot([0, 0], [0, 0.9], "--", color="0.75", linewidth=1.0)
    axis.plot([0, xr], [0, yr], ":", color="black", linewidth=2.0)
    axis.plot([0, x], [0, y], color=color, linewidth=4)
    axis.scatter([0], [0], s=45, color="black", zorder=3)
    axis.scatter([xr], [yr], s=80, color="white", edgecolor="black", linewidth=1.2, zorder=4)
    axis.scatter([x], [y], s=220, color=color, edgecolor="black", linewidth=0.8, zorder=5)

    axis.set_title(title, fontsize=9)
    axis.set_aspect("equal")
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.axis("off")


def make_frame_strip(result, metrics):
    theta = result["theta"]
    theta_ref = result["theta_ref"]
    error = result["error"]
    frame_ids = [0, 40, 80, 120, 180, 240, len(theta) - 1]

    fig, axes = plt.subplots(1, len(frame_ids), figsize=(12, 2.2))
    for axis, idx in zip(axes, frame_ids):
        time = idx * ref_dqn.DT
        draw_disk_axis(
            axis,
            theta[idx],
            theta_ref[idx],
            error[idx],
            f"t = {time:.1f} s",
        )

    fig.suptitle(
        f"Reference-tracking DQN rollout, final error = {metrics['final_error']:.3f} rad",
        fontsize=12,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "reference_tracking_frame_strip.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def make_colored_tracking_plot(result):
    theta = result["theta"]
    theta_ref = result["theta_ref"]
    error = result["error"]
    time = np.arange(len(theta)) * ref_dqn.DT

    points = np.array([time, theta]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    close = np.array([closeness_to_reference(value) for value in error[:-1]])

    fig, axis = plt.subplots(figsize=(8, 3.5))
    line_collection = LineCollection(segments, cmap=RED_TO_BLUE, linewidth=2.2)
    line_collection.set_array(close)
    axis.add_collection(line_collection)

    axis.plot(time, theta_ref, "--", color="black", linewidth=1.5, label="reference")
    axis.axhline(np.pi, linestyle=":", color="0.4", linewidth=1, label="upright")
    axis.set_xlim(time.min(), time.max())
    axis.set_ylim(min(theta.min(), theta_ref.min()) - 0.25, max(theta.max(), theta_ref.max()) + 0.25)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("angle theta (rad)")
    axis.set_title("Reference-tracking DQN trajectory")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    colorbar = fig.colorbar(line_collection, ax=axis)
    colorbar.set_label("closeness to reference")

    out = OUTPUT_DIR / "reference_tracking_colored_trajectory.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved:", out)


def make_error_and_input_plot(result):
    time = np.arange(len(result["theta"])) * ref_dqn.DT

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)

    axes[0].plot(time, result["error"], color="#2f6fdd", linewidth=1.8)
    axes[0].axhline(0.0, linestyle="--", color="black", linewidth=1)
    axes[0].set_ylabel("error (rad)")
    axes[0].set_title("Reference-tracking error and motor voltage")
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, result["u"], color="#d62728", linewidth=1.8)
    axes[1].axhline(ref_dqn.UMAX, linestyle="--", color="black", linewidth=1)
    axes[1].axhline(-ref_dqn.UMAX, linestyle="--", color="black", linewidth=1)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("voltage (V)")
    axes[1].grid(alpha=0.25)

    out = OUTPUT_DIR / "reference_tracking_error_input.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved:", out)


def make_gif(result):
    theta = result["theta"]
    theta_ref = result["theta_ref"]
    error = result["error"]
    frame_ids = np.arange(0, len(theta), 4)

    fig, axis = plt.subplots(figsize=(4, 4))

    def update(frame_number):
        axis.clear()
        idx = int(frame_ids[frame_number])
        draw_disk_axis(
            axis,
            theta[idx],
            theta_ref[idx],
            error[idx],
            f"tracking DQN, t = {idx * ref_dqn.DT:.1f} s",
        )

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=70)
    out = OUTPUT_DIR / "reference_tracking_animation.gif"
    animation.save(out, writer=PillowWriter(fps=14))
    plt.close(fig)
    print("Saved:", out)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result, metrics = get_rollout()

    np.savez(
        OUTPUT_DIR / "reference_tracking_rollout.npz",
        theta=result["theta"],
        omega=result["omega"],
        theta_ref=result["theta_ref"],
        ref_offset=result["ref_offset"],
        u=result["u"],
        error=result["error"],
        reward=result["reward"],
    )

    make_frame_strip(result, metrics)
    make_colored_tracking_plot(result)
    make_error_and_input_plot(result)
    make_gif(result)


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
DQN_DIR = ROOT / "02_policy_learning_swingup" / "01_q_learning_swingup"
MODEL_FILE = DQN_DIR / "results_dqn_swingup" / "dqn_model_best.pth"
OUTPUT_DIR = ROOT / "reports" / "figures"

sys.path.insert(0, str(DQN_DIR))
import main_train_dpn_swingup as dqn  # noqa: E402


RED_TO_BLUE = LinearSegmentedColormap.from_list(
    "red_to_blue",
    ["#d62728", "#2f6fdd"],
)


def get_rollout():
    agent = dqn.DQNAgent()
    agent.load(MODEL_FILE)

    env = dqn.gym.make(dqn.ENV_ID, dt=dqn.DT, umax=dqn.UMAX)
    result, metrics, _ = dqn.rollout_episode(
        env=env,
        agent=agent,
        epsilon=0.0,
        seed=100,
        train=False,
    )
    env.close()
    return result, metrics


def closeness_to_upright(theta):
    error = dqn.wrap_to_pi(theta - dqn.THETA_TARGET)
    return 1.0 - min(abs(error) / np.pi, 1.0)


def disk_xy(theta, length=0.85):
    x = length * np.sin(theta)
    y = -length * np.cos(theta)
    return x, y


def draw_disk_axis(axis, theta, title):
    close = closeness_to_upright(theta)
    color = RED_TO_BLUE(close)
    x, y = disk_xy(theta)

    axis.plot([0, 0], [0, 0.9], "--", color="0.75", linewidth=1.0)
    axis.plot([0, x], [0, y], color=color, linewidth=4)
    axis.scatter([0], [0], s=45, color="black", zorder=3)
    axis.scatter([x], [y], s=220, color=color, edgecolor="black", linewidth=0.8, zorder=4)

    axis.set_title(title, fontsize=9)
    axis.set_aspect("equal")
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.axis("off")


def make_frame_strip(result, metrics):
    theta = result["theta"]
    frame_ids = [0, 40, 80, 120, 180, 240, len(theta) - 1]

    fig, axes = plt.subplots(1, len(frame_ids), figsize=(12, 2.2))
    for axis, idx in zip(axes, frame_ids):
        time = idx * dqn.DT
        draw_disk_axis(axis, theta[idx], f"t = {time:.1f} s")

    fig.suptitle(
        f"DQN swing-up rollout, final error = {metrics['final_error']:.3f} rad",
        fontsize=12,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "control_dqn_swingup_frame_strip.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def make_phase_colored_plot(result):
    theta = result["theta"]
    time = np.arange(len(theta)) * dqn.DT
    points = np.array([time, theta]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    close = np.array([closeness_to_upright(value) for value in theta[:-1]])

    fig, axis = plt.subplots(figsize=(8, 3.5))
    line_collection = LineCollection(segments, cmap=RED_TO_BLUE, linewidth=2.2)
    line_collection.set_array(close)
    axis.add_collection(line_collection)

    axis.axhline(np.pi, linestyle="--", color="black", linewidth=1, label="upright")
    axis.axhline(-np.pi, linestyle="--", color="black", linewidth=1)
    axis.axhline(0.0, linestyle=":", color="0.4", linewidth=1, label="bottom")
    axis.set_xlim(time.min(), time.max())
    axis.set_ylim(min(theta.min(), -3.4), max(theta.max(), 3.4))
    axis.set_xlabel("time (s)")
    axis.set_ylabel("angle theta (rad)")
    axis.set_title("DQN swing-up trajectory, colored by closeness to upright")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    colorbar = fig.colorbar(line_collection, ax=axis)
    colorbar.set_label("closeness to upright")

    out = OUTPUT_DIR / "control_dqn_swingup_colored_trajectory.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved:", out)


def make_gif(result):
    theta = result["theta"]
    frame_ids = np.arange(0, len(theta), 4)

    fig, axis = plt.subplots(figsize=(4, 4))

    def update(frame_number):
        axis.clear()
        idx = int(frame_ids[frame_number])
        draw_disk_axis(axis, theta[idx], f"DQN swing-up, t = {idx * dqn.DT:.1f} s")

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=70)
    out = OUTPUT_DIR / "control_dqn_swingup_animation.gif"
    animation.save(out, writer=PillowWriter(fps=14))
    plt.close(fig)
    print("Saved:", out)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result, metrics = get_rollout()
    np.savez(
        OUTPUT_DIR / "control_dqn_swingup_rollout.npz",
        theta=result["theta"],
        omega=result["omega"],
        u=result["u"],
        error=result["error"],
        reward=result["reward"],
    )
    make_frame_strip(result, metrics)
    make_phase_colored_plot(result)
    make_gif(result)


if __name__ == "__main__":
    main()

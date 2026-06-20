import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = (
    ROOT
    / "02_policy_learning_swingup"
    / "01_q_learning_swingup"
    / "results_dqn_swingup"
    / "dqn_training_log.json"
)
OUTPUT = ROOT / "reports" / "figures" / "dqn_training_progress_pendulums.png"
GIF_OUTPUT = ROOT / "reports" / "figures" / "dqn_training_progress_pendulums.gif"

THETA_TARGET = np.pi
RED_TO_BLUE = LinearSegmentedColormap.from_list(
    "red_to_blue",
    ["#d62728", "#2f6fdd"],
)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def disk_xy(theta, length=0.82):
    x = length * np.sin(theta)
    y = -length * np.cos(theta)
    return x, y


def draw_pendulum(axis, theta, upright_ratio, title, subtitle):
    color = RED_TO_BLUE(upright_ratio)
    x, y = disk_xy(theta)

    axis.plot([0, 0], [0, 0.9], "--", color="0.75", linewidth=1.0)
    axis.plot([0, x], [0, y], color=color, linewidth=4)
    axis.scatter([0], [0], s=40, color="black", zorder=3)
    axis.scatter([x], [y], s=220, color=color, edgecolor="black", linewidth=0.8, zorder=4)

    axis.set_title(title + "\n" + subtitle, fontsize=8)
    axis.set_aspect("equal")
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)
    axis.axis("off")


def closest_log_item(log, episode):
    return min(log, key=lambda item: abs(item["episode"] - episode))


def main():
    log = json.loads(LOG_FILE.read_text())
    best = max(log, key=lambda item: item["score"])

    wanted_episodes = [0, 100, 300, 600, 900, 1200, best["episode"]]
    items = [closest_log_item(log, episode) for episode in wanted_episodes]

    fig, axes = plt.subplots(1, len(items), figsize=(13, 2.7))

    for axis, item in zip(axes, items):
        # final_error = wrap(theta - pi), so theta can be reconstructed for display
        theta = wrap_to_pi(THETA_TARGET + item["final_error"])
        title = f"episode {item['episode']}"
        subtitle = f"upright {item['upright_ratio']:.2f}"
        draw_pendulum(axis, theta, item["upright_ratio"], title, subtitle)

    fig.suptitle(
        "DQN training progress: final disk position and upright ratio",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.02,
        "Red means poor swing-up performance; blue means the episode spent more time near upright.",
        ha="center",
        fontsize=9,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.88])
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUTPUT)
    print("Best episode:", best["episode"])

    # Animated version for slides. This shows training progress over time,
    # using the final state and upright ratio stored for each episode.
    gif_items = log[::20]
    if gif_items[-1]["episode"] != best["episode"]:
        gif_items.append(best)

    fig, axis = plt.subplots(figsize=(4.8, 4.2))

    def update(frame_number):
        axis.clear()
        item = gif_items[frame_number]
        theta = wrap_to_pi(THETA_TARGET + item["final_error"])
        title = f"episode {item['episode']}"
        subtitle = (
            f"upright ratio = {item['upright_ratio']:.2f}\n"
            f"success = {item['success']}"
        )
        draw_pendulum(axis, theta, item["upright_ratio"], title, subtitle)

    animation = FuncAnimation(fig, update, frames=len(gif_items), interval=120)
    animation.save(GIF_OUTPUT, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("Saved:", GIF_OUTPUT)


if __name__ == "__main__":
    main()

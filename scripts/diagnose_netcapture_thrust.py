"""
Diagnostic script: apply constant thrust to all 4 drones in the NetCapture scene
and record whether the multi-drone net system can actually fly.

This bypasses RL entirely to isolate the action → force → physics chain.

Usage:
    python scripts/diagnose_netcapture_thrust.py

Output:
    results_video/diagnose_thrust/
        action_-1.0.mp4   (zero thrust, expected: free fall)
        action_-0.14.mp4  (hover thrust, expected: near-hover)
        action_0.0.mp4    (50% thrust, expected: slow ascent)
        action_0.5.mp4    (75% thrust, expected: moderate climb)
        action_1.0.mp4    (100% thrust, expected: max climb)
        *_data.npz        (net_centre z and drone z time series)
        comparison.png    (z-position vs time for all test cases)
"""

import ctypes
ctypes.CDLL("libX11.so.6").XInitThreads()

import os
import sys
import time
import numpy as np
import torch

from omegaconf import OmegaConf
from omni_drones import init_simulation_app

# ── Project path setup ──────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_use_pbd_rope = False  # ← 设为 False 使用 D6 关节绳索, True 使用 PBD 粒子绳索
_output_dir = os.path.join(
    _project_root,
    "results_video",
    "diagnose_thrust_pbd" if _use_pbd_rope else "diagnose_thrust_d6",
)
os.makedirs(_output_dir, exist_ok=True)


def build_cfg(headless: bool = True, num_envs: int = 4) -> OmegaConf:
    """Build a NetCapture config without requiring Hydra CLI overrides.

    Manually merges base env, base sim, and task configs following the
    same pattern as ``scripts/test_net_capture_scene.py``.
    """
    base_env = OmegaConf.load(os.path.join(_project_root, "cfg", "base", "env_base.yaml"))
    base_sim = OmegaConf.load(os.path.join(_project_root, "cfg", "base", "sim_base.yaml"))
    task_cfg = OmegaConf.load(
        os.path.join(_project_root, "cfg", "task", "NetCapture", "NetCapture.yaml")
    )

    # Merge: base values are defaults, task_cfg overrides them
    merged = OmegaConf.merge(base_env, base_sim, task_cfg)
    merged.env.num_envs = num_envs
    merged.use_pbd_rope = _use_pbd_rope  # 覆盖绳索类型

    # IsaacEnv reads cfg.sim / cfg.env at root level
    # (Hydra resolves these via ${task.sim} / ${task.env} interpolation)
    return OmegaConf.create({
        "task": merged,
        "sim": merged.sim,
        "env": merged.env,
        "headless": headless,
        "viewer": {
            "resolution": [1280, 720],
            "eye": [8.0, 0.0, 6.0],
            "lookat": [0.0, 0.0, 1.0],
        },
    })


def main():
    # ── Config ───────────────────────────────────────────────────────────────
    cfg = build_cfg(headless=True, num_envs=4)

    # ── Simulation app + environment ──────────────────────────────────────────
    init_simulation_app(cfg)

    from omni_drones.envs.net_capture import NetCapture
    env = NetCapture(cfg, headless=cfg.headless)

    # Enable off-screen rendering for video capture
    env.enable_render(True)

    num_envs = cfg.env.num_envs
    num_drones = cfg.task.num_drones
    num_rotors = 4  # Hummingbird

    max_steps = cfg.env.max_episode_length
    fps = 1.0 / (cfg.sim.dt * cfg.sim.substeps)

    print("=" * 60)
    print("NetCapture Thrust Diagnostic")
    rope_type = "PBD particle" if _use_pbd_rope else "D6 joint"
    print(f"  Rope type: {rope_type}")
    print(f"  Envs: {num_envs}  |  Drones: {num_drones}  |  Rotors/drone: {num_rotors}")
    print(f"  Max thrust/drone: ~24 N  |  System weight: ~41 N")
    print(f"  Hover action ≈ -0.14  (43% throttle)")
    print(f"  Steps: {max_steps}  |  dt: {cfg.sim.dt:.3f}s  |  fps: {fps:.1f}")
    print("=" * 60)

    # ── Test cases ────────────────────────────────────────────────────────────
    test_cases = [
        ("action_-1.0",  -1.0),
        ("action_-0.14", -0.14),
        ("action_0.0",   0.0),
        ("action_0.5",   0.5),
        ("action_1.0",   1.0),
    ]

    all_results = {}

    for test_name, action_val in test_cases:
        print(f"\n{'─' * 50}")
        print(f"Test: {test_name}  (constant action = {action_val})")
        print(f"{'─' * 50}")

        # Fresh episode (positions reset, constraints settled)
        env.reset()

        # Constant action for all envs, all drones, all rotors
        action_tensor = torch.full(
            (num_envs, num_drones, num_rotors),
            action_val,
            device=env.device,
            dtype=torch.float32,
        )

        # ── Step loop ─────────────────────────────────────────────────────
        frames = []
        net_z_history = []
        drone_z_history = []

        for step in range(1, max_steps + 1):
            # Apply thrust, step physics, update state
            env.drone.apply_action(action_tensor)
            env.sim.step(render=True)
            env._compute_state_and_obs()

            net_z = env.net_centre[:, 2].cpu().numpy().copy()
            drone_z = (
                env.drone.get_state()[..., 2]
                .reshape(num_envs, num_drones)
                .mean(dim=-1)
                .cpu()
                .numpy()
                .copy()
            )

            frame = env.render(mode="rgb_array")
            frames.append(frame)
            net_z_history.append(net_z)
            drone_z_history.append(drone_z)

            if step % 100 == 0:
                print(f"  Step {step:4d}: net_z = {net_z[0]:.4f}, drone_z = {drone_z[0]:.4f}")

        # ── Stack and save ────────────────────────────────────────────────
        net_z_arr = np.stack(net_z_history)     # (max_steps, num_envs)
        drone_z_arr = np.stack(drone_z_history)
        frames_arr = np.stack(frames)           # (max_steps, H, W, 3)

        npz_path = os.path.join(_output_dir, f"{test_name}_data.npz")
        np.savez(npz_path, net_z=net_z_arr, drone_z=drone_z_arr, action=action_val)

        mp4_path = os.path.join(_output_dir, f"{test_name}.mp4")
        import imageio
        imageio.mimsave(mp4_path, frames_arr, fps=fps, codec="libx264")

        all_results[test_name] = {
            "net_z": net_z_arr,
            "drone_z": drone_z_arr,
            "action": action_val,
        }

        print(f"  → Saved: {os.path.basename(npz_path)}, {os.path.basename(mp4_path)}")

    # ── Comparison plot ───────────────────────────────────────────────────────
    _plot_comparison(test_cases, all_results)
    print(f"\nComparison plot: {os.path.join(_output_dir, 'comparison.png')}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    env.close()
    print("\nDone.")


def _plot_comparison(test_cases, all_results):
    """Generate a side-by-side plot of net_centre Z and drone Z over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_cases)))

    for (test_name, action_val), color in zip(test_cases, colors):
        net_z = all_results[test_name]["net_z"][:, 0]    # env 0 only
        drone_z = all_results[test_name]["drone_z"][:, 0]
        label = f"a={action_val}"
        axes[0].plot(net_z, label=label, color=color, linewidth=1.5)
        axes[1].plot(drone_z, label=label, color=color, linewidth=1.5)

    # Target hover height
    for ax in axes:
        ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5, label="target z=2.0")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Net Centre Z (m, env-local)")
    axes[0].set_title("Net Centre Z vs Time (env 0)")

    axes[1].set_ylabel("Mean Drone Z (m, env-local)")
    axes[1].set_title("Mean Drone Z vs Time (env 0)")

    fig.suptitle("NetCapture Constant-Thrust Diagnostic", fontsize=14, fontweight="bold")
    fig.tight_layout()

    png_path = os.path.join(_output_dir, "comparison.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()

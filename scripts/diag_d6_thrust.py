"""
Minimal diagnostic for the net_capture branch (D6 joint ropes).
Applies constant thrust to all 4 drones to test whether the
multi-drone net system can actually fly.

Usage (on net_capture branch):
    conda activate sim
    python scripts/diag_d6_thrust.py

Output:
    results_video/diag_d6_thrust/
        action_*.mp4, action_*_data.npz, comparison.png
"""

import ctypes
ctypes.CDLL("libX11.so.6").XInitThreads()

import os, sys, numpy as np, torch

# ── Path setup ────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_output_dir = os.path.join(_project_root, "results_video", "diag_d6_thrust")
os.makedirs(_output_dir, exist_ok=True)


def main():
    # ── Config (match net_capture branch defaults) ───────────────────────
    from omegaconf import OmegaConf

    base_env = OmegaConf.load(os.path.join(_project_root, "cfg", "base", "env_base.yaml"))
    base_sim = OmegaConf.load(os.path.join(_project_root, "cfg", "base", "sim_base.yaml"))
    task_cfg = OmegaConf.load(
        os.path.join(_project_root, "cfg", "task", "NetCapture", "NetCapture.yaml")
    )
    merged = OmegaConf.merge(base_env, base_sim, task_cfg)
    merged.env.num_envs = 4

    cfg = OmegaConf.create({
        "task": merged,
        "sim": merged.sim,
        "env": merged.env,
        "headless": True,
        "viewer": {"resolution": [1280, 720], "eye": [8., 0., 6.], "lookat": [0., 0., 1.]},
    })

    # ── Init ─────────────────────────────────────────────────────────────
    from omni_drones import init_simulation_app
    init_simulation_app(cfg)

    from omni_drones.envs.net_capture import NetCapture
    env = NetCapture(cfg, headless=cfg.headless)
    env.enable_render(True)

    num_envs = cfg.env.num_envs
    num_drones = cfg.task.num_drones
    num_rotors = 4
    max_steps = min(cfg.env.max_episode_length, 300)
    fps = 1.0 / (cfg.sim.dt * cfg.sim.substeps)

    print("=" * 60)
    print("NetCapture D6 Rope Thrust Diagnostic")
    print(f"  Envs: {num_envs}  |  Drones: {num_drones}  |  Steps: {max_steps}")
    print(f"  dt: {cfg.sim.dt:.3f}s  |  fps: {fps:.1f}")
    print("=" * 60)

    test_cases = [
        ("action_-1.0",  -1.0),
        ("action_-0.14", -0.14),
        ("action_0.0",   0.0),
        ("action_0.5",   0.5),
        ("action_1.0",   1.0),
    ]

    all_results = {}

    for test_name, action_val in test_cases:
        print(f"\n{'─'*50}")
        print(f"Test: {test_name}  (action = {action_val})")
        print(f"{'─'*50}")

        env.reset()

        action_tensor = torch.full(
            (num_envs, num_drones, num_rotors), action_val,
            device=env.device, dtype=torch.float32,
        )

        frames, net_z_hist, drone_z_hist = [], [], []

        for step in range(1, max_steps + 1):
            env.drone.apply_action(action_tensor)
            env.sim.step(render=True)
            env._compute_state_and_obs()

            net_z = env.net_centre[:, 2].cpu().numpy().copy()
            drone_states = env.drone.get_state()
            drone_z = drone_states[..., 2].reshape(num_envs, num_drones).mean(-1).cpu().numpy().copy()

            frame = env.render(mode="rgb_array")
            frames.append(frame)
            net_z_hist.append(net_z)
            drone_z_hist.append(drone_z)

            if step % 50 == 0:
                print(f"  Step {step:4d}: net_z={net_z[0]:.4f}, drone_z={drone_z[0]:.4f}")

        net_z_arr = np.stack(net_z_hist)
        drone_z_arr = np.stack(drone_z_hist)
        frames_arr = np.stack(frames)

        np.savez(os.path.join(_output_dir, f"{test_name}_data.npz"),
                 net_z=net_z_arr, drone_z=drone_z_arr, action=action_val)

        import imageio
        imageio.mimsave(os.path.join(_output_dir, f"{test_name}.mp4"),
                        frames_arr, fps=fps, codec="libx264")

        all_results[test_name] = {"net_z": net_z_arr, "drone_z": drone_z_arr, "action": action_val}
        print(f"  → Saved: {test_name}.mp4, {test_name}_data.npz")

    # ── Plot ──────────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_cases)))
    for (tn, av), c in zip(test_cases, colors):
        axes[0].plot(all_results[tn]["net_z"][:, 0], label=f"a={av}", color=c, lw=1.5)
        axes[1].plot(all_results[tn]["drone_z"][:, 0], label=f"a={av}", color=c, lw=1.5)
    for ax in axes:
        ax.axhline(y=merged.get("target_pos", [0,0,2])[2], color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Step"); ax.legend(fontsize=8)
    axes[0].set_ylabel("Net Centre Z (m)"); axes[0].set_title("Net Centre Z vs Time (env 0)")
    axes[1].set_ylabel("Mean Drone Z (m)"); axes[1].set_title("Mean Drone Z vs Time (env 0)")
    fig.suptitle("NetCapture D6 Rope Thrust Diagnostic", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(_output_dir, "comparison.png"), dpi=150)
    print(f"\nComparison: {os.path.join(_output_dir, 'comparison.png')}")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()

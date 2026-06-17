"""
Single-action diagnostic for the net_capture branch (D6 joint ropes).
Each run tests ONE or ALL constant thrust levels from a fresh simulation.

Usage (on net_capture branch):
    conda activate sim

    # Test a single action:
    python scripts/diag_d6_thrust.py --action 0.0
    python scripts/diag_d6_thrust.py --action 0.5 --steps 300

    # Test ALL 5 default actions, with proper reset between each:
    python scripts/diag_d6_thrust.py --all

    # Generate comparison plot from saved npz files:
    python scripts/diag_d6_thrust.py --plot

Action range: [-1.0, 1.0]
    -1.0  = 0% throttle   (zero thrust → free fall)
    -0.14 ≈ 43% throttle  (hover equilibrium for 4-drone+net system)
     0.0  = 50% throttle  (slow ascent)
     1.0  = 100% throttle (max climb)

    Any float in [-1, 1] is valid.

Output:
    results_video/diag_d6_thrust/
        action_{val}.mp4
        action_{val}_data.npz
"""

import ctypes
ctypes.CDLL("libX11.so.6").XInitThreads()

import argparse, os, sys, glob, numpy as np, torch

# ── Path setup ────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_output_dir = os.path.join(_project_root, "results_video", "diag_d6_thrust")
os.makedirs(_output_dir, exist_ok=True)

DEFAULT_ACTIONS = [-0.14, 0.0, 0.5, 1.0, -1.0]   # hover → rise → crash


def parse_args():
    parser = argparse.ArgumentParser(
        description="NetCapture D6 rope thrust diagnostic")
    parser.add_argument("--action", type=float, default=None,
                        help="Single action value in [-1.0, 1.0]")
    parser.add_argument("--all", action="store_true",
                        help="Run all 5 default actions with proper reset between each")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plot from saved npz files")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of simulation steps per action (default: 500)")
    return parser.parse_args()


def action_name(val: float) -> str:
    s = f"{val:.6f}".rstrip("0").rstrip(".")
    return f"action_{s}"


def run_env(cfg, actions: list, max_steps: int):
    """Create ONE env and run all actions with proper reset between each."""
    from omni_drones import init_simulation_app
    init_simulation_app(cfg)

    from omni_drones.envs.net_capture import NetCapture
    env = NetCapture(cfg, headless=cfg.headless)
    env.enable_render(True)

    num_envs = cfg.env.num_envs
    num_drones = cfg.task.num_drones
    num_rotors = 4
    fps = 1.0 / (cfg.sim.dt * cfg.sim.substeps)

    print("=" * 60)
    print("NetCapture D6 Rope Thrust Diagnostic")
    print(f"  Envs: {num_envs}  |  Drones: {num_drones}  |  Steps/action: {max_steps}")
    print(f"  Actions: {actions}")
    print(f"  dt: {cfg.sim.dt:.3f}s  |  Video fps: {fps:.1f}")
    print("=" * 60)

    all_results = {}

    for action_val in actions:
        name = action_name(action_val)
        norm = (action_val + 1.0) / 2.0
        throttle_est = np.sqrt(max(norm, 0))
        thrust_pct = norm * 100

        print(f"\n{'─'*50}")
        print(f"Test: {name}  (action = {action_val:.6f})")
        print(f"  Nominal throttle: {throttle_est:.1%}  |  Thrust: ~{thrust_pct:.0f}% max")
        print(f"{'─'*50}")

        # Full reset: teleports all bodies to spawn+offset, zeros velocities
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

            if step % 100 == 0:
                print(f"  Step {step:4d}: net_z={net_z[0]:.4f}, drone_z={drone_z[0]:.4f}")

        net_z_arr = np.stack(net_z_hist)
        drone_z_arr = np.stack(drone_z_hist)
        frames_arr = np.stack(frames)

        np.savez(os.path.join(_output_dir, f"{name}_data.npz"),
                 net_z=net_z_arr, drone_z=drone_z_arr, action=action_val)

        import imageio
        imageio.mimsave(os.path.join(_output_dir, f"{name}.mp4"),
                        frames_arr, fps=fps, codec="libx264")

        all_results[name] = {"net_z": net_z_arr, "drone_z": drone_z_arr, "action": action_val}
        print(f"  → Saved: {name}.mp4, {name}_data.npz")

    env.close()
    return all_results


def run_plot():
    """Generate comparison plot from all saved npz files."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz_files = sorted(glob.glob(os.path.join(_output_dir, "action_*_data.npz")))
    if len(npz_files) < 2:
        print("Need at least 2 npz files for comparison plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(npz_files)))

    for npz_path, color in zip(npz_files, colors):
        data = np.load(npz_path)
        name = os.path.basename(npz_path).replace("_data.npz", "")
        net_z = data["net_z"][:, 0]
        drone_z = data["drone_z"][:, 0]
        av = data["action"].item()
        axes[0].plot(net_z, label=f"a={av:.3f}", color=color, lw=1.5)
        axes[1].plot(drone_z, label=f"a={av:.3f}", color=color, lw=1.5)

    for ax in axes:
        ax.set_xlabel("Step"); ax.legend(fontsize=8)
    axes[0].set_ylabel("Net Centre Z (m)"); axes[0].set_title("Net Centre Z vs Time (env 0)")
    axes[1].set_ylabel("Mean Drone Z (m)"); axes[1].set_title("Mean Drone Z vs Time (env 0)")
    fig.suptitle("NetCapture D6 Rope Thrust Diagnostic", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(_output_dir, "comparison.png"), dpi=150)
    print(f"Comparison plot: {os.path.join(_output_dir, 'comparison.png')}")


if __name__ == "__main__":
    args = parse_args()

    if args.plot:
        run_plot()
        sys.exit(0)

    # Build config
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

    if args.all:
        actions = DEFAULT_ACTIONS
    elif args.action is not None:
        if args.action < -1.0 or args.action > 1.0:
            print(f"Error: action must be in [-1.0, 1.0], got {args.action}")
            sys.exit(1)
        actions = [args.action]
    else:
        print("Usage:  python scripts/diag_d6_thrust.py --all")
        print("        python scripts/diag_d6_thrust.py --action <float>")
        print("        python scripts/diag_d6_thrust.py --action 0.5 --steps 300")
        print("        python scripts/diag_d6_thrust.py --plot")
        print(f"\nAction range: [-1.0, 1.0] (any float)")
        print(f"  -1.0 = 0% throttle  |  -0.14 ≈ hover  |  0.0 = 50%  |  1.0 = 100%")
        sys.exit(1)

    run_env(cfg, actions, args.steps)

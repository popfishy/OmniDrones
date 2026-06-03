"""
Minimal script to inspect the NetCapture scene.

Headless mode (default, recommended for remote/no-X11):
    python scripts/test_net_capture_scene.py
    → saves USD stage to scripts/_outputs/test_net_capture.usd
    → open this file in Isaac Sim GUI to inspect interactively

GUI mode (only if local X11 is working):
    python scripts/test_net_capture_scene.py --gui
"""

import argparse
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true",
                    help="Run with GUI window (requires working X11/GLFW)")
parser.add_argument("--duration", type=float, default=120.0,
                    help="Seconds to keep alive (GUI mode only)")
parser.add_argument("--envs", type=int, default=1,
                    help="Number of environments")
args = parser.parse_args()

# Same initialization as train.py
import ctypes
ctypes.CDLL("libX11.so.6").XInitThreads()

from omni_drones import init_simulation_app
from omegaconf import OmegaConf

# Build headless config
cfg = OmegaConf.create({
    "headless": not args.gui,
    "sim": {"device": "cuda:0"},
})
init_simulation_app(cfg)

import omni.usd
import torch
from omni_drones.envs.net_capture import NetCapture

# ---- Load task config ----
task_cfg = OmegaConf.load("cfg/task/NetCapture/NetCapture.yaml")
base_env = OmegaConf.load("cfg/base/env_base.yaml")
base_sim = OmegaConf.load("cfg/base/sim_base.yaml")
task_cfg = OmegaConf.merge(base_env, base_sim, task_cfg)
task_cfg.env.num_envs = args.envs

# NetCapture expects cfg.task.xxx for task params, but IsaacEnv reads
# cfg.sim / cfg.env at root level.  Replicate Hydra's interpolation:
#   sim: ${task.sim}
#   env: ${task.env}
cfg = OmegaConf.create({
    "task": task_cfg,
    "sim": task_cfg.sim,
    "env": task_cfg.env,
    "headless": not args.gui,
    "viewer": {
        "eye": [8., 0., 6.],
        "lookat": [0., 0., 1.],
        "resolution": [1280, 720],
    },
})

# ---- Create environment ----
print(f"Creating NetCapture scene ({args.envs} env(s))...")
env = NetCapture(cfg, headless=not args.gui)

# ---- Let physics settle ----
print("Letting physics settle (100 steps)...")
for i in range(100):
    env.sim.step(render=True)

print(f"\nScene ready.")
print(f"  Mode:      {'GUI' if args.gui else 'headless'}")
print(f"  Envs:      {args.envs}")
print(f"  Drone:     {cfg.task.drone_model.name}")
print(f"  Net:       {cfg.task.net_rows}×{cfg.task.net_cols}, spacing={cfg.task.net_spacing}m")
print(f"  Ropes:     {cfg.task.rope_links} links × {cfg.task.rope_link_length}m")
print(f"  Nodes:     {cfg.task.net_rows * cfg.task.net_cols} (edges: {cfg.task.net_rows*(cfg.task.net_cols-1) + (cfg.task.net_rows-1)*cfg.task.net_cols})")

# ---- Save USD stage for later inspection ----
import os
os.makedirs("scripts/outputs", exist_ok=True)
usd_path = "scripts/outputs/test_net_capture.usd"
stage = omni.usd.get_context().get_stage()
stage.Export(usd_path)
print(f"\nUSD stage saved to: {usd_path}")
print(f"To inspect interactively, open Isaac Sim and load this file.")

if args.gui:
    print(f"\nGUI active — inspect for {args.duration}s (Ctrl-C to exit)")
    try:
        t0 = time.time()
        while time.time() - t0 < args.duration:
            env.sim.step(render=True)
    except KeyboardInterrupt:
        print("\nInterrupted.")
else:
    # Headless: step a bit more then exit
    print("Running a few more steps (headless)...")
    for i in range(50):
        env.sim.step(render=(i % 10 == 0))

print("Closing...")
env.close()
print("Done.")

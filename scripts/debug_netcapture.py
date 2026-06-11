#!/usr/bin/env python3
"""
NetCapture incremental diagnostic: 1 env → 4 envs (cloned).
Edit STEP to test progressively.

Usage:  cd ~/OmniDrones/scripts && python debug_netcapture.py
"""

import torch, sys
sys.path.insert(0, '..')

from omni_drones import init_simulation_app

print("=== NetCapture Cloning Diagnostic ===\n")

cfg = {
    "sim": {
        "device": "cuda:0",
        "dt": 0.016,
        "substeps": 1,
        "gravity": [0, 0, -9.81],
        "replicate_physics": False,
        "use_gpu_pipeline": True,
        "use_gpu": True,
        "enable_scene_query_support": True,
    },
    "headless": True,
}
simulation_app = init_simulation_app(cfg)

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import prims as prim_utils
from omni.isaac.cloner import GridCloner
from omni_drones.envs.net_capture.utils import NetCaptureGroup, NetCaptureCfg
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.scene as scene_utils
from omni_drones.views import RigidPrimView

sim_params = cfg["sim"].copy()
sim = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=sim_params.pop("dt", 0.016),
    rendering_dt=0.016,
    backend="torch",
    sim_params=sim_params,
    physics_prim_path="/physicsScene",
    device="cuda:0",
)
dev = sim.device

STEP = 5   # 1=drone, 2=drone+net, 3=full 1env, 4=cloned 4envs

if STEP == 1:
    print("── Step 1: Drone (articulation) 1 env ──")
    drone, _ = MultirotorBase.make("Hummingbird", "LeePositionController")
    scene_utils.design_scene()
    drone.spawn(translations=[(0, 0, 1.5)], prim_paths=["/World/envs/env_0/hummingbird"])
    sim.reset()
    drone.initialize("/World/envs/env_0/hummingbird")
    print(f"  pos={drone.get_world_poses(True)[0][0,0].tolist()}")
    for _ in range(10):
        sim.step(render=False)
    print("  ✓ PASSED\n")

elif STEP == 2:
    print("── Step 2: Drone + Net (1 env) ──")
    drone, _ = MultirotorBase.make("Hummingbird", "LeePositionController")
    scene_utils.design_scene()
    scene_utils.create_net(
        xform_path="/World/envs/env_0/net",
        rows=6, cols=6, spacing=0.25,
        node_mass=0.01, corner_mass=0.02,
        enable_collision=False,
    )
    drone.spawn(translations=[(0, 0, 1.5)], prim_paths=["/World/envs/env_0/hummingbird"])
    sim.reset()
    drone.initialize("/World/envs/env_0/hummingbird")
    for _ in range(10):
        sim.step(render=False)
    print("  ✓ PASSED\n")

elif STEP == 3:
    print("── Step 3: Full NetCapture (1 env, no cloning) ──")
    drone, _ = MultirotorBase.make("Hummingbird", "LeePositionController")
    scene_utils.design_scene()
    group = NetCaptureGroup(drone=drone, cfg=NetCaptureCfg(
        num_drones=4, net_rows=6, net_cols=6, net_spacing=0.25, rope_links=12))
    group.spawn(translations=[(0, 0, 0.5)], enable_collision=False)
    sim.reset()
    group.initialize()
    for i in range(10):
        sim.step(render=False)
    print("  ✓ PASSED\n")

elif STEP == 4:
    print("── Step 4: Full NetCapture (4 envs via GridCloner) ──")
    drone, _ = MultirotorBase.make("Hummingbird", "LeePositionController")
    scene_utils.design_scene()

    # Setup template (same as IsaacEnv._design_scene)
    template_env = "/World/envs/env_0"
    if not prim_utils.is_prim_path_valid(template_env):
        prim_utils.define_prim(template_env)

    group = NetCaptureGroup(drone=drone, cfg=NetCaptureCfg(
        num_drones=4, net_rows=6, net_cols=6, net_spacing=0.25, rope_links=12))
    group.spawn(translations=[(0, 0, 0.5)], enable_collision=False)

    # Clone to 4 environments (same as IsaacEnv.__init__)
    num_envs = 4
    env_spacing = 6.0
    cloner = GridCloner(spacing=env_spacing)
    cloner.define_base_env("/World/envs")
    env_paths = cloner.generate_paths("/World/envs/env", num_envs)
    print(f"  Cloning {num_envs} environments...")
    env_positions = cloner.clone(
        source_prim_path=template_env,
        prim_paths=env_paths,
        replicate_physics=False,
    )
    print(f"  Done. Positions: {env_positions[0]}, {env_positions[1]}, ...")

    # Filter collisions (same as IsaacEnv)
    cloner.filter_collisions(
        sim.get_physics_context().prim_path,
        "/World/collisions",
        prim_paths=env_paths,
        global_paths=["/World/defaultGroundPlane"],
    )

    sim.reset()

    # Initialize views (same as NetCapture.__init__ + group.initialize)
    drone.initialize(f"/World/envs/.*/{drone.name.lower()}_*")
    net_nodes_view = RigidPrimView(
        f"/World/envs/.*/Group_.*/net/node_*",
        reset_xform_properties=False,
    )
    net_nodes_view.initialize()
    net_edges_view = RigidPrimView(
        f"/World/envs/.*/Group_.*/net/edge_*/capsule",
        reset_xform_properties=False,
    )
    net_edges_view.initialize()
    rope_segs_view = RigidPrimView(
        f"/World/envs/.*/Group_.*/rope_*/seg_*",
        reset_xform_properties=False,
    )
    rope_segs_view.initialize()

    # Cache init poses (same as NetCapture.__init__)
    init_drone_pos, init_drone_rot = drone.get_world_poses(clone=True)
    init_net_pos, init_net_rot = net_nodes_view.get_world_poses(clone=True)
    print(f"  drone pos shape={init_drone_pos.shape}, net pos shape={init_net_pos.shape}")

    # Simulate training: 3 resets + steps (like training)
    for episode in range(3):
        # Reset all envs
        n = drone.n  # 4 drones per env
        n_nodes = 36  # 6x6 net
        all_env_ids = torch.arange(num_envs, device=dev)

        d_pos = init_drone_pos.reshape(num_envs, n, 3)[all_env_ids].reshape(-1, 3)
        d_rot = init_drone_rot.reshape(num_envs, n, 4)[all_env_ids].reshape(-1, 4)
        drone.set_world_poses(d_pos, d_rot, all_env_ids)
        drone.set_velocities(torch.zeros(num_envs * n, 6, device=dev), all_env_ids)

        n_ids = (all_env_ids.unsqueeze(-1) * n_nodes + torch.arange(n_nodes, device=dev)).reshape(-1)
        n_pos = init_net_pos.reshape(num_envs, n_nodes, 3)[all_env_ids].reshape(-1, 3)
        n_rot = init_net_rot.reshape(num_envs, n_nodes, 4)[all_env_ids].reshape(-1, 4)
        net_nodes_view.set_world_poses(n_pos, n_rot, n_ids)
        net_nodes_view.set_velocities(torch.zeros(num_envs * n_nodes, 6, device=dev), n_ids)

        # Step
        for _ in range(5):
            sim.step(render=False)

    print("  ✓ 3 episodes (reset→step) PASSED\n")

elif STEP == 5:
    print("── Step 5: Full reset simulation (mimics training _reset_idx) ──")
    drone, _ = MultirotorBase.make("Hummingbird", "LeePositionController")
    scene_utils.design_scene()

    template_env = "/World/envs/env_0"
    if not prim_utils.is_prim_path_valid(template_env):
        prim_utils.define_prim(template_env)

    group_cfg = NetCaptureCfg(
        num_drones=4, net_rows=6, net_cols=6, net_spacing=0.25, rope_links=12)
    group = NetCaptureGroup(drone=drone, cfg=group_cfg)
    group.spawn(translations=[(0, 0, 0.5)], enable_collision=False)

    num_envs = 4
    cloner = GridCloner(spacing=6.0)
    cloner.define_base_env("/World/envs")
    env_paths = cloner.generate_paths("/World/envs/env", num_envs)
    cloner.clone(source_prim_path=template_env, prim_paths=env_paths,
                 replicate_physics=False)
    cloner.filter_collisions(
        sim.get_physics_context().prim_path, "/World/collisions",
        prim_paths=env_paths, global_paths=["/World/defaultGroundPlane"],
    )
    sim.reset()

    group.initialize()

    # Cache ALL init poses (matching NetCapture.__init__ exactly)
    init_drone_pos, init_drone_rot = drone.get_world_poses(clone=True)
    init_net_nodes_pos, init_net_nodes_rot = group.net_nodes_view.get_world_poses(clone=True)
    init_net_edges_pos, init_net_edges_rot = group.net_edges_view.get_world_poses(clone=True)
    init_rope_segs_pos, init_rope_segs_rot = group.rope_segs_view.get_world_poses(clone=True)

    n = drone.n  # 4
    n_nodes = 36
    n_edges = init_net_edges_pos.shape[0] // num_envs
    n_segs = init_rope_segs_pos.shape[0] // num_envs

    print(f"  Cached: drone={init_drone_pos.shape}, net_nodes={init_net_nodes_pos.shape}")
    print(f"  edges={init_net_edges_pos.shape}, rope_segs={init_rope_segs_pos.shape}")

    # Full training reset simulation
    for episode in range(3):
        all_env_ids = torch.arange(num_envs, device=dev)

        # --- Full _reset_idx for all envs ---
        drone._reset_idx(all_env_ids)  # includes mass randomization

        # Drone root poses+velocities (ArticulationView API)
        d_pos = init_drone_pos.reshape(num_envs, n, 3)[all_env_ids].reshape(-1, 3)
        d_rot = init_drone_rot.reshape(num_envs, n, 4)[all_env_ids].reshape(-1, 4)
        drone.set_world_poses(d_pos, d_rot, all_env_ids)
        drone.set_velocities(torch.zeros(num_envs * n, 6, device=dev), all_env_ids)

        # Net nodes
        n_ids = (all_env_ids.unsqueeze(-1) * n_nodes + torch.arange(n_nodes, device=dev)).reshape(-1)
        net_p = init_net_nodes_pos.reshape(num_envs, n_nodes, 3)[all_env_ids].reshape(-1, 3)
        net_r = init_net_nodes_rot.reshape(num_envs, n_nodes, 4)[all_env_ids].reshape(-1, 4)
        group.net_nodes_view.set_world_poses(net_p, net_r, n_ids)
        group.net_nodes_view.set_velocities(torch.zeros(num_envs * n_nodes, 6, device=dev), n_ids)

        # Net edges
        e_ids = (all_env_ids.unsqueeze(-1) * n_edges + torch.arange(n_edges, device=dev)).reshape(-1)
        e_p = init_net_edges_pos.reshape(num_envs, n_edges, 3)[all_env_ids].reshape(-1, 3)
        e_r = init_net_edges_rot.reshape(num_envs, n_edges, 4)[all_env_ids].reshape(-1, 4)
        group.net_edges_view.set_world_poses(e_p, e_r, e_ids)
        group.net_edges_view.set_velocities(torch.zeros(num_envs * n_edges, 6, device=dev), e_ids)

        # Rope segments
        s_ids = (all_env_ids.unsqueeze(-1) * n_segs + torch.arange(n_segs, device=dev)).reshape(-1)
        s_p = init_rope_segs_pos.reshape(num_envs, n_segs, 3)[all_env_ids].reshape(-1, 3)
        s_r = init_rope_segs_rot.reshape(num_envs, n_segs, 4)[all_env_ids].reshape(-1, 4)
        group.rope_segs_view.set_world_poses(s_p, s_r, s_ids)
        group.rope_segs_view.set_velocities(torch.zeros(num_envs * n_segs, 6, device=dev), s_ids)

        # sim.step after reset (our fix from isaac_env.py)
        sim.step(render=False)

        # Compute drone state (matches _compute_state_and_obs)
        drone_states = drone.get_state()
        net_pos_world, _ = group.net_nodes_view.get_world_poses()
        net_vel = group.net_nodes_view.get_velocities()
        net_pos_world = net_pos_world.reshape(num_envs, -1, 3)
        net_vel = net_vel.reshape(num_envs, -1, 6)

        print(f"  Episode {episode}: drone_z={drone_states[0,0,2]:.3f}, "
              f"net_z={net_pos_world[0,:,2].mean():.3f}, "
              f"net_vel_max={net_vel.abs().max():.3f}")

        # Physics steps
        for step in range(5):
            drone.apply_action(torch.zeros(num_envs, n, 4, device=dev))
            sim.step(render=False)

    print("  ✓ PASSED — full reset + mass randomization + edges + ropes + step\n")

simulation_app.close()

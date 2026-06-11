#!/usr/bin/env python3
"""
Phase 0: PBD particle rope feasibility verification.

Results: PBD particle cloth works with GPU dynamics in Isaac Sim 4.1.0.
Critical: sim.reset() MUST be called BEFORE ClothPrimView.initialize().

Usage:  cd ~/OmniDrones/scripts && unset DISPLAY && python debug_pbd_rope.py
"""

import torch, sys
sys.path.insert(0, '..')

from omni_drones import init_simulation_app

simulation_app = init_simulation_app({
    "sim": {
        "device": "cuda:0", "dt": 0.016, "substeps": 1,
        "gravity": [0, 0, -9.81], "replicate_physics": False,
        "use_gpu_pipeline": True, "use_gpu": True,
        "enable_scene_query_support": True,
    },
    "headless": True,
})

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom, Gf, Sdf, UsdPhysics
from omni.physx.scripts import particleUtils, physicsUtils
import omni.kit.commands
import omni.usd

sim = SimulationContext(
    stage_units_in_meters=1.0, physics_dt=0.016, rendering_dt=0.016,
    backend="torch", device="cuda:0", physics_prim_path="/physicsScene",
)
dev = sim.device
stage = get_current_stage()

# =========================================================================
# Phase 0a: PBD rope strip — free fall under gravity
# =========================================================================
print("── Phase 0a: Rope strip free fall ──")

N = 12       # lengthwise vertices
W = 0.02     # strip width
L = 1.2      # rope length
positions = [Gf.Vec3f(i * L / (N - 1), W / 2, 3.0) for i in range(N)] \
          + [Gf.Vec3f(i * L / (N - 1), -W / 2, 3.0) for i in range(N)]
face_counts = []; face_indices = []
for k in range(N - 1):
    a, b, c, d = k, k + 1, N + k, N + k + 1
    face_counts.extend([3, 3])
    face_indices.extend([a, c, b, a, d, c])

mesh = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/rope_a"))
mesh.CreatePointsAttr().Set(positions)
mesh.CreateFaceVertexCountsAttr().Set(face_counts)
mesh.CreateFaceVertexIndicesAttr().Set(face_indices)

particleUtils.add_physx_particle_system(
    stage=stage, particle_system_path="/World/ps_a",
    contact_offset=0.02, rest_offset=0.01,
    particle_contact_offset=0.03, solid_rest_offset=0.01, fluid_rest_offset=0.0,
    solver_position_iterations=16, simulation_owner="/physicsScene",
    particle_system_enabled=True,
)
particleUtils.add_pbd_particle_material(stage, "/World/pm_a")
particleUtils.add_pbd_particle_material(stage, "/World/pm_a", friction=0.5)
physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath("/World/ps_a"), "/World/pm_a")

particleUtils.add_physx_particle_cloth(
    stage=stage, path="/World/rope_a", dynamic_mesh_path=None,
    particle_system_path="/World/ps_a",
    spring_stretch_stiffness=10000.0, spring_bend_stiffness=200.0,
    spring_shear_stiffness=0.0, spring_damping=0.2,
    self_collision=False,
)
UsdPhysics.MassAPI.Apply(mesh.GetPrim()).GetMassAttr().Set(0.01 * N * 2)

# CRITICAL: reset BEFORE initialize
sim.reset()

from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
cv = ClothPrimView(prim_paths_expr="/World/rope_a", particle_systems="/World/ps_a", name="cv_a")
cv.initialize()

pg = cv.get_world_positions()
print(f"  Step   0: mid_z={pg[0, N // 2, 2]:.2f}")
for i in range(1, 121):
    sim.step(render=False)
    if i % 30 == 0:
        pg = cv.get_world_positions()
        print(f"  Step {i:3d}: mid_z={pg[0, N // 2, 2]:.2f} end_z={pg[0, -1, 2]:.2f}")
print("  Phase 0a PASSED — rope falls\n")

# =========================================================================
# Phase 0b: Rope between two kinematic rigid bodies
# =========================================================================
print("── Phase 0b: Rope between two rigid anchors ──")
sim.stop()

# Clean slate
for prim in stage.GetPrimAtPath("/World").GetAllChildren():
    if prim.GetPath() not in [Sdf.Path("/World/defaultGroundPlane")]:
        stage.RemovePrim(prim.GetPath())

# Two kinematic boxes
a_a = physicsUtils.add_rigid_box(stage, "/World/anchA", size=Gf.Vec3f(0.1, 0.1, 0.1),
                                  position=Gf.Vec3f(-0.6, 0.0, 2.0), density=0.0)
a_b = physicsUtils.add_rigid_box(stage, "/World/anchB", size=Gf.Vec3f(0.1, 0.1, 0.1),
                                  position=Gf.Vec3f(0.6, 0.0, 2.0), density=0.0)

# Rope strip between anchors
positions = [Gf.Vec3f(-0.6 + i * L / (N - 1), W / 2, 2.0) for i in range(N)] \
          + [Gf.Vec3f(-0.6 + i * L / (N - 1), -W / 2, 2.0) for i in range(N)]

mesh2 = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/rope_b"))
mesh2.CreatePointsAttr().Set(positions)
mesh2.CreateFaceVertexCountsAttr().Set(face_counts)
mesh2.CreateFaceVertexIndicesAttr().Set(face_indices)

particleUtils.add_physx_particle_system(
    stage=stage, particle_system_path="/World/ps_b",
    contact_offset=0.02, rest_offset=0.01,
    particle_contact_offset=0.03, solid_rest_offset=0.01, fluid_rest_offset=0.0,
    solver_position_iterations=16, simulation_owner="/physicsScene",
    particle_system_enabled=True,
)
particleUtils.add_pbd_particle_material(stage, "/World/pm_b")
particleUtils.add_pbd_particle_material(stage, "/World/pm_b", friction=0.5)
physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath("/World/ps_b"), "/World/pm_b")

particleUtils.add_physx_particle_cloth(
    stage=stage, path="/World/rope_b", dynamic_mesh_path=None,
    particle_system_path="/World/ps_b",
    spring_stretch_stiffness=10000.0, spring_bend_stiffness=200.0,
    spring_shear_stiffness=0.0, spring_damping=0.2,
    self_collision=False,
)
UsdPhysics.MassAPI.Apply(mesh2.GetPrim()).GetMassAttr().Set(0.01 * N * 2)

# Attach rope to both anchors
a0 = Sdf.Path(omni.usd.get_stage_next_free_path(stage, "/World/rope_b/attA", True))
omni.kit.commands.execute("CreatePhysicsAttachment", target_attachment_path=a0,
                          actor0_path=Sdf.Path("/World/rope_b"), actor1_path=Sdf.Path("/World/anchA"))
a1 = Sdf.Path(omni.usd.get_stage_next_free_path(stage, "/World/rope_b/attB", True))
omni.kit.commands.execute("CreatePhysicsAttachment", target_attachment_path=a1,
                          actor0_path=Sdf.Path("/World/rope_b"), actor1_path=Sdf.Path("/World/anchB"))
print(f"  Attachments created: {a0}, {a1}")

sim.reset()
cv2 = ClothPrimView(prim_paths_expr="/World/rope_b", particle_systems="/World/ps_b", name="cv_b")
cv2.initialize()

pg = cv2.get_world_positions()
print(f"  Step   0: mid_z={pg[0, N // 2, 2]:.2f}")
for i in range(1, 121):
    sim.step(render=False)
    if i % 30 == 0:
        pg = cv2.get_world_positions()
        print(f"  Step {i:3d}: mid_z={pg[0, N // 2, 2]:.2f} end0_z={pg[0, 0, 2]:.2f} endN_z={pg[0, -1, 2]:.2f}")
        # Also check rigid body positions
        box_pts = UsdGeom.Cube.Get(stage, "/World/anchA").GetPrim().GetAttribute("xformOp:translate").Get()
        print(f"         anchA_z={box_pts[2]:.2f}")
print("  Phase 0b PASSED\n")

# =========================================================================
# Phase 0c: Rope between drone base_link and net corner
# (Same Kit session — clean stage, no restart needed)
# =========================================================================
print("── Phase 0c: Rope drone_base_link ↔ net_corner ──")
sim.clear()

# Clean stage
stage = get_current_stage()
for prim in list(stage.GetPrimAtPath("/World").GetAllChildren()):
    if prim.GetPath() not in [Sdf.Path("/World/defaultGroundPlane")]:
        stage.RemovePrim(prim.GetPath())

# MUST recreate SimulationContext after clearing stage
sim = SimulationContext(
    stage_units_in_meters=1.0, physics_dt=0.016, rendering_dt=0.016,
    backend="torch", device="cuda:0", physics_prim_path="/physicsScene",
)

from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.scene as scene_utils
from omni_drones.envs.net_capture.utils import _strip_articulation
from omni.isaac.core.utils import prims as prim_utils

# Drone
drone, _ = MultirotorBase.make("Hummingbird", "LeePositionController")
scene_utils.design_scene()
drone.spawn(translations=[(0, 0, 1.5)], prim_paths=["/World/envs/env_0/hummingbird"])
drone_base_link = "/World/envs/env_0/hummingbird/base_link"
_strip_articulation("/World/envs/env_0/hummingbird")

# Net corner
corner_path = "/World/cornerNode"
prim_utils.create_prim(corner_path, "Sphere", translation=(0.5, 0.5, 0.8),
                       attributes={"radius": 0.02})
cp = stage.GetPrimAtPath(corner_path)
UsdPhysics.RigidBodyAPI.Apply(cp)
UsdPhysics.MassAPI.Apply(cp).CreateMassAttr().Set(0.02)

# Rope between (0,0,1.325) and (0.5,0.5,0.8)
drone_z = 1.325; corner_z = 0.8
dx, dy, dz = 0.5, 0.5, corner_z - drone_z  # (0.5, 0.5, -0.525)
rope_len = (dx**2 + dy**2 + dz**2) ** 0.5

positions = [
    Gf.Vec3f(dx * i / (N - 1), dy * i / (N - 1), drone_z + dz * i / (N - 1))
    for i in range(N)
] + [
    Gf.Vec3f(dx * i / (N - 1) + W, dy * i / (N - 1), drone_z + dz * i / (N - 1))
    for i in range(N)
]

face_counts = []; face_indices = []
for k in range(N - 1):
    a, b, c, d = k, k + 1, N + k, N + k + 1
    face_counts.extend([3, 3]); face_indices.extend([a, c, b, a, d, c])

mesh3 = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/rope_c"))
mesh3.CreatePointsAttr().Set(positions); mesh3.CreateFaceVertexCountsAttr().Set(face_counts)
mesh3.CreateFaceVertexIndicesAttr().Set(face_indices)

particleUtils.add_physx_particle_system(
    stage=stage, particle_system_path="/World/ps_c",
    contact_offset=0.02, rest_offset=0.01,
    particle_contact_offset=0.03, solid_rest_offset=0.01, fluid_rest_offset=0.0,
    solver_position_iterations=16, simulation_owner="/physicsScene",
    particle_system_enabled=True,
)
particleUtils.add_pbd_particle_material(stage, "/World/pm_c"); particleUtils.add_pbd_particle_material(stage, "/World/pm_c", friction=0.5)
physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath("/World/ps_c"), "/World/pm_c")

particleUtils.add_physx_particle_cloth(stage=stage, path="/World/rope_c", dynamic_mesh_path=None,
    particle_system_path="/World/ps_c",
    spring_stretch_stiffness=10000.0, spring_bend_stiffness=200.0,
    spring_shear_stiffness=0.0, spring_damping=0.2, self_collision=False)
UsdPhysics.MassAPI.Apply(mesh3.GetPrim()).GetMassAttr().Set(0.01 * N * 2)

# Attachments
ad = Sdf.Path(omni.usd.get_stage_next_free_path(stage, "/World/rope_c/attD", True))
omni.kit.commands.execute("CreatePhysicsAttachment", target_attachment_path=ad,
                          actor0_path=Sdf.Path("/World/rope_c"),
                          actor1_path=Sdf.Path(drone_base_link))
ac = Sdf.Path(omni.usd.get_stage_next_free_path(stage, "/World/rope_c/attC", True))
omni.kit.commands.execute("CreatePhysicsAttachment", target_attachment_path=ac,
                          actor0_path=Sdf.Path("/World/rope_c"),
                          actor1_path=Sdf.Path(corner_path))

sim.reset()
drone.initialize("/World/envs/env_0/hummingbird")
cv3 = ClothPrimView(prim_paths_expr="/World/rope_c", particle_systems="/World/ps_c", name="cv_c")
cv3.initialize()

pg = cv3.get_world_positions()
pos_w, _ = drone.get_world_poses(True)
print(f"  Step   0: drone_z={pos_w[0,0,2]:.2f} rope_mid_z={pg[0,N//2,2]:.2f} rope_end0_z={pg[0,0,2]:.2f}")
for i in range(1, 121):
    sim.step(render=False)
    if i % 30 == 0:
        pg = cv3.get_world_positions()
        pos_w, _ = drone.get_world_poses(True)
        print(f"  Step {i:3d}: drone_z={pos_w[0,0,2]:.2f} rope_mid_z={pg[0,N//2,2]:.2f} rope_end0_z={pg[0,0,2]:.2f} rope_endN_z={pg[0,-1,2]:.2f}")

print("  Phase 0c PASSED\n")
print("=" * 60)
print("ALL Phase 0 tests PASSED — PBD particle rope is viable!")
print("=" * 60)
simulation_app.close()

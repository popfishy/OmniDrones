# Plan: Fix PhysX GPU API Errors During NetCapture Reset

## Context

Training `NetCapture` with `headless=true` triggers PhysX errors during `_reset_idx`:

```
PxRigidDynamic::setLinearVelocity(): illegal with eENABLE_DIRECT_GPU_API
PxRigidDynamic::setAngularVelocity(): illegal with eENABLE_DIRECT_GPU_API
PxRigidDynamic::setGlobalPose: pose is not valid
```

The NetCapture environment strips `ArticulationRootAPI` from drones and uses
`RigidPrimView` (GPU tensor API) for all bodies.  All OmniDrones code paths
deliberately route through `_physics_view.set_transforms()` /
`_physics_view.set_velocities()`, which are GPU-safe.  **No OmniDrones code
directly calls CPU PhysX APIs.**

## Root Cause

The `_reset_idx` ordering creates a **temporary constraint inconsistency**
between FixedJoint-connected RigidBodies (drone base_link ↔ rotors):

1. `self.drone.set_world_poses(d_pos, d_rot, env_ids)` at line 259 — teleports base_link to init position
2. … net/rope/velocity operations (lines 261–305) …
3. `self.drone.rotors_view.set_world_poses(rot_pos, rot_rot, env_ids)` at line 293 — teleports rotors to init position

Between steps 1 and 3, base_link is at its init position but rotors are still
at their **previous episode's** position.  The FixedJoints connecting them
are now inconsistent.  PhysX internally attempts to resolve this via per-actor
CPU APIs (`PxRigidDynamic::setGlobalPose/setLinearVelocity/setAngularVelocity`),
which are illegal when `eENABLE_DIRECT_GPU_API` is enabled.

The "pose is not valid" error is a secondary consequence of PhysX constraint
resolution producing NaN from inconsistent connected-body positions.

## Fix

### File 1: `omni_drones/envs/net_capture/net_capture.py`

**Reorder `_reset_idx` to set rotor poses immediately after base_link poses:**

Move the rotor `set_world_poses` / `set_velocities` calls from lines 291–295
to right after the base_link `set_world_poses` call at line 259.  This
minimizes the window where FixedJoint-connected bodies are inconsistent.

Also reorder to set drone velocities immediately after drone poses (before
net/rope operations), ensuring the drone subsystem is fully consistent before
touching unrelated bodies.

**New order:**
1. `self.drone._reset_idx(env_ids)` — tensors only
2. `self.drone.set_world_poses(d_pos, d_rot, env_ids)` — base_link poses
3. `self.drone.rotors_view.set_world_poses(rot_pos, rot_rot, env_ids)` — **NOW: immediately after base_link**
4. `self.drone.set_velocities(zero, env_ids)` — base_link velocities  
5. `self.drone.rotors_view.set_velocities(zero, env_ids)` — rotor velocities
6. Net nodes/edges/ropes: poses + velocities

### File 2: `omni_drones/envs/isaac_env.py`

**Uncomment `self.sim.step(render=False)` at line 255:**

```python
# self.sim.step(render=False)  →  self.sim.step(render=False)
```

Running one GPU physics step after `_reset_idx` allows PhysX to resolve all
constraint states through the GPU pipeline atomically, rather than letting
individual `set_transforms()` calls trigger partial constraint resolution.

**Add NaN guard for debugging:** In `_reset`, after `_reset_idx`, optionally
check that drone states are finite to catch NaN propagation early.

## Verification

1. Run the training command from the error report:
   ```bash
   python train.py algo=mappo headless=true task=NetCapture/NetCapture \
     total_frames=5_000_000 wandb.mode=offline algo.entropy_coef=0.01
   ```
2. Confirm no `PxRigidDynamic::setLinearVelocity` or `setAngularVelocity` errors appear
3. Confirm no `setGlobalPose: pose is not valid` errors appear
4. Verify episodes complete without NaN in rewards/states
5. Confirm the "Physics Simulation View is not created yet" warnings are also resolved (they indicate `_physics_view` not being available at init time)

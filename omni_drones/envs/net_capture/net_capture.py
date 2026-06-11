# MIT License
#
# Copyright (c) 2026 Jiaqi Yang, National University of Defense Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import omni.usd
import torch
import torch.distributions as D
from torch.func import vmap
from pxr import Gf

import omni_drones.utils.scene as scene_utils

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
)

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.utils.torch import (
    cpos,
    off_diag,
    others,
    quat_axis,
    euler_to_quaternion,
)
from omni_drones.robots.drone import MultirotorBase

from .utils import NetCaptureGroup, NetCaptureCfg


class NetCapture(IsaacEnv):
    r"""
    A cooperative control task where a group of UAVs carry a flexible net
    connected via ropes. The goal for the agents is to collaboratively
    control the net to descend and capture a static target object on the ground.

    ## Observation

    The observation space contains the following items:

    - ``obs_self`` (1, \*): The state of each UAV observed by itself, containing
      its kinematic information with position relative to the net centre.
      It also includes a one-hot vector indicating each drone's identity.
    - ``obs_others`` (k-1, \*): The observed states of other agents.
    - ``obs_net`` (1, \*): Net centre + 4 corner positions relative to the drone.
    - ``obs_target`` (1, \*): Target position relative to the drone.

    ## Reward

    - ``reward_coverage``: Reward for the net covering the target in the XY plane.
    - ``reward_descend``: Reward for the net descending to the target's height.
    - ``reward_capture``: Bonus when the net drapes over the target.
    - ``reward_up``: Reward for keeping drones upright.
    - ``reward_effort``: Penalty for energy consumption.

    ## Config

    | Parameter           | Type  | Default       | Description                          |
    | ------------------- | ----- | ------------- | ------------------------------------ |
    | ``drone_model``     | str   | "hummingbird" |                                      |
    | ``num_drones``      | int   | 4             |                                      |
    | ``net_rows``        | int   | 3             | Number of node rows in the net.      |
    | ``net_cols``        | int   | 3             | Number of node columns in the net.   |
    | ``net_spacing``     | float | 0.5           | Distance between adjacent net nodes. |
    | ``rope_links``      | int   | 12            | Number of segments per rope.         |
    | ``target_size``     | list  | [0.3,0.3,0.3] | Target cube dimensions.              |
    | ``reset_thres``     | float | 2.0           | Net deviation that triggers reset.   |
    | ``capture_thres``   | float | 0.3           | XY distance to count as covering.    |
    | ``safe_distance``   | float | 0.5           | Min separation before penalty.       |
    """

    def __init__(self, cfg, headless):
        self.reward_coverage_weight = cfg.task.reward_coverage_weight
        self.reward_descend_weight = cfg.task.reward_descend_weight
        self.reward_up_weight = cfg.task.reward_up_weight
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reset_thres = cfg.task.reset_thres
        self.safe_distance = cfg.task.safe_distance
        self.action_scale = cfg.task.get("action_scale", 1.0)

        # Must be set before super().__init__() because _set_specs() uses them
        self.net_rows = cfg.task.net_rows
        self.net_cols = cfg.task.net_cols
        self.net_spacing = cfg.task.net_spacing
        self.n_nodes = self.net_rows * self.net_cols

        super().__init__(cfg, headless)

        # group.initialize() is now called inside _design_scene() BEFORE sim.reset()
        # to ensure views are registered before the GPU physics view is created.

        # Target is a pure visual prim (no RigidBody) — track position via tensor
        # to avoid GPU API conflicts.  Position set via USD API in _reset_idx.
        self._target_prim_paths = [
            f"/World/envs/env_{i}/target" for i in range(self.num_envs)
        ]
        self._target_heading_prim_paths = [
            f"/World/envs/env_{i}/target_heading" for i in range(self.num_envs)
        ]

        # Cache initial poses for GPU-compatible reset via RigidPrimView
        self.init_drone_pos, self.init_drone_rot = self.drone.get_world_poses(clone=True)
        # Also cache rotor poses (reset with RigidPrimView alongside base_link)
        self.init_rotor_pos, self.init_rotor_rot = self.drone.rotors_view.get_world_poses(clone=True)
        self.init_net_nodes_pos, self.init_net_nodes_rot = self.group.net_nodes_view.get_world_poses(clone=True)
        self.init_net_edges_pos, self.init_net_edges_rot = self.group.net_edges_view.get_world_poses(clone=True)
        if not self.cfg.task.get("use_pbd_rope", True):
            self.init_rope_segs_pos, self.init_rope_segs_rot = self.group.rope_segs_view.get_world_poses(clone=True)

        # Target above initial drone height (1.325), forcing upward scooping.
        # Net at 0.5, drones at ~1.325, target at 1.5–2.5.
        self.target_pos_dist = D.Uniform(
            torch.tensor([-1.0, -1.0, 1.5], device=self.device),
            torch.tensor([1.0, 1.0, 2.5], device=self.device),
        )
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_heading_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.alpha = 0.8

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        group_cfg = NetCaptureCfg(
            num_drones=self.cfg.task.num_drones,
            net_rows=self.cfg.task.net_rows,
            net_cols=self.cfg.task.net_cols,
            net_spacing=self.cfg.task.net_spacing,
            rope_links=self.cfg.task.rope_links,
        )

        self.group = NetCaptureGroup(drone=self.drone, cfg=group_cfg)

        scene_utils.design_scene()

        self.group.spawn(translations=[(0, 0, 0.5)], enable_collision=True)
        # Net at env z = 0.5, drones at 0.5 + 0.825 = 1.325
        # Target above drones at z ∈ [1.5, 2.5]

        # Visual-only target marker (pure USD prim, NO RigidBody — avoids GPU API errors)
        # Position updated via USD API in _reset_idx; read from self.target_pos tensor.
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import UsdGeom, Gf

        target_path = "/World/envs/env_0/target"
        target = prim_utils.create_prim(
            prim_path=target_path,
            prim_type="Sphere",
            translation=(0.0, 0.0, 2.0),
            attributes={"radius": 0.06},
        )
        UsdGeom.Imageable(target).MakeVisible()

        # Heading direction arrow: long capsule along X, rotated to heading in XY plane.
        arrow_path = "/World/envs/env_0/target_heading"
        arrow = UsdGeom.Capsule.Define(
            omni.usd.get_context().get_stage(),
            arrow_path,
        )
        arrow.CreateAxisAttr("X")       # capsule along X (in XY plane)
        arrow.CreateHeightAttr(0.3)     # shaft length 0.3m
        arrow.CreateRadiusAttr(0.015)   # thin shaft
        arrow.AddTranslateOp().Set(Gf.Vec3f(0, 0, 2.0))
        arrow.AddOrientOp().Set(Gf.Quatf(1.0))  # create xformOp:orient so we can set it later
        UsdGeom.Imageable(arrow.GetPrim()).MakeVisible()

        # ---- Initialize all views BEFORE super().__init__ locks the physics ----
        # This is critical: views must register their prim path patterns while the
        # stage is still being built.  sim.reset() (called later in IsaacEnv.__init__)
        # creates the GPU physics simulation view; views initialized after that
        # point may fail to create their _physics_view handles.
        self.group.initialize()

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1] + self.drone.n

        # Net observation per drone: net centre (6D) + 4 corners (4×6D) = 30D
        net_obs_dim = (1 + 4) * 6
        # Target observation per drone: position (3D) + heading vector (3D)
        target_obs_dim = 6

        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, drone_state_dim)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n - 1, 13 + 1)),
            "obs_net": UnboundedContinuousTensorSpec((1, net_obs_dim)),
            "obs_target": UnboundedContinuousTensorSpec((1, target_obs_dim)),
        }).to(self.device)

        # Centralized critic: full state
        observation_central_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)),
            "state_net": UnboundedContinuousTensorSpec((self.n_nodes, 6)),
            "state_target": UnboundedContinuousTensorSpec((1, 6)),
        }).to(self.device)

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
                "observation_central": observation_central_spec,
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([self.drone.action_spec] * self.drone.n, dim=0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state"),
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(self.drone.n),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "net_target_dist": UnboundedContinuousTensorSpec(1),
            "net_height": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(self.drone.n),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        # Sample target
        target_pos = self.target_pos_dist.sample(env_ids.shape)
        self.target_pos[env_ids] = target_pos

        n = self.drone.n
        self.drone._reset_idx(env_ids)

        # ---- Reset drone bodies (base_link + rotors) atomically ----
        # Set both base_link and rotor poses back-to-back to avoid a constraint
        # inconsistency window that would trigger CPU-side PhysX API calls.
        d_pos = self.init_drone_pos.reshape(self.num_envs, n, 3)[env_ids].reshape(-1, 3)
        d_rot = self.init_drone_rot.reshape(self.num_envs, n, 4)[env_ids].reshape(-1, 4)
        self.drone.set_world_poses(d_pos, d_rot, env_ids)

        rot_pos = self.init_rotor_pos.reshape(self.num_envs, n, self.drone.num_rotors, 3)[env_ids]
        rot_rot = self.init_rotor_rot.reshape(self.num_envs, n, self.drone.num_rotors, 4)[env_ids]
        self.drone.rotors_view.set_world_poses(rot_pos, rot_rot, env_ids)

        # Zero drone velocities
        self.drone.set_velocities(
            torch.zeros(len(env_ids) * n, 6, device=self.device), env_ids)
        self.drone.rotors_view.set_velocities(
            torch.zeros(len(env_ids) * n * self.drone.num_rotors, 6, device=self.device), env_ids)

        # ---- Reset net nodes (1D view, build flat indices) ----
        n_nodes = self.n_nodes
        n_ids = (env_ids.unsqueeze(-1) * n_nodes
                 + torch.arange(n_nodes, device=self.device)).reshape(-1)
        net_pos = self.init_net_nodes_pos.reshape(self.num_envs, n_nodes, 3)[env_ids].reshape(-1, 3)
        net_rot = self.init_net_nodes_rot.reshape(self.num_envs, n_nodes, 4)[env_ids].reshape(-1, 4)
        self.group.net_nodes_view.set_world_poses(net_pos, net_rot, n_ids)

        # ---- Reset net edge capsules ----
        n_edges = self.init_net_edges_pos.shape[0] // self.num_envs
        e_ids = (env_ids.unsqueeze(-1) * n_edges
                 + torch.arange(n_edges, device=self.device)).reshape(-1)
        edge_pos = self.init_net_edges_pos.reshape(self.num_envs, n_edges, 3)[env_ids].reshape(-1, 3)
        edge_rot = self.init_net_edges_rot.reshape(self.num_envs, n_edges, 4)[env_ids].reshape(-1, 4)
        self.group.net_edges_view.set_world_poses(edge_pos, edge_rot, e_ids)

        if self.cfg.task.get("use_pbd_rope", True):
            # PBD rope: particles are attached to rigid bodies via
            # PhysxPhysicsAttachment.  When rigid bodies are teleported
            # (set_world_poses above), PhysX automatically repositions the
            # attached particles.  No separate rope reset needed.
            pass
        else:
            # ---- Reset D6 joint rope segments ----
            n_segs = self.init_rope_segs_pos.shape[0] // self.num_envs
            s_ids = (env_ids.unsqueeze(-1) * n_segs
                     + torch.arange(n_segs, device=self.device)).reshape(-1)
            seg_pos = self.init_rope_segs_pos.reshape(self.num_envs, n_segs, 3)[env_ids].reshape(-1, 3)
            seg_rot = self.init_rope_segs_rot.reshape(self.num_envs, n_segs, 4)[env_ids].reshape(-1, 4)
            self.group.rope_segs_view.set_world_poses(seg_pos, seg_rot, s_ids)

        # ---- Zero net velocities (GPU API) ----
        self.group.net_nodes_view.set_velocities(
            torch.zeros(len(env_ids) * n_nodes, 6, device=self.device), n_ids)
        self.group.net_edges_view.set_velocities(
            torch.zeros(len(env_ids) * n_edges, 6, device=self.device), e_ids)
        if not self.cfg.task.get("use_pbd_rope", True):
            self.group.rope_segs_view.set_velocities(
                torch.zeros(len(env_ids) * n_segs, 6, device=self.device), s_ids)

        # Sample target heading — always upward (z > 0).
        # Uniform direction in the upper hemisphere with at least 30° upward tilt.
        theta = torch.rand(env_ids.shape, device=self.device) * (torch.pi / 3)  # [0, 60°] from vertical
        phi = torch.rand(env_ids.shape, device=self.device) * (2 * torch.pi)     # [0, 360°] azimuth
        heading_vec = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),                          # cos(θ) > 0 → upward
        ], dim=-1)  # (n, 3), unit vector, z ≥ 0.5
        self.target_heading_vec[env_ids] = heading_vec

        # Quaternion for arrow indicator: rotate from +Z to heading_vec.
        # Axis = cross(+Z, heading), angle = acos(dot(+Z, heading))
        z_axis = torch.tensor([0., 0., 1.], device=self.device)
        axis = torch.cross(z_axis.expand_as(heading_vec), heading_vec, dim=-1)
        axis_norm = axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        axis = axis / axis_norm
        angle = torch.acos(heading_vec[..., 2:3].clamp(-1, 1))  # acos(z·heading)
        half_angle = angle / 2
        heading_quat = torch.cat([
            torch.cos(half_angle),                              # w
            axis[..., 0:1] * torch.sin(half_angle),             # x
            axis[..., 1:2] * torch.sin(half_angle),             # y
            axis[..., 2:3] * torch.sin(half_angle),             # z
        ], dim=-1)  # (n, 4)

        # Set target visual marker poses via USD API (NO RigidPrimView — avoids GPU conflict)
        target_pos_world = target_pos + self.envs_positions[env_ids]
        for idx, env_id in enumerate(env_ids.tolist()):
            tp = target_pos_world[idx].tolist()
            # set_prim_property takes a prim PATH string, not a Prim object
            omni.isaac.core.utils.prims.set_prim_property(
                self._target_prim_paths[env_id], "xformOp:translate", Gf.Vec3f(*tp))
            # Position + orient heading arrow
            ap = (target_pos_world[idx] + heading_vec[idx] * 0.15).tolist()
            hq = heading_quat[idx].tolist()
            omni.isaac.core.utils.prims.set_prim_property(
                self._target_heading_prim_paths[env_id], "xformOp:translate", Gf.Vec3f(*ap))
            omni.isaac.core.utils.prims.set_prim_property(
                self._target_heading_prim_paths[env_id], "xformOp:orient",
                Gf.Quatf(hq[0], hq[1], hq[2], hq[3]))

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions * self.action_scale)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()

        # Net node states:
        # RigidPrimView returns flat (total_nodes, 3) — reshape to (N, n_nodes, 3) first
        # so get_env_poses can correctly subtract per-environment positions.
        net_pos_world, _ = self.group.net_nodes_view.get_world_poses()
        net_vel_world = self.group.net_nodes_view.get_velocities()
        net_pos_world = net_pos_world.reshape(self.num_envs, -1, 3)
        net_vel = net_vel_world.reshape(self.num_envs, -1, 6)
        net_pos, _ = self.get_env_poses((net_pos_world, None))

        # Reshape to (num_envs, rows, cols, 6)
        net_pos_grid = net_pos.reshape(self.num_envs, self.net_rows, self.net_cols, 3)
        net_vel_grid = net_vel.reshape(self.num_envs, self.net_rows, self.net_cols, 6)

        # Net centre (mean of all node positions)
        self.net_centre = net_pos.reshape(self.num_envs, -1, 3).mean(dim=1)  # (num_envs, 3)
        self.net_centre_vel = net_vel[..., :3].reshape(self.num_envs, -1, 3).mean(dim=1)  # (num_envs, 3)

        # Corner nodes: (0,0), (0,-1), (-1,0), (-1,-1)
        # Instance variable for reward function (stretch penalty)
        self.corner_pos = torch.stack([
            net_pos_grid[:, 0, 0],
            net_pos_grid[:, 0, -1],
            net_pos_grid[:, -1, 0],
            net_pos_grid[:, -1, -1],
        ], dim=1)  # (num_envs, 4, 3)
        corner_vel = torch.stack([
            net_vel_grid[:, 0, 0, :3],
            net_vel_grid[:, 0, -1, :3],
            net_vel_grid[:, -1, 0, :3],
            net_vel_grid[:, -1, -1, :3],
        ], dim=1)  # (num_envs, 4, 3) — linear velocity only

        # Compute net normal from corner cross product
        v1 = self.corner_pos[:, 1] - self.corner_pos[:, 0]  # n02 - n00
        v2 = self.corner_pos[:, 2] - self.corner_pos[:, 0]  # n20 - n00
        self.net_normal = torch.nn.functional.normalize(
            torch.cross(v1, v2, dim=-1), dim=-1
        )  # (num_envs, 3)

        # Target position in env frame
        target_pos = self.target_pos  # env-local, set by _reset_idx

        # Relative positions
        drone_pos = self.drone_states[..., :3]  # (num_envs, n, 3)
        net_centre_rpos = self.net_centre.unsqueeze(1) - drone_pos  # drone → net centre
        corner_rpos = self.corner_pos.unsqueeze(2) - drone_pos.unsqueeze(1)  # (N, 4, n, 3)
        target_rpos = target_pos.unsqueeze(1) - drone_pos  # drone → target (n, 3)

        # Drone inter-relative positions
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)
        self.drone_pdist = torch.norm(self.drone_rpos, dim=-1, keepdim=True)

        # ---- Assemble per-drone observation ----
        obs = TensorDict({}, [self.num_envs, self.drone.n])
        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)
        obs["obs_self"] = torch.cat(
            [net_centre_rpos, self.drone_states[..., 3:], identity], dim=-1
        ).unsqueeze(2)  # (num_envs, n, 1, state_dim)

        obs["obs_others"] = torch.cat(
            [self.drone_rpos, self.drone_pdist, vmap(others)(self.drone_states[..., 3:13])],
            dim=-1,
        )  # (num_envs, n, n-1, ...)

        # Net observation per drone: centre + 4 corners (pos+vel = 6 each → 30 total)
        net_obs_self = torch.cat([
            net_centre_rpos,
            self.net_centre_vel.unsqueeze(1).expand(-1, self.drone.n, -1),
        ], dim=-1)  # (num_envs, n, 6)

        # Corner positions: (num_envs, 4, n, 3) → (num_envs, n, 12)
        corner_rpos_flat = corner_rpos.permute(0, 2, 1, 3).reshape(
            self.num_envs, self.drone.n, -1
        )
        # Corner linear velocities: (num_envs, 4, 3) → (num_envs, n, 12)
        corner_vel_flat = corner_vel.unsqueeze(2).expand(-1, -1, self.drone.n, -1)
        corner_vel_flat = corner_vel_flat.permute(0, 2, 1, 3).reshape(
            self.num_envs, self.drone.n, -1
        )
        corner_obs_self = torch.cat([corner_rpos_flat, corner_vel_flat], dim=-1)

        obs["obs_net"] = torch.cat([
            net_obs_self, corner_obs_self,
        ], dim=-1).unsqueeze(2)  # (num_envs, n, 1, net_obs_dim)

        # Target: position (3) + heading vector (3) relative to each drone
        heading_rpos = self.target_heading_vec.unsqueeze(1).expand(-1, self.drone.n, -1)
        obs["obs_target"] = torch.cat([
            target_rpos, heading_rpos,
        ], dim=-1).unsqueeze(2)  # (num_envs, n, 1, 6)

        # ---- Assemble centralized critic observation ----
        state = TensorDict({}, self.num_envs)
        state["state_drones"] = obs["obs_self"].squeeze(2)  # (num_envs, n, state_dim)
        state["state_net"] = torch.cat([
            net_pos.reshape(self.num_envs, -1, 3),
            net_vel[..., :3].reshape(self.num_envs, -1, 3),
        ], dim=-1)  # (num_envs, n_nodes, 6)
        state["state_target"] = torch.cat([
            target_pos, self.target_heading_vec,
        ], dim=-1).unsqueeze(1)  # (num_envs, 1, 6)

        # ---- Stats ----
        self.net_target_dist = torch.norm(
            self.net_centre - target_pos, dim=-1, keepdim=True
        )

        self.stats["net_target_dist"].lerp_(self.net_target_dist, (1 - self.alpha))
        self.stats["net_height"].lerp_(self.net_centre[..., 2:3], (1 - self.alpha))
        self.stats["uprightness"].lerp_(
            self.drone.up[..., 2].mean(-1, keepdim=True), (1 - self.alpha)
        )
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1 - self.alpha))

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "observation_central": state,
                },
                "stats": self.stats.clone(),
            },
            self.num_envs,
        )

    def _compute_reward_and_done(self):
        """
        "Scooping" reward — fly net beneath target, lift from below.

        Let:
          p_net, p_tgt ∈ ℝ³ : net centre & target position
          dist_xy = ‖p_net[...,:2] - p_tgt[...,:2]‖   (XY plane)
          z_diff  = p_tgt_z - p_net_z                    (+ when net below target)

        1. r_xy      = exp(-2·dist_xy)                   XY alignment under target
        2. r_z       = exp(-2·z_diff)    if z_diff > 0   climb toward target
                       exp( 1·z_diff)    if z_diff ≤ 0   stable above target
        3. r_capture = r_xy · r_z                        full scoop: aligned + lifted
        4. r_stretch = exp(-2·|diag_ideal - diag_actual|) prevent net collapse
        5. r_head    = ((n_net·v_tgt + 1)/2)²            heading alignment
        6. r_phead   = r_capture · r_head                coupled
        7. r_up      = mean((u_i_z + 1)/2)²              drone upright
        8. r_eff     = -w_eff · mean(throttle)           energy
        9. r_smooth  = -w_smooth · mean(Δthrottle)       smoothness
       10. r_sep     = (min_dist/d_safe)²                anti-collision factor

        r = r_sep · (w_cap·r_capture + w_head·r_phead + 0.3·r_stretch + w_up·r_up + r_eff + r_smooth)
        """
        target_pos = self.target_pos  # env-local, set by _reset_idx
        N, K = self.num_envs, self.drone.n

        # --- 1. XY alignment ---
        dist_xy = torch.norm(self.net_centre[..., :2] - target_pos[..., :2], dim=-1, keepdim=True)
        self.net_target_dist = torch.norm(self.net_centre - target_pos, dim=-1, keepdim=True)
        r_xy = torch.exp(-dist_xy * 2.0)

        # --- 2. Upward scooper ---
        z_diff = target_pos[..., 2:3] - self.net_centre[..., 2:3]
        r_z = torch.where(
            z_diff > 0,
            torch.exp(-z_diff * 2.0),   # net below: encourage climbing up
            torch.exp(z_diff * 1.0),     # net above: gentle decay, stay close
        )
        r_capture = r_xy * r_z

        # --- 3. Anti-collapse stretch ---
        # 6×6 net, spacing=0.25 → edge length = (6-1)*0.25 = 1.25
        # ideal diagonal = sqrt(1.25² + 1.25²) ≈ 1.7678
        diag_ideal = (self.net_rows - 1) * self.net_spacing * (2 ** 0.5)
        diag1 = torch.norm(self.corner_pos[:, 0] - self.corner_pos[:, 3], dim=-1, keepdim=True)
        diag2 = torch.norm(self.corner_pos[:, 1] - self.corner_pos[:, 2], dim=-1, keepdim=True)
        r_stretch = torch.exp(-torch.abs(diag1 - diag_ideal) * 2.0) * \
                    torch.exp(-torch.abs(diag2 - diag_ideal) * 2.0)

        # --- 4. Dynamic scooping: attitude + velocity alignment ---
        # 4a. Static attitude: net normal aligned with target heading
        attitude_align = (self.net_normal * self.target_heading_vec).sum(dim=-1, keepdim=True)
        r_attitude = torch.square((attitude_align + 1) / 2)

        # 4b. Dynamic velocity: net centre velocity direction aligned with target heading
        net_speed = torch.norm(self.net_centre_vel, dim=-1, keepdim=True).clamp(min=1e-5)
        net_vel_dir = self.net_centre_vel / net_speed
        vel_align = (net_vel_dir * self.target_heading_vec).sum(dim=-1, keepdim=True)
        r_vel_dir = torch.square((vel_align + 1) / 2)
        r_speed_scale = (net_speed / 1.0).clamp(0, 1)   # max score at ≥1 m/s

        # Combined: must be near target, face right, and MOVE in target direction
        r_scoop = r_capture * r_attitude * (0.5 + 0.5 * r_vel_dir * r_speed_scale)

        # --- 5. Auxiliary terms ---
        r_up = torch.square((self.drone.up[..., 2] + 1) / 2).mean(-1, keepdim=True)
        r_eff = -self.reward_effort_weight * self.effort.mean(-1, keepdim=True)
        r_smooth = -self.reward_action_smoothness_weight * \
            self.drone.throttle_difference.mean(-1, keepdim=True)
        sep = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
        r_sep = torch.square((sep / self.safe_distance).clamp(0, 1))

        # --- Total ---
        r_total = (
            self.reward_coverage_weight * r_scoop
            + 0.3 * r_stretch
            + self.reward_up_weight * r_up
            + r_eff
            + r_smooth
        ) * r_sep

        reward = torch.zeros(N, K, 1, device=self.device)
        reward[:] = r_total.reshape(N, 1, 1)

        # --- Termination ---
        misbehave = (
            (self.drone_states[..., 2] > 5.0).any(-1, keepdim=True)   # runaway
            | (self.net_target_dist > self.reset_thres)
        )
        hasnan = torch.isnan(self.drone_states).any(-1)
        terminated = misbehave | hasnan.any(-1, keepdim=True)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["return"].add_(reward.mean(1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)

        return TensorDict(
            {
                "agents": {"reward": reward},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

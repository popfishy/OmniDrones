# MIT License
#
# Copyright (c) 2026 Jiaqi Yang, University of Defence Technology
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

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
)

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
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
        self.n_nodes = self.net_rows * self.net_cols

        super().__init__(cfg, headless)

        self.group.initialize()

        # Target is independent — not under the group
        self.target = RigidPrimView(
            "/World/envs/.*/target",
            reset_xform_properties=False,
        )
        self.target.initialize()

        # Heading indicator view (visual-only, for pose updates)
        self.target_heading_view = RigidPrimView(
            "/World/envs/.*/target_heading",
            reset_xform_properties=False,
        )
        self.target_heading_view.initialize()

        # Cache initial drone velocities for reset
        self.init_drone_vels = torch.zeros_like(self.drone.get_velocities())

        # Target: 3D point ABOVE initial drone height so drones must fly upward.
        # Drone z ≈ 3.825 in env frame (3.0 group + 0.825 above net).
        # Target z ∈ [4.5, 5.5] gives 0.7–1.7 m of upward maneuvering room.
        self.target_pos_dist = D.Uniform(
            torch.tensor([-1.0, -1.0, 4.5], device=self.device),
            torch.tensor([1.0, 1.0, 5.5], device=self.device),
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

        self.group.spawn(translations=[(0, 0, 3.0)], enable_collision=True)

        # Visual-only target marker (small sphere, no physics, no collision)
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import UsdPhysics, UsdGeom, Gf

        target = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            prim_type="Sphere",
            translation=(0.0, 0.0, 2.0),
            attributes={"radius": 0.06},
        )
        UsdPhysics.RigidBodyAPI.Apply(target)
        UsdPhysics.CollisionAPI.Apply(target)
        kit_utils.set_collision_properties(target.GetPath(), collision_enabled=False)
        kit_utils.set_rigid_body_properties(target.GetPath(), disable_gravity=True)

        # Heading direction arrow: long capsule along X, rotated to heading in XY plane.
        arrow = UsdGeom.Capsule.Define(
            omni.usd.get_context().get_stage(),
            "/World/envs/env_0/target_heading",
        )
        arrow.CreateAxisAttr("X")       # capsule along X (in XY plane)
        arrow.CreateHeightAttr(0.3)     # shaft length 0.3m
        arrow.CreateRadiusAttr(0.015)   # thin shaft
        arrow.AddTranslateOp().Set(Gf.Vec3f(0, 0, 2.0))
        UsdPhysics.RigidBodyAPI.Apply(arrow.GetPrim())
        UsdPhysics.CollisionAPI.Apply(arrow.GetPrim())
        kit_utils.set_collision_properties(arrow.GetPrim().GetPath(), collision_enabled=False)
        kit_utils.set_rigid_body_properties(arrow.GetPrim().GetPath(), disable_gravity=True)

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
        # Sample target position
        target_pos = self.target_pos_dist.sample(env_ids.shape)
        self.target_pos[env_ids] = target_pos

        # Reset drone internal state (throttles, randomization)
        self.drone._reset_idx(env_ids)

        # Zero drone velocities
        self.drone.set_velocities(
            torch.zeros(len(env_ids) * self.drone.n, 6, device=self.device),
            env_indices=env_ids,
        )

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

        # Set target visual marker poses
        world_pos = target_pos + self.envs_positions[env_ids]
        self.target.set_world_poses(positions=world_pos, env_indices=env_ids)
        # Arrow: offset by half shaft length along heading so base is at target
        arrow_world_pos = world_pos + heading_vec * 0.15
        self.target_heading_view.set_world_poses(
            positions=arrow_world_pos,
            orientations=heading_quat,
            env_indices=env_ids,
        )

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
        corner_pos = torch.stack([
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
        v1 = corner_pos[:, 1] - corner_pos[:, 0]  # n02 - n00
        v2 = corner_pos[:, 2] - corner_pos[:, 0]  # n20 - n00
        self.net_normal = torch.nn.functional.normalize(
            torch.cross(v1, v2, dim=-1), dim=-1
        )  # (num_envs, 3)

        # Target position in env frame
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())

        # Relative positions
        drone_pos = self.drone_states[..., :3]  # (num_envs, n, 3)
        net_centre_rpos = self.net_centre.unsqueeze(1) - drone_pos  # drone → net centre
        corner_rpos = corner_pos.unsqueeze(2) - drone_pos.unsqueeze(1)  # (N, 4, n, 3)
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
        Reward function (shared across all K drones):

        Let:
          p_net  ∈ ℝ³ : net centre position
          n_net  ∈ ℝ³ : net normal (unit vector)
          p_tgt  ∈ ℝ³ : target position
          v_tgt  ∈ ℝ³ : target heading (unit, upward)
          p_i    ∈ ℝ³ : drone i position
          u_i    ∈ ℝ³ : drone i up vector (unit)

        1.  r_pos = exp(-β · ‖p_tgt - p_net‖)              position: exponential decay with distance
        2.  r_head = ((n_net · v_tgt + 1) / 2)²            heading: alignment of net normal → target direction
        3.  r_phead = r_pos · r_head                       coupled: heading only matters when close
        4.  r_up = mean_i[ ((u_i · e_z + 1) / 2)² ]       upright: drones stay level
        5.  r_eff = -(w_eff / K) Σ_i throttle_i            effort: energy penalty
        6.  r_sep = min(1, min_{i≠j} ‖p_i-p_j‖ / d_safe)²  separation: anti-collision factor
        7.  r_smooth = -(w_smooth / K) Σ_i ‖Δ throttle_i‖  smoothness: abrupt-change penalty

        Total (per timestep, same for all drones):
          r = r_sep · ( w_pos·r_pos + w_head·r_phead + w_up·r_up + r_eff + r_smooth )
        """
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())

        N, K = self.num_envs, self.drone.n

        # --- 1. Position ---
        dist = torch.norm(self.net_centre - target_pos, dim=-1, keepdim=True)  # (N, 1)
        self.net_target_dist = dist
        r_pos = torch.exp(-dist * self.reward_distance_scale)

        # --- 2. Heading alignment ---
        alignment = (self.net_normal * self.target_heading_vec).sum(dim=-1, keepdim=True)  # [-1, 1]
        r_head = torch.square((alignment + 1) / 2)   # [0, 1], 1 = perfect alignment
        r_phead = r_pos * r_head                      # coupled

        # --- 3. Upright ---
        r_up = torch.square((self.drone.up[..., 2] + 1) / 2).mean(-1, keepdim=True)  # (N, 1)

        # --- 4. Effort ---
        r_eff = -self.reward_effort_weight * self.effort.mean(-1, keepdim=True)  # (N, 1)

        # --- 5. Separation (multiplicative like TransportTrack) ---
        sep = self.drone_pdist.min(dim=-2).values.min(dim=-2).values  # (N, 1)
        r_sep = torch.square((sep / self.safe_distance).clamp(0, 1))

        # --- 6. Action smoothness ---
        r_smooth = -self.reward_action_smoothness_weight * \
            self.drone.throttle_difference.mean(-1, keepdim=True)  # (N, 1)

        # --- Total ---
        r_total = (
            self.reward_coverage_weight * r_pos
            + self.reward_descend_weight * r_phead
            + self.reward_up_weight * r_up
            + r_eff
            + r_smooth
        ) * r_sep

        reward = torch.zeros(N, K, 1, device=self.device)
        reward[:] = r_total.reshape(N, 1, 1)

        # --- Termination ---
        misbehave = (
            (self.drone_states[..., 2] < 0.2).any(-1, keepdim=True)
            | (dist > self.reset_thres)
        )
        hasnan = torch.isnan(self.drone_states).any(-1)

        terminated = misbehave | hasnan.any(-1, keepdim=True)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # --- Stats ---
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

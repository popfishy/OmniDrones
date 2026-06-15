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


import omni
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
)
from omni_drones.robots.drone import MultirotorBase

from .utils import NetCaptureGroup, NetCaptureCfg


class NetCapture(IsaacEnv):
    r"""
    A cooperative hover control task where a group of UAVs carry a flexible net
    connected via PBD particle ropes. The goal for the agents is to collaboratively
    control the net so that its centre hovers at a fixed target point in 3D space.

    Analogous to ``DragonHover`` but for a multi-drone tethered-net system.

    ## Observation

    The observation space contains the following items:

    - ``obs_self`` (1, \*): The state of each UAV observed by itself, containing
      its kinematic information with net-centre-relative position.
      It also includes a one-hot vector indicating each drone's identity.
    - ``obs_others`` (k-1, \*): The observed states of other agents.
    - ``obs_net`` (1, \*): Net centre + 4 corner positions relative to the drone.
    - ``obs_target`` (1, 3): Target hover position relative to the drone.

    ## Reward

    .. math::

        r = r_{pos} + r_{pos} \cdot w_{up} \cdot r_{up} + r_{eff} + r_{smooth}

    - ``r_pos``: :math:`\exp(-\beta \cdot \|p_{net} - p_{tgt}\|)` — net centre distance.
    - ``r_up``: :math:`mean((u_z + 1)/2)^2` — drone uprightness.
    - ``r_eff``: Energy penalty.
    - ``r_smooth``: Action smoothness penalty.

    ## Config

    | Parameter           | Type  | Default       | Description                         |
    | ------------------- | ----- | ------------- | ----------------------------------- |
    | ``drone_model``     | str   | "hummingbird" |                                     |
    | ``num_drones``      | int   | 4             |                                     |
    | ``net_rows``        | int   | 6             | Number of node rows in the net.     |
    | ``net_cols``        | int   | 6             | Number of node columns in the net.  |
    | ``net_spacing``     | float | 0.25          | Distance between adjacent nodes.    |
    | ``rope_links``      | int   | 16            | Number of particles per PBD rope.   |
    | ``reset_thres``     | float | 4.0           | Net drift that triggers reset.      |
    | ``action_scale``    | float | 1.0           | Scale applied to RL actions.        |
    """

    def __init__(self, cfg, headless):
        self.reward_up_weight = cfg.task.reward_up_weight
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reset_thres = cfg.task.reset_thres
        self.action_scale = cfg.task.get("action_scale", 1.0)

        # Must be set before super().__init__() because _set_specs() uses them
        self.net_rows = cfg.task.net_rows
        self.net_cols = cfg.task.net_cols
        self.net_spacing = cfg.task.net_spacing
        self.n_nodes = self.net_rows * self.net_cols

        super().__init__(cfg, headless)

        self.group.initialize()

        # Visual-only target marker (pure USD prim, NO RigidBody)
        self._target_prim_paths = [
            f"/World/envs/env_{i}/target" for i in range(self.num_envs)
        ]

        # Cache initial poses for GPU-compatible reset via RigidPrimView
        self.init_drone_pos, self.init_drone_rot = self.drone.get_world_poses(clone=True)
        self.init_rotor_pos, self.init_rotor_rot = self.drone.rotors_view.get_world_poses(clone=True)
        self.init_net_nodes_pos, self.init_net_nodes_rot = self.group.net_nodes_view.get_world_poses(clone=True)
        self.init_net_edges_pos, self.init_net_edges_rot = self.group.net_edges_view.get_world_poses(clone=True)
        if not self.cfg.task.get("use_pbd_rope", True):
            self.init_rope_segs_pos, self.init_rope_segs_rot = self.group.rope_segs_view.get_world_poses(clone=True)

        # Fixed target hover point (net centre target, env-local coordinates)
        target_pos_cfg = cfg.task.get("target_pos", [0.0, 0.0, 2.0])
        self.target_pos = torch.tensor(target_pos_cfg, device=self.device).repeat(self.num_envs, 1)

        # Random initial position offset (env-local, ±range from spawn point)
        init_offset_cfg = cfg.task.get("init_drone_offset", [1.0, 1.0, 0.5])
        self.init_pos_dist = D.Uniform(
            torch.tensor(init_offset_cfg, device=self.device) * -1,
            torch.tensor(init_offset_cfg, device=self.device),
        )
        self.alpha = 0.8

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        use_pbd = self.cfg.task.get("use_pbd_rope", True)
        group_cfg = NetCaptureCfg(
            num_drones=self.cfg.task.num_drones,
            net_rows=self.cfg.task.net_rows,
            net_cols=self.cfg.task.net_cols,
            net_spacing=self.cfg.task.net_spacing,
            rope_links=self.cfg.task.rope_links,
            use_pbd_rope=use_pbd,
            rope_link_length=self.cfg.task.get("rope_link_length", 0.1),
        )
        if use_pbd:
            for key in (
                "pbd_particle_mass", "pbd_stretch_stiffness", "pbd_bend_stiffness",
                "pbd_shear_stiffness", "pbd_spring_damping", "pbd_velocity_damping",
                "pbd_solver_iterations",
            ):
                if key in self.cfg.task:
                    setattr(group_cfg, key, self.cfg.task[key])

        self.group = NetCaptureGroup(drone=self.drone, cfg=group_cfg)

        scene_utils.design_scene()

        self.group.spawn(translations=[(0, 0, 0.5)], enable_collision=True)
        # Net at env z = 0.5, drones at 0.5 + 1.125 = 1.625
        # Target at z = 2.0 (fixed hover point for net centre)

        # Visual-only target marker (pure USD prim, NO RigidBody — avoids GPU API errors)
        # Position updated via USD API in _reset_idx.
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import UsdGeom

        target_path = "/World/envs/env_0/target"
        target = prim_utils.create_prim(
            prim_path=target_path,
            prim_type="Sphere",
            translation=(0.0, 0.0, 2.0),
            attributes={"radius": 0.06},
        )
        UsdGeom.Imageable(target).MakeVisible()

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1] + self.drone.n

        # Net observation per drone: net centre (6D) + 4 corners (4×6D) = 30D
        net_obs_dim = (1 + 4) * 6
        # Target observation per drone: relative position only (3D)
        target_obs_dim = 3

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
            "state_target": UnboundedContinuousTensorSpec((1, 3)),
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
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(self.drone.n),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        # Fixed target hover point (no per-episode sampling)
        target_pos = self.target_pos[env_ids]

        n = self.drone.n
        self.drone._reset_idx(env_ids)

        # Sample random initial position offset (env-local)
        offset = self.init_pos_dist.sample((len(env_ids),))  # (len(env_ids), 3)

        # ---- Reset drone bodies (base_link + rotors) atomically ----
        # Apply same random offset to the entire drone+net system to avoid
        # constraint inconsistency between FixedJoint-connected bodies.
        d_pos_init = self.init_drone_pos.reshape(self.num_envs, n, 3)[env_ids]  # (len, n, 3)
        d_pos = (d_pos_init + offset.unsqueeze(1)).reshape(-1, 3)
        d_rot = self.init_drone_rot.reshape(self.num_envs, n, 4)[env_ids].reshape(-1, 4)
        self.drone.set_world_poses(d_pos, d_rot, env_ids)

        rot_pos_init = self.init_rotor_pos.reshape(self.num_envs, n, self.drone.num_rotors, 3)[env_ids]
        rot_pos = (rot_pos_init + offset.unsqueeze(1).unsqueeze(2)).reshape(-1, 3)
        rot_rot = self.init_rotor_rot.reshape(self.num_envs, n, self.drone.num_rotors, 4)[env_ids]
        self.drone.rotors_view.set_world_poses(rot_pos, rot_rot, env_ids)

        # Zero drone velocities
        self.drone.set_velocities(
            torch.zeros(len(env_ids) * n, 6, device=self.device), env_ids)
        self.drone.rotors_view.set_velocities(
            torch.zeros(len(env_ids) * n * self.drone.num_rotors, 6, device=self.device), env_ids)

        # ---- Reset net nodes (offset to match drones) ----
        n_nodes = self.n_nodes
        n_ids = (env_ids.unsqueeze(-1) * n_nodes
                 + torch.arange(n_nodes, device=self.device)).reshape(-1)
        net_pos_init = self.init_net_nodes_pos.reshape(self.num_envs, n_nodes, 3)[env_ids]
        net_pos = (net_pos_init + offset.unsqueeze(1)).reshape(-1, 3)
        net_rot = self.init_net_nodes_rot.reshape(self.num_envs, n_nodes, 4)[env_ids].reshape(-1, 4)
        self.group.net_nodes_view.set_world_poses(net_pos, net_rot, n_ids)

        # ---- Reset net edge capsules ----
        n_edges = self.init_net_edges_pos.shape[0] // self.num_envs
        e_ids = (env_ids.unsqueeze(-1) * n_edges
                 + torch.arange(n_edges, device=self.device)).reshape(-1)
        edge_pos_init = self.init_net_edges_pos.reshape(self.num_envs, n_edges, 3)[env_ids]
        edge_pos = (edge_pos_init + offset.unsqueeze(1)).reshape(-1, 3)
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
            seg_pos_init = self.init_rope_segs_pos.reshape(self.num_envs, n_segs, 3)[env_ids]
            seg_pos = (seg_pos_init + offset.unsqueeze(1)).reshape(-1, 3)
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

        # Set target visual marker (fixed position, just copy to prim)
        target_pos_world = target_pos + self.envs_positions[env_ids]
        for idx, env_id in enumerate(env_ids.tolist()):
            tp = target_pos_world[idx].tolist()
            omni.isaac.core.utils.prims.set_prim_property(
                self._target_prim_paths[env_id], "xformOp:translate", Gf.Vec3f(*tp))

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

        # Target: relative position only (3D)
        obs["obs_target"] = target_rpos.unsqueeze(2)  # (num_envs, n, 1, 3)

        # ---- Assemble centralized critic observation ----
        state = TensorDict({}, self.num_envs)
        state["state_drones"] = obs["obs_self"].squeeze(2)  # (num_envs, n, state_dim)
        state["state_net"] = torch.cat([
            net_pos.reshape(self.num_envs, -1, 3),
            net_vel[..., :3].reshape(self.num_envs, -1, 3),
        ], dim=-1)  # (num_envs, n_nodes, 6)
        state["state_target"] = target_pos.unsqueeze(1)  # (num_envs, 1, 3)

        # ---- Stats ----
        self.net_target_dist = torch.norm(
            self.net_centre - target_pos, dim=-1, keepdim=True
        )

        self.stats["net_target_dist"].lerp_(self.net_target_dist, (1 - self.alpha))
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
        r"""
        DragonHover-style reward for multi-drone net hover control.

        .. math::

            r = r_{pos} + r_{pos} \cdot w_{up} \cdot r_{up} + r_{eff} + r_{smooth}

        - r_pos  = exp(-β · ‖p_net - p_tgt‖)   net centre distance
        - r_up   = mean((u_z + 1)/2)²          drone uprightness
        - r_eff  = -w_eff · mean(throttle)      energy penalty
        - r_smooth = -w_smooth · mean(Δthrottle) action smoothness
        """
        target_pos = self.target_pos  # env-local, fixed
        N, K = self.num_envs, self.drone.n

        # Net centre distance to target
        self.net_target_dist = torch.norm(
            self.net_centre - target_pos, dim=-1, keepdim=True
        )
        r_pos = torch.exp(-self.net_target_dist * self.reward_distance_scale)

        # Uprightness: mean of drone body z-axis alignment with world Z
        r_up = torch.square((self.drone.up[..., 2] + 1) / 2).mean(-1, keepdim=True)

        # Energy penalty
        r_eff = -self.reward_effort_weight * self.effort.mean(-1, keepdim=True)

        # Action smoothness
        r_smooth = -self.reward_action_smoothness_weight * \
            self.drone.throttle_difference.mean(-1, keepdim=True)

        # Total: pose reward multiplicatively gates uprightness bonus
        r_total = r_pos + r_pos * self.reward_up_weight * r_up + r_eff + r_smooth

        reward = torch.zeros(N, K, 1, device=self.device)
        reward[:] = r_total.reshape(N, 1, 1)

        # --- Termination ---
        misbehave = (
            (self.drone_states[..., 2] < 0.2).any(-1, keepdim=True)    # crash
            | (self.drone_states[..., 2] > 5.0).any(-1, keepdim=True)   # runaway
            | (self.net_target_dist > self.reset_thres)                  # net drifts too far
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

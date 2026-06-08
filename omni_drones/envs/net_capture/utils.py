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


from typing import Sequence, Union

import omni.isaac.core.utils.prims as prim_utils
import torch

from omni_drones.views import RigidPrimView, ArticulationView
from pxr import UsdPhysics

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.robots.drone import MultirotorBase
from dataclasses import dataclass


@dataclass
class NetCaptureCfg(RobotCfg):
    num_drones: int = 4
    net_rows: int = 6
    net_cols: int = 6
    net_spacing: float = 0.25
    rope_links: int = 12
    rope_link_length: float = 0.1
    node_mass: float = 0.01
    corner_mass: float = 0.02

    def __post_init__(self):
        if self.num_drones not in (4, 6):
            raise ValueError("num_drones must be 4 or 6.")


class NetCaptureGroup(RobotBase):
    """
    Spanning-tree articulation architecture:

      drone_i  (ArticulationRoot):  base_link + rotors + rope segs
        └── rope seg_11 ──Fixed(excl)── net/node_corner

      net      (ArticulationRoot):  36 nodes + 35 tree edges
        + 25 loop-closing D6 joints (excludeFromArticulation=True)

    Reset uses ArticulationView.set_world_poses() — GPU compatible.
    """

    def __init__(
        self,
        name: str = "Group",
        drone: Union[str, MultirotorBase] = "Firefly",
        cfg: NetCaptureCfg = None,
        is_articulation: bool = False,
    ) -> None:
        super().__init__(name, cfg, is_articulation)
        if isinstance(drone, str):
            drone = MultirotorBase.REGISTRY[drone]()
        # Drone keeps its own ArticulationRoot — rope segments will be added
        # as articulation links so the whole drone+rope tree resets together.
        drone.is_articulation = True
        self.drone = drone
        self.translations = []

        self.num_drones = cfg.num_drones
        self.net_rows = cfg.net_rows
        self.net_cols = cfg.net_cols
        self.net_spacing = cfg.net_spacing
        self.rope_links = cfg.rope_links
        self.rope_link_length = cfg.rope_link_length
        self.node_mass = cfg.node_mass
        self.corner_mass = cfg.corner_mass

    def spawn(
        self,
        translations=(0, 0, 0),
        prim_paths: Sequence[str] = None,
        enable_collision: bool = False,
    ):
        translations = torch.atleast_2d(
            torch.as_tensor(translations, device=self.device)
        )
        self.translations.extend(translations.tolist())
        n = translations.shape[0]

        if prim_paths is None:
            prim_paths = [f"/World/envs/env_0/{self.name}_{i}" for i in range(n)]

        prims = []
        for prim_path, translation in zip(prim_paths, translations):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            xform = prim_utils.create_prim(prim_path, translation=translation)

            # ---- 1.  Net Articulation ----
            net_info = scene_utils.create_net(
                xform_path=f"{prim_path}/net",
                rows=self.net_rows,
                cols=self.net_cols,
                spacing=self.net_spacing,
                node_mass=self.node_mass,
                corner_mass=self.corner_mass,
                enable_collision=enable_collision,
            )
            self.net_info = net_info

            # ---- 2.  Corner map + drone positions ----
            corner_indices = [
                (0, 0),
                (0, self.net_cols - 1),
                (self.net_rows - 1, 0),
                (self.net_rows - 1, self.net_cols - 1),
            ]
            z_drone = (self.rope_links - 1) * self.rope_link_length * 0.75
            drone_translations = self._compute_drone_positions(corner_indices, z_drone)

            # ---- 3.  Drones (articulations) + ropes (independent) ----
            for i in range(self.num_drones):
                drone_path = f"/World/envs/env_0/{self.drone.name.lower()}_{i}"
                self.drone.spawn(
                    translations=translation + drone_translations[i],
                    prim_paths=[drone_path],
                )

                r, c = corner_indices[i]
                corner_node = net_info["nodes"][r][c]
                drone_base_link = f"{drone_path}/base_link"

                # Rope as independent RigidBodies under GROUP (not inside drone
                # articulation).  Both ends use excludeFromArticulation=True:
                # drone is its own articulation, net is an articulation.
                rope_translation = drone_translations[i].tolist()
                scene_utils.create_rope(
                    xform_path=f"{prim_path}/rope_{i}",
                    translation=rope_translation,
                    from_prim=corner_node,                # net corner ↔ links[-1]
                    to_prim=drone_base_link,              # drone     ↔ links[0]
                    num_links=self.rope_links,
                    link_length=self.rope_link_length,
                    color=(0.4, 0.2, 0.1),
                    enable_collision=False,
                    exclude_from_articulation=True,
                )

            prims.append(xform)

        self.n += n
        return prims

    def _compute_drone_positions(self, corner_indices, z_drone: float = 1.0):
        x_offset = -(self.net_cols - 1) * self.net_spacing / 2.0
        y_offset = (self.net_rows - 1) * self.net_spacing / 2.0

        positions = []
        for r, c in corner_indices:
            x = x_offset + c * self.net_spacing
            y = y_offset - r * self.net_spacing
            positions.append([x, y, z_drone])

        return torch.tensor(positions, device=self.device)

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        if prim_paths_expr is None:
            prim_paths_expr = f"/World/envs/.*/{self.name}_.*"
        self.prim_paths_expr = prim_paths_expr

        # Drone ArticulationView (includes rope segments as articulation links)
        self.drone.initialize(f"/World/envs/.*/{self.drone.name.lower()}_*")

        # Net ArticulationView — for GPU-compatible reset
        self.net_articulation = ArticulationView(
            f"{self.prim_paths_expr}/net",
            reset_xform_properties=False,
            shape=(-1, 1),
        )
        self.net_articulation.initialize()

        # RigidPrimView for reading net node positions (observation only, no set)
        self.net_nodes_view = RigidPrimView(
            f"{self.prim_paths_expr}/net/node_*",
            reset_xform_properties=False,
        )
        self.net_nodes_view.initialize()

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        return self.drone.apply_action(actions)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        return env_ids

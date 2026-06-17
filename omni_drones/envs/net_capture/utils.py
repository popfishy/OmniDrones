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


from typing import Sequence, Union

import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import torch

from omni.isaac.core.prims import RigidPrimView
from pxr import UsdPhysics

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import quat_axis
from dataclasses import dataclass


@dataclass
class NetCaptureCfg(RobotCfg):
    num_drones: int = 4
    net_rows: int = 3
    net_cols: int = 3
    net_spacing: float = 0.5
    rope_links: int = 12
    rope_link_length: float = 0.06
    node_mass: float = 0.01
    corner_mass: float = 0.02

    def __post_init__(self):
        if self.num_drones not in (4, 6):
            raise ValueError("num_drones must be 4 or 6.")


class NetCaptureGroup(RobotBase):
    """
    Group of drones + net + ropes as INDEPENDENT rigid bodies (NOT a PhysX articulation).

    The 2D net grid contains cycles (quadrilateral loops), which are forbidden
    in PhysX articulations.  Each drone keeps its own ArticulationRoot; ropes and
    net nodes/edges are standalone RigidBodies connected by ordinary D6 joints.
    """

    def __init__(
        self,
        name: str = "Group",
        drone: Union[str, MultirotorBase] = "Firefly",
        cfg: NetCaptureCfg = None,
        is_articulation: bool = False,  # group is NOT an articulation
    ) -> None:
        super().__init__(name, cfg, is_articulation)
        if isinstance(drone, str):
            drone = MultirotorBase.REGISTRY[drone]()
        # Drone uses RigidPrimView (not ArticulationView) for state/forces,
        # matching the TransportationGroup pattern.  The drone USD still has
        # ArticulationRootAPI for the rotor joints, but the Python wrapper
        # interacts through RigidPrimView + applied forces.
        drone.is_articulation = False
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

        self.alpha = 0.9

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
            xform = prim_utils.create_prim(
                prim_path,
                translation=translation,
            )

            # ---- 1. Create net (standalone RigidBodies + D6 joints) ----
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

            # ---- 2. Compute drone positions (directly above each corner) ----
            corner_indices = [
                (0, 0),                                 # drone_0 → n(0, 0)
                (0, self.net_cols - 1),                 # drone_1 → n(0, C-1)
                (self.net_rows - 1, 0),                 # drone_2 → n(R-1, 0)
                (self.net_rows - 1, self.net_cols - 1), # drone_3 → n(R-1, C-1)
            ]

            # Rope chain effective length:
            #   (N-1) * (link_length - link_length/4) = (N-1) * link_length * 0.75
            # Match drone height to this so the chain naturally reaches the net
            # with ZERO initial tension in the FixedJoints.
            z_drone = (self.rope_links - 1) * self.rope_link_length * 0.75
            drone_translations = self._compute_drone_positions(corner_indices, z_drone)

            # ---- 3. Spawn drones (each is its own articulation) + ropes ----
            for i in range(self.num_drones):
                drone_path = f"/World/envs/env_0/{self.drone.name.lower()}_{i}"
                drone_prim = self.drone.spawn(
                    translations=translation + drone_translations[i],
                    prim_paths=[drone_path],
                )[0]

                r, c = corner_indices[i]
                corner_node = net_info["nodes"][r][c]
                drone_base_link = f"{drone_path}/base_link"

                # Rope xform at drone position.  The chain extends along local X
                # which (after xform rotation of 90° Y) points downward (world -Z).
                #   links[0]  = at xform origin (drone height)
                #   links[-1] = at bottom of chain
                #   to_prim   = FixedJoint → links[0]  (drone end)
                #   from_prim = FixedJoint → links[-1] (net-corner end)
                # We swap the original from/to semantics so the corner connects
                # to the bottom of the dangling chain.
                rope_translation = drone_translations[i].tolist()
                scene_utils.create_rope(
                    xform_path=f"{prim_path}/rope_{i}",
                    translation=rope_translation,
                    from_prim=corner_node,          # net corner ↔ links[-1] (bottom)
                    to_prim=drone_base_link,        # drone      ↔ links[0]  (top)
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
        """Place each drone directly above its assigned net corner."""
        x_offset = -(self.net_cols - 1) * self.net_spacing / 2.0
        y_offset = (self.net_rows - 1) * self.net_spacing / 2.0

        positions = []
        for r, c in corner_indices:
            x = x_offset + c * self.net_spacing
            y = y_offset - r * self.net_spacing
            positions.append([x, y, z_drone])

        return torch.tensor(positions, device=self.device)

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        # Resolve group prim_paths_expr (for net_nodes_view)
        if prim_paths_expr is None:
            prim_paths_expr = f"/World/envs/.*/{self.name}_.*"
        self.prim_paths_expr = prim_paths_expr

        # Drones live at env level, not under the group
        self.drone.initialize(f"/World/envs/.*/{self.drone.name.lower()}_*")

        # RigidPrimView for net nodes (under the group)
        self.net_nodes_view = RigidPrimView(
            f"{self.prim_paths_expr}/net/node_*",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
        )
        self.net_nodes_view.initialize()

        # RigidPrimView for net edge capsules
        self.net_edges_view = RigidPrimView(
            f"{self.prim_paths_expr}/net/edge_*/capsule",
            reset_xform_properties=False,
        )
        self.net_edges_view.initialize()

        # RigidPrimView for D6 rope segments
        self.rope_segs_view = RigidPrimView(
            f"{self.prim_paths_expr}/rope_*/seg_*",
            reset_xform_properties=False,
        )
        self.rope_segs_view.initialize()

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        return self.drone.apply_action(actions)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        return env_ids

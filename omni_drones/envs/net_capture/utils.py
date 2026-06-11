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

from omni_drones.views import RigidPrimView
from pxr import UsdPhysics, PhysxSchema

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.core.utils.stage import get_current_stage
from omni.kit.commands import execute
from dataclasses import dataclass


@dataclass
class NetCaptureCfg(RobotCfg):
    num_drones: int = 4
    net_rows: int = 6
    net_cols: int = 6
    net_spacing: float = 0.25
    rope_links: int = 16          # more segments → smoother rope curvature
    rope_link_length: float = 0.1
    node_mass: float = 0.01
    corner_mass: float = 0.02

    # PBD particle rope (GPU-native, NO D6 joints)
    use_pbd_rope: bool = True
    pbd_particle_mass: float = 0.01
    pbd_stretch_stiffness: float = 1e6   # high → rope-like (not rubber band)
    pbd_bend_stiffness: float = 1.0      # low  → soft/flexible (not stiff wire)
    pbd_spring_damping: float = 0.2
    pbd_solver_iterations: int = 32      # higher → stiff constraints converge

    def __post_init__(self):
        if self.num_drones not in (4, 6):
            raise ValueError("num_drones must be 4 or 6.")


def _strip_articulation(drone_prim_path: str):
    """Remove articulation schemas and set excludeFromArticulation on all
    joints under *drone_prim_path* so the drone behaves as standalone
    RigidBodies (no implicit articulations).

    Called after *drone.spawn()* to undo the ArticulationRootAPI baked into
    the USD asset, preventing PhysX from creating implicit articulations that
    legally require CPU-side APIs (PxRigidDynamic::setGlobalPose /
    setLinearVelocity / setAngularVelocity) which are *illegal* with
    eENABLE_DIRECT_GPU_API.
    """
    stage = get_current_stage()
    drone_prim = stage.GetPrimAtPath(drone_prim_path)
    if not drone_prim:
        return

    # 1. Remove ArticulationRootAPI
    if drone_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        execute("UnapplyAPISchema", api=UsdPhysics.ArticulationRootAPI, prim=drone_prim)
    if drone_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        execute("UnapplyAPISchema", api=PhysxSchema.PhysxArticulationAPI, prim=drone_prim)

    # 2. Set excludeFromArticulation=True on ALL joints under the drone.
    #    Without ArticulationRootAPI these joints would form implicit
    #    articulations → same CPU-API-illegal problem.
    for prim in stage.Traverse():
        if not prim.GetPath().pathString.startswith(drone_prim_path):
            continue
        if prim.IsA(UsdPhysics.Joint) or prim.GetTypeName() == "PhysicsJoint":
            prim.GetAttribute("physics:excludeFromArticulation").Set(True)


class NetCaptureGroup(RobotBase):
    """
    Architecture:

      drone_i  (standalone RigidBodies — ArticulationRootAPI removed,
                 all joint excludeFromArticulation=True)
        └── base_link + 4 rotors (FixedJoints → standalone constraints)

      rope_i   (standalone RigidBodies — all D6 joints have
                 excludeFromArticulation=True)
        └── compliant D6 (5e2 stiffness) → net corner node
        └── compliant D6 (5e2 stiffness) → drone base_link

      net      (standalone RigidBodies — 36 nodes + 60 edge capsules,
                 all D6 joints have excludeFromArticulation=True)

    ALL bodies are standalone RigidBodies — the drone's internal USD
    ArticulationRootAPI is stripped at spawn time.  This means
    RigidPrimView.set_world_poses (GPU API: _physics_view.set_transforms)
    works on every body without triggering illegal CPU API calls.
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
        # Drone bodies are standalone RigidBodies (ArticulationRootAPI stripped
        # at spawn time).  RigidPrimView uses GPU API for set_world_poses /
        # set_velocities — no CPU API calls.
        drone.is_articulation = False
        self.drone = drone
        self.translations = []
        self._rope_infos = []  # PBD rope info dicts

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

            # ---- 1.  Net ----
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

            # ---- 3.  Drones (standalone RigidBodies) + ropes ----
            for i in range(self.num_drones):
                drone_path = f"/World/envs/env_0/{self.drone.name.lower()}_{i}"
                self.drone.spawn(
                    translations=translation + drone_translations[i],
                    prim_paths=[drone_path],
                )

                # Strip articulation schemas so ALL drone bodies are
                # standalone RigidBodies — no CPU-API-illegal calls.
                _strip_articulation(drone_path)

                r, c = corner_indices[i]
                corner_node_path = str(net_info["nodes"][r][c].GetPath())
                drone_base_link = f"{drone_path}/base_link"

                drone_offset = drone_translations[i]
                # Net corner: same XY as drone, but Z=0 in group-local frame
                x_offset = -(self.net_cols - 1) * self.net_spacing / 2.0
                y_offset = (self.net_rows - 1) * self.net_spacing / 2.0
                cx = x_offset + c * self.net_spacing
                cy = y_offset - r * self.net_spacing

                if self.cfg.use_pbd_rope:
                    # PBD particle rope — GPU native, no D6 joints
                    # Compute start/end positions so rope endpoints EXACTLY
                    # match drone base_link and net corner positions.
                    # Positions are in Group_0-local coords (ropeMesh is a
                    # child of Group_0 xform — no extra translate op).
                    # This ensures PhysxAutoAttachmentAPI can find the
                    # closest particles via proximity search.
                    start_pos = drone_offset.tolist()     # (cx, cy, z_drone)
                    end_pos = [cx, cy, 0.]                # net z=0 in group frame
                    ps_path = f"{prim_path}/particleSystem"
                    rope_info = scene_utils.create_pbd_rope(
                        xform_path=f"{prim_path}/rope_pbd_{i}",
                        start_pos=start_pos,
                        end_pos=end_pos,
                        particle_system_path=ps_path,
                        from_prim=corner_node_path,
                        to_prim=drone_base_link,
                        num_particles=self.rope_links,
                        particle_mass=self.cfg.pbd_particle_mass,
                        stretch_stiffness=self.cfg.pbd_stretch_stiffness,
                        bend_stiffness=self.cfg.pbd_bend_stiffness,
                        spring_damping=self.cfg.pbd_spring_damping,
                        solver_position_iterations=self.cfg.pbd_solver_iterations,
                    )
                    self._rope_infos.append(rope_info)
                else:
                    # Legacy D6 joint rope
                    scene_utils.create_rope(
                        xform_path=f"{prim_path}/rope_{i}",
                        translation=drone_offset.tolist(),
                        from_prim=corner_node_path,
                        to_prim=drone_base_link,
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

        # Drones use RigidPrimView (GPU-compatible custom override).
        # With ArticulationRootAPI stripped and all joints having
        # excludeFromArticulation=True, no CPU API calls are triggered.
        self.drone.initialize(f"/World/envs/.*/{self.drone.name.lower()}_*")

        # Net + rope: standalone RigidPrimViews (GPU API).
        self.net_nodes_view = RigidPrimView(
            f"{self.prim_paths_expr}/net/node_*",
            reset_xform_properties=False,
        )
        self.net_nodes_view.initialize()

        self.net_edges_view = RigidPrimView(
            f"{self.prim_paths_expr}/net/edge_*/capsule",
            reset_xform_properties=False,
        )
        self.net_edges_view.initialize()

        if self.cfg.use_pbd_rope:
            from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
            self.rope_cloth_view = ClothPrimView(
                f"{self.prim_paths_expr}/rope_pbd_*/ropeMesh",
                particle_systems=f"{self.prim_paths_expr}/particleSystem",
                name="rope_pbd_view",
            )
            self.rope_cloth_view.initialize()
        else:
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

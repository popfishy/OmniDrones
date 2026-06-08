# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
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


from typing import Sequence, Union, Optional

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.physx.scripts.utils as script_utils
import torch

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

import omni_drones.utils.kit as kit_utils


def design_scene():
    kit_utils.create_ground_plane(
        "/World/defaultGroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
        improve_patch_friction=True,
    )
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
    )


def create_rope(
    xform_path: str = "/World/rope",
    translation=(0, 0, 0),
    from_prim: Union[str, Usd.Prim] = None,
    to_prim: Union[str, Usd.Prim] = None,
    num_links: int = 24,
    link_length: float = 0.06,
    rope_damping: float = 10.0,
    rope_stiffness: float = 1.0,
    color=(0.4, 0.2, 0.1),
    enable_collision: bool = False,
    exclude_from_articulation: bool = True,
):
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()

    stage = stage_utils.get_current_stage()
    ropeXform = UsdGeom.Xform.Define(stage, xform_path)
    ropeXform.AddTranslateOp().Set(Gf.Vec3f(*translation))
    ropeXform.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))
    link_radius = 0.02
    joint_offset = link_length / 2 - link_length / 8

    links = []
    for i in range(num_links):
        link_path = f"{xform_path}/seg_{i}"
        location = (i * (link_length - link_length / 4), 0, 0)

        capsuleGeom = UsdGeom.Capsule.Define(stage, link_path)
        capsuleGeom.CreateHeightAttr(link_length / 2)
        capsuleGeom.CreateRadiusAttr(link_radius)
        capsuleGeom.CreateAxisAttr("X")
        capsuleGeom.AddTranslateOp().Set(location)
        capsuleGeom.AddOrientOp().Set(Gf.Quatf(1.0))
        capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        capsuleGeom.CreateDisplayColorAttr().Set([color])

        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        massAPI.CreateMassAttr().Set(0.01)

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        # physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        # physxCollisionAPI.CreateContactOffsetAttr().Set(0.02)
        capsuleGeom.GetPrim().GetAttribute("physics:collisionEnabled")

        if len(links) > 0:
            # jointPath = f"{link_path}/joint_{i}"
            # joint = UsdPhysics.Joint.Define(stage, jointPath)
            # joint.CreateBody0Rel().SetTargets([links[-1].GetPath()])
            # joint.CreateBody1Rel().SetTargets([link_path])

            # joint.CreateLocalPos0Attr().Set(Gf.Vec3f(joint_offset, 0, 0))
            # joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
            # joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-joint_offset, 0, 0))
            # joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

            # # locked DOF (lock - low is greater than high)
            # d6Prim = joint.GetPrim()
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)
            # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "rotX")
            # limitAPI.CreateLowAttr(1.0)
            # limitAPI.CreateHighAttr(-1.0)

            # # Moving DOF:
            # dofs = ["rotY", "rotZ"]
            # for d in dofs:
            #     limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, d)
            #     limitAPI.CreateLowAttr(-110)
            #     limitAPI.CreateHighAttr(110)

            #     # joint drives for rope dynamics:
            #     driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, d)
            #     driveAPI.CreateTypeAttr("force")
            #     driveAPI.CreateDampingAttr(rope_damping)
            #     driveAPI.CreateStiffnessAttr(rope_stiffness)
            joint: Usd.Prim = script_utils.createJoint(
                stage, "D6", links[-1], capsuleGeom.GetPrim()
            )
            joint.GetAttribute("physics:localPos0").Set((joint_offset, 0.0, 0.0))
            joint.GetAttribute("physics:localPos1").Set((-joint_offset, 0.0, 0.0))
            joint.GetAttribute("limit:rotY:physics:low").Set(-110)
            joint.GetAttribute("limit:rotY:physics:high").Set(110)
            joint.GetAttribute("limit:rotZ:physics:low").Set(-110)
            joint.GetAttribute("limit:rotZ:physics:high").Set(110)
            UsdPhysics.DriveAPI.Apply(joint, "rotY")
            UsdPhysics.DriveAPI.Apply(joint, "rotZ")
            joint.GetAttribute("drive:rotY:physics:damping").Set(rope_damping)
            joint.GetAttribute("drive:rotY:physics:stiffness").Set(rope_stiffness)
            joint.GetAttribute("drive:rotZ:physics:damping").Set(rope_damping)
            joint.GetAttribute("drive:rotZ:physics:stiffness").Set(rope_stiffness)

        links.append(capsuleGeom.GetPrim())

    if from_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "Fixed", from_prim, links[-1])
        joint.GetAttribute("physics:excludeFromArticulation").Set(exclude_from_articulation)

    if to_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "Fixed", links[0], to_prim)
        joint.GetAttribute("physics:excludeFromArticulation").Set(exclude_from_articulation)

    return links


def create_bar(
    prim_path: str,
    length: float,
    translation=(0, 0, 0),
    from_prim: str = None,
    to_prim: str = None,
    mass: float = 0.02,
    enable_collision=False,
    color=(0.4, 0.4, 0.2),
):
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()

    stage = stage_utils.get_current_stage()

    capsuleGeom = UsdGeom.Capsule.Define(stage, f"{prim_path}/Capsule")
    capsuleGeom.CreateHeightAttr(length)
    capsuleGeom.CreateRadiusAttr(0.012)
    capsuleGeom.CreateAxisAttr("Z")
    capsuleGeom.AddTranslateOp().Set(Gf.Vec3f(*translation))
    capsuleGeom.AddOrientOp().Set(Gf.Quatf(1.0))
    capsuleGeom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    capsuleGeom.CreateDisplayColorAttr().Set([color])

    UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
    massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
    massAPI.CreateMassAttr().Set(mass)

    UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
    prim: Usd.Prim = capsuleGeom.GetPrim()
    prim.GetAttribute("physics:collisionEnabled").Set(enable_collision)

    if from_prim is not None:
        sphere = prim_utils.create_prim(
            f"{prim_path}/Sphere",
            "Sphere",
            translation=(0, 0, -length),
            attributes={"radius": 0.02},
        )
        UsdPhysics.RigidBodyAPI.Apply(sphere)
        UsdPhysics.CollisionAPI.Apply(sphere)
        sphere.GetAttribute("physics:collisionEnabled").Set(False)

        script_utils.createJoint(stage, "Fixed", from_prim, sphere)
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", prim, sphere)
        joint.GetAttribute("limit:rotX:physics:low").Set(-120)
        joint.GetAttribute("limit:rotX:physics:high").Set(120)
        joint.GetAttribute("limit:rotY:physics:low").Set(-120)
        joint.GetAttribute("limit:rotY:physics:high").Set(120)
        UsdPhysics.DriveAPI.Apply(joint, "rotX")
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        joint.GetAttribute("drive:rotX:physics:damping").Set(0.0002)
        joint.GetAttribute("drive:rotY:physics:damping").Set(0.0002)

    if to_prim is not None:
        joint: Usd.Prim = script_utils.createJoint(stage, "D6", prim, to_prim)
        joint.GetAttribute("limit:rotX:physics:low").Set(-120)
        joint.GetAttribute("limit:rotX:physics:high").Set(120)
        joint.GetAttribute("limit:rotY:physics:low").Set(-120)
        joint.GetAttribute("limit:rotY:physics:high").Set(120)
        UsdPhysics.DriveAPI.Apply(joint, "rotX")
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        joint.GetAttribute("drive:rotX:physics:damping").Set(0.0002)
        joint.GetAttribute("drive:rotY:physics:damping").Set(0.0002)

    return prim


def create_net(
    xform_path: str = "/World/net",
    rows: int = 5,
    cols: int = 5,
    spacing: float = 0.5,
    node_radius: float = 0.02,
    node_mass: float = 0.01,
    corner_mass: float = 0.02,
    edge_damping: float = 10.0,
    edge_stiffness: float = 1.0,
    color: tuple = (0.3, 0.3, 0.3),
    enable_collision: bool = True,
) -> dict:
    """Create a 2D net grid as a single PhysX Articulation.

    Direct D6 joints between adjacent node pairs (no intermediate capsule
    bodies).  Snake spanning tree: row-0 L→R, then down to row-1 R→L, etc.
    Loop-closing vertical D6 joints use excludeFromArticulation=True.

    Visual-only capsules are placed for appearance (no RigidBody).

    Returns:
        dict with ``nodes`` (list of lists of Usd.Prim) keyed by (r,c).
    """
    stage = stage_utils.get_current_stage()

    # ---- 1.  Articulation root ----
    net_root = UsdGeom.Xform.Define(stage, xform_path)
    net_root.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
    UsdPhysics.ArticulationRootAPI.Apply(net_root.GetPrim())
    PhysxSchema.PhysxArticulationAPI.Apply(net_root.GetPrim())

    # Centre the net around origin in XY
    x_offset = -(cols - 1) * spacing / 2.0
    y_offset = (rows - 1) * spacing / 2.0
    half_sp = spacing / 2.0

    # ---- 2.  Create nodes (articulation links) ----
    node_prims: list = []
    for r in range(rows):
        row_prims: list = []
        for c in range(cols):
            node_path = f"{xform_path}/node_{r}_{c}"
            pos_x = x_offset + c * spacing
            pos_y = y_offset - r * spacing

            sphere = UsdGeom.Sphere.Define(stage, node_path)
            sphere.CreateRadiusAttr(node_radius)
            sphere.AddTranslateOp().Set(Gf.Vec3f(pos_x, pos_y, 0))
            sphere.CreateDisplayColorAttr().Set([color])

            script_utils.setRigidBody(sphere.GetPrim(), "convexHull", False)

            is_corner = (r == 0 or r == rows - 1) and (c == 0 or c == cols - 1)
            mass_val = corner_mass if is_corner else node_mass
            mass_api = UsdPhysics.MassAPI.Apply(sphere.GetPrim())
            mass_api.CreateMassAttr().Set(mass_val)

            sphere.GetPrim().GetAttribute("physics:collisionEnabled").Set(enable_collision)
            row_prims.append(sphere.GetPrim())
        node_prims.append(row_prims)

    link_radius = 0.015

    # Helper: D6 joint config (rotY/Z drives, trans+rotX locked)
    def _config_joint(joint: Usd.Prim):
        for dof in ("transX", "transY", "transZ", "rotX"):
            limit_api = UsdPhysics.LimitAPI.Apply(joint, dof)
            limit_api.CreateLowAttr(1.0)
            limit_api.CreateHighAttr(-1.0)
        for dof in ("rotY", "rotZ"):
            limit_api = UsdPhysics.LimitAPI.Apply(joint, dof)
            limit_api.CreateLowAttr(-110)
            limit_api.CreateHighAttr(110)
            drive_api = UsdPhysics.DriveAPI.Apply(joint, dof)
            drive_api.CreateTypeAttr("force")
            drive_api.CreateDampingAttr(edge_damping)
            drive_api.CreateStiffnessAttr(edge_stiffness)

    # Helper: direct D6 joint between two adjacent nodes (no capsule body).
    def _direct_joint(a, b, pos0, pos1, exclude=False):
        j = script_utils.createJoint(stage, "D6", a, b)
        j.GetAttribute("physics:localPos0").Set(pos0)
        j.GetAttribute("physics:localPos1").Set(pos1)
        if exclude:
            j.GetAttribute("physics:excludeFromArticulation").Set(True)
        _config_joint(j)

    # Helper: visual-only capsule (no RigidBody).
    def _vis_capsule(cx, cy, rot_z, tag):
        edge_path = f"{xform_path}/{tag}"
        xf = UsdGeom.Xform.Define(stage, edge_path)
        xf.AddTranslateOp().Set(Gf.Vec3f(cx, cy, 0))
        if rot_z:
            xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 90))
        cap = UsdGeom.Capsule.Define(stage, f"{edge_path}/capsule_vis")
        cap.CreateHeightAttr(spacing)
        cap.CreateRadiusAttr(link_radius)
        cap.CreateAxisAttr("X")
        cap.CreateDisplayColorAttr().Set([color])

    # ---- 3.  Spanning-tree D6 joints (snake: row 0 L→R, row 1 R→L, ...) ----
    h_pos0 = (half_sp, 0.0, 0.0)      # left  node → joint at right  edge
    h_pos1 = (-half_sp, 0.0, 0.0)     # right node → joint at left   edge
    v_pos0 = (0.0, -half_sp, 0.0)     # upper node → joint at bottom edge
    v_pos1 = (0.0, half_sp, 0.0)      # lower node → joint at top    edge

    for r in range(rows):
        c_range = range(cols - 1)
        if r % 2 == 0:
            # Even row: left to right
            for c in c_range:
                a, b = node_prims[r][c], node_prims[r][c + 1]
                _direct_joint(a, b, h_pos0, h_pos1)
                _vis_capsule(x_offset + c * spacing + half_sp,
                             y_offset - r * spacing, False, f"edge_h_{r}_{c}")
        else:
            # Odd row: right to left
            for c in c_range:
                a, b = node_prims[r][c + 1], node_prims[r][c]
                _direct_joint(a, b, h_pos0, h_pos1)
                _vis_capsule(x_offset + c * spacing + half_sp,
                             y_offset - r * spacing, False, f"edge_h_{r}_{c}")
        # Vertical connector to next row (except last row)
        if r < rows - 1:
            col_last = cols - 1 if r % 2 == 1 else 0
            a = node_prims[r][col_last]
            b = node_prims[r + 1][col_last]
            _direct_joint(a, b, v_pos0, v_pos1)
            _vis_capsule(x_offset + col_last * spacing,
                         y_offset - r * spacing - half_sp, True,
                         f"edge_v_conn_{r}")

    # ---- 4.  Loop-closing D6 joints (excludeFromArticulation) ----
    loop_pos0 = (0.0, -half_sp, 0.0)
    loop_pos1 = (0.0, half_sp, 0.0)
    for r in range(rows - 1):
        for c in range(cols):
            if r % 2 == 1 and c == cols - 1:
                continue  # already in tree
            if r % 2 == 0 and c == 0:
                continue  # already in tree
            a = node_prims[r][c]
            b = node_prims[r + 1][c]
            _direct_joint(a, b, loop_pos0, loop_pos1, exclude=True)
            _vis_capsule(x_offset + c * spacing,
                         y_offset - r * spacing - half_sp, True,
                         f"edge_v_loop_{r}_{c}")

    # ---- 5.  Articulation properties ----
    kit_utils.set_articulation_properties(
        xform_path,
        enable_self_collisions=False,
        solver_position_iteration_count=12,
        solver_velocity_iteration_count=4,
    )

    return {"nodes": node_prims}



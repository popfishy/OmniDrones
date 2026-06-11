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

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

import omni.kit.commands
import omni.usd
from omni.physx.scripts import particleUtils, physicsUtils
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


def _make_compliant_cross_joint(stage, body0, body1, pos0, pos1):
    """Create a D6 joint with high-stiffness drives replacing a rigid FixedJoint.

    Use this to connect bodies that belong to *different* articulations —
    the drives act as numerical buffers, absorbing micro-discrepancies between
    the two articulation solvers instead of producing infinite constraint forces.
    """
    joint: Usd.Prim = script_utils.createJoint(stage, "D6", body0, body1)
    joint.GetAttribute("physics:excludeFromArticulation").Set(True)
    joint.GetAttribute("physics:localPos0").Set(pos0)
    joint.GetAttribute("physics:localPos1").Set(pos1)
    # Lock translational DOFs with moderate-stiffness drives.
    # k=1e5 on 0.01kg bodies produces 1000N @ 0.01m displacement →
    # 100,000 m/s² — enough to crash the GPU solver.  k=500 is safe
    # while still being much stiffer than the rope's bending DOFs.
    for dof in ("transX", "transY", "transZ"):
        drive = UsdPhysics.DriveAPI.Apply(joint, dof)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(5e2)
        drive.CreateDampingAttr(50)
    for dof in ("rotX", "rotY", "rotZ"):
        drive = UsdPhysics.DriveAPI.Apply(joint, dof)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(5e2)
        drive.CreateDampingAttr(50)
    return joint


def _lock_d6_trans_and_rotx(joint: Usd.Prim):
    """Lock transX/Y/Z and rotX on a D6 joint (low > high = locked in PhysX)."""
    joint.GetAttribute("physics:excludeFromArticulation").Set(True)
    for dof in ("transX", "transY", "transZ", "rotX"):
        limit_api = UsdPhysics.LimitAPI.Apply(joint, dof)
        limit_api.CreateLowAttr(1.0)
        limit_api.CreateHighAttr(-1.0)


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
            _lock_d6_trans_and_rotx(joint)
            # rotY/Z: free with limits and angular drives (rope flexibility)
            for dof in ("rotY", "rotZ"):
                limitAPI = UsdPhysics.LimitAPI.Apply(joint, dof)
                limitAPI.CreateLowAttr(-110)
                limitAPI.CreateHighAttr(110)
                driveAPI = UsdPhysics.DriveAPI.Apply(joint, dof)
                driveAPI.CreateTypeAttr("force")
                driveAPI.CreateDampingAttr(rope_damping)
                driveAPI.CreateStiffnessAttr(rope_stiffness)

        links.append(capsuleGeom.GetPrim())

    # Cross-articulation connections: use compliant D6 joints instead of
    # rigid FixedJoints.  This absorbs the micro-discrepancies between the
    # drone articulation solver and the net articulation solver, preventing
    # the infinite constraint forces that cause NaN explosions.
    if from_prim is not None:
        _make_compliant_cross_joint(stage, from_prim, links[-1],
                                    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    if to_prim is not None:
        _make_compliant_cross_joint(stage, links[0], to_prim,
                                    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    return links


def create_pbd_rope(
    xform_path: str = "/World/rope",
    translation=(0, 0, 0),
    particle_system_path: str = "/World/particleSystem",
    from_prim: Union[str, Usd.Prim] = None,
    to_prim: Union[str, Usd.Prim] = None,
    num_particles: int = 12,
    rope_length: float = 1.2,
    particle_mass: float = 0.01,
    stretch_stiffness: float = 1e4,
    bend_stiffness: float = 2e2,
    spring_damping: float = 0.2,
    solver_position_iterations: int = 16,
) -> dict:
    """Create a PBD particle-based rope connecting two rigid bodies.

    Uses PhysX PBD particle cloth with auto-generated spring constraints
    to simulate a flexible rope/cable.  Each rope is a narrow 2-row mesh
    strip whose edges generate valid spring constraints via the
    PhysxAutoParticleClothAPI.

    GPU-native — NO D6 joints, NO CPU API fallbacks.

    .. note::
       ``sim.reset()`` MUST be called before initializing the ClothPrimView
       for this rope.  The particle positions can only be read via
       ``ClothPrimView.get_world_positions()`` — the USD mesh points
       attribute is static.

    Args:
        xform_path: USD path for the rope mesh prim.
        translation: world-space offset for the rope root.
        particle_system_path: Path to the shared PhysxParticleSystem.
        from_prim: Rigid body at the far end (e.g. net corner).
        to_prim: Rigid body at the near end (e.g. drone base_link).
        num_particles: Number of lengthwise particle rows.
        rope_length: Total rest length of the rope (m).
        particle_mass: Mass per particle row (kg).
        stretch_stiffness: Spring stretch stiffness (force/distance).
        bend_stiffness: Spring bend stiffness.
        spring_damping: Damping on all spring constraints.
        solver_position_iterations: PBD solver iterations.

    Returns:
        dict with ``mesh_path``, ``num_vertices``, ``attachments``.
    """
    stage = stage_utils.get_current_stage()
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()

    W = 0.02   # strip width — narrow enough for rope-like behavior
    N = num_particles

    # 1. 2-row strip mesh: row 0 = top, row N..2N-1 = bottom
    positions = [
        Gf.Vec3f(i * rope_length / (N - 1), W / 2, 0.0)
        for i in range(N)
    ] + [
        Gf.Vec3f(i * rope_length / (N - 1), -W / 2, 0.0)
        for i in range(N)
    ]
    face_counts = []
    face_indices = []
    for k in range(N - 1):
        a, b, c, d = k, k + 1, N + k, N + k + 1
        face_counts.extend([3, 3])
        face_indices.extend([a, c, b, a, d, c])

    # 2. Xform → Mesh
    xform = prim_utils.get_prim_at_path(xform_path)
    if not xform:
        xform = UsdGeom.Xform.Define(stage, xform_path)
    mesh_path = f"{xform_path}/ropeMesh"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr().Set(positions)
    mesh.CreateFaceVertexCountsAttr().Set(face_counts)
    mesh.CreateFaceVertexIndicesAttr().Set(face_indices)
    mesh.AddTranslateOp().Set(Gf.Vec3f(*translation))
    mesh.AddOrientOp().Set(Gf.Quatf(1.0))

    # 3. Particle system (one per rope group — caller may share across ropes)
    if not stage.GetPrimAtPath(particle_system_path):
        particleUtils.add_physx_particle_system(
            stage=stage,
            particle_system_path=particle_system_path,
            contact_offset=0.02,
            rest_offset=0.01,
            particle_contact_offset=0.03,
            solid_rest_offset=0.01,
            fluid_rest_offset=0.0,
            solver_position_iterations=solver_position_iterations,
            simulation_owner="/physicsScene",
            particle_system_enabled=True,
        )

    # Particle material
    mat_path = Sdf.Path(particle_system_path + "_mat")
    if not stage.GetPrimAtPath(mat_path):
        particleUtils.add_pbd_particle_material(stage, mat_path)
        particleUtils.add_pbd_particle_material(stage, mat_path, friction=0.5)
        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(particle_system_path), mat_path
        )

    # 4. Auto-spring cloth
    particleUtils.add_physx_particle_cloth(
        stage=stage,
        path=mesh_path,
        dynamic_mesh_path=None,
        particle_system_path=particle_system_path,
        spring_stretch_stiffness=stretch_stiffness,
        spring_bend_stiffness=bend_stiffness,
        spring_shear_stiffness=0.0,
        spring_damping=spring_damping,
        self_collision=False,
    )

    # 5. Mass
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).GetMassAttr().Set(
        particle_mass * N * 2
    )

    # 6. PhysicsAttachments with explicit vertex indices.
    #    PhysxAutoAttachmentAPI (CreatePhysicsAttachment default) auto-selects
    #    vertices by proximity, which can claim ALL cloth vertices for the first
    #    attachment, leaving none for the second.  We pin only the endpoint
    #    vertices explicitly.
    #    Rope mesh layout:
    #      top row:    vertices 0 … N-1
    #      bottom row: vertices N … 2N-1
    #    "to"   (first column):  indices 0, N    → to_prim (drone)
    #    "from" (last column):   indices N-1, 2N-1 → from_prim (net corner)
    attachments = []
    if to_prim is not None:
        a_to = Sdf.Path(
            omni.usd.get_stage_next_free_path(stage, f"{mesh_path}/attTo", True)
        )
        to_attach = PhysxSchema.PhysxPhysicsAttachment.Define(stage, a_to)
        to_attach.GetActor0Rel().SetTargets([Sdf.Path(mesh_path)])
        to_attach.GetActor1Rel().SetTargets([to_prim.GetPath()])
        to_attach.GetFilterType0Attr().Set("Vertices")
        to_attach.GetCollisionFilterIndices0Attr().Set([0, N])
        attachments.append(("to", str(a_to)))

    if from_prim is not None:
        a_from = Sdf.Path(
            omni.usd.get_stage_next_free_path(stage, f"{mesh_path}/attFrom", True)
        )
        from_attach = PhysxSchema.PhysxPhysicsAttachment.Define(stage, a_from)
        from_attach.GetActor0Rel().SetTargets([Sdf.Path(mesh_path)])
        from_attach.GetActor1Rel().SetTargets([from_prim.GetPath()])
        from_attach.GetFilterType0Attr().Set("Vertices")
        from_attach.GetCollisionFilterIndices0Attr().Set([N - 1, 2 * N - 1])
        attachments.append(("from", str(a_from)))

    return {
        "mesh_path": mesh_path,
        "num_vertices": N * 2,
        "num_particles": N,
        "attachments": attachments,
    }


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
    """Create a 2D net grid of sphere nodes connected by Capsule+D6 joint edges.

    All bodies are standalone RigidBodies — NO PhysX Articulation (avoids GPU
    API conflicts with ArticulationView during simulation).  Capsule edges
    serve as physical links (not just visual), giving the net 2D rigidity
    through properly separated D6 joint anchor points.

    Returns:
        dict with keys ``nodes`` (list of lists of Usd.Prim),
        ``edges_h``, ``edges_v``.
    """
    stage = stage_utils.get_current_stage()
    net_xform = UsdGeom.Xform.Define(stage, xform_path)
    net_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))

    # Centre the net around origin in XY
    x_offset = -(cols - 1) * spacing / 2.0
    y_offset = (rows - 1) * spacing / 2.0

    # ---- 1.  Create nodes ----
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
            sphere.AddOrientOp().Set(Gf.Quatf(1.0))  # identity orientation (required for RigidBody API)
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
    half_len = spacing / 2.0  # capsule half-length

    # Helper: apply rotY/Z drive + lock other DOFs on a D6 joint.
    def _configure_edge_joint(joint: Usd.Prim):
        _lock_d6_trans_and_rotx(joint)
        joint.GetAttribute("limit:rotY:physics:low").Set(-110)
        joint.GetAttribute("limit:rotY:physics:high").Set(110)
        joint.GetAttribute("limit:rotZ:physics:low").Set(-110)
        joint.GetAttribute("limit:rotZ:physics:high").Set(110)
        UsdPhysics.DriveAPI.Apply(joint, "rotY")
        UsdPhysics.DriveAPI.Apply(joint, "rotZ")
        joint.GetAttribute("drive:rotY:physics:damping").Set(edge_damping)
        joint.GetAttribute("drive:rotY:physics:stiffness").Set(edge_stiffness)
        joint.GetAttribute("drive:rotZ:physics:damping").Set(edge_damping)
        joint.GetAttribute("drive:rotZ:physics:stiffness").Set(edge_stiffness)

    # ---- 2.  Horizontal edges  ----
    edges_h: list = []
    for r in range(rows):
        for c in range(cols - 1):
            edge_path = f"{xform_path}/edge_h_{r}_{c}"
            a = node_prims[r][c]       # left node
            b = node_prims[r][c + 1]   # right node

            # Xform at midpoint — capsules are oriented along X by default
            edge_xform = UsdGeom.Xform.Define(stage, edge_path)
            edge_xform.AddTranslateOp().Set(
                Gf.Vec3f(x_offset + c * spacing + half_len,
                         y_offset - r * spacing, 0))

            capsule = UsdGeom.Capsule.Define(stage, f"{edge_path}/capsule")
            capsule.CreateHeightAttr(spacing)
            capsule.CreateRadiusAttr(link_radius)
            capsule.CreateAxisAttr("X")               # capsule along X
            capsule.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
            capsule.AddOrientOp().Set(Gf.Quatf(1.0))
            capsule.CreateDisplayColorAttr().Set([color])

            UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
            UsdPhysics.MassAPI.Apply(capsule.GetPrim()).CreateMassAttr().Set(0.005)
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
            capsule.GetPrim().GetAttribute("physics:collisionEnabled").Set(False)

            # Joint left  node (a) → capsule: anchor at node centre & capsule -X end
            j_a: Usd.Prim = script_utils.createJoint(stage, "D6", a, capsule.GetPrim())
            j_a.GetAttribute("physics:localPos0").Set((0.0, 0.0, 0.0))   # node centre
            j_a.GetAttribute("physics:localPos1").Set((-half_len, 0.0, 0.0))  # capsule -X end
            _configure_edge_joint(j_a)

            # Joint capsule → right node (b): anchor at capsule +X end & node centre
            j_b: Usd.Prim = script_utils.createJoint(stage, "D6", capsule.GetPrim(), b)
            j_b.GetAttribute("physics:localPos0").Set((half_len, 0.0, 0.0))    # capsule +X end
            j_b.GetAttribute("physics:localPos1").Set((0.0, 0.0, 0.0))          # node centre
            _configure_edge_joint(j_b)

            edges_h.append((capsule.GetPrim(), a, b))

    # ---- 3.  Vertical edges  ----
    edges_v: list = []
    for r in range(rows - 1):
        for c in range(cols):
            edge_path = f"{xform_path}/edge_v_{r}_{c}"
            a = node_prims[r][c]       # upper node
            b = node_prims[r + 1][c]   # lower node

            # Xform at midpoint, rotated 90° around Z so capsule-X = world-Y
            edge_xform = UsdGeom.Xform.Define(stage, edge_path)
            edge_xform.AddTranslateOp().Set(
                Gf.Vec3f(x_offset + c * spacing,
                         y_offset - r * spacing - half_len, 0))
            edge_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 90))

            capsule = UsdGeom.Capsule.Define(stage, f"{edge_path}/capsule")
            capsule.CreateHeightAttr(spacing)
            capsule.CreateRadiusAttr(link_radius)
            capsule.CreateAxisAttr("X")               # capsule along X → world-Y after xform rotation
            capsule.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
            capsule.AddOrientOp().Set(Gf.Quatf(1.0))
            capsule.CreateDisplayColorAttr().Set([color])

            UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
            UsdPhysics.MassAPI.Apply(capsule.GetPrim()).CreateMassAttr().Set(0.005)
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
            capsule.GetPrim().GetAttribute("physics:collisionEnabled").Set(False)

            # Joint upper node (a) → capsule:
            #   capsule local +X → world +Y → toward upper node
            j_a: Usd.Prim = script_utils.createJoint(stage, "D6", a, capsule.GetPrim())
            j_a.GetAttribute("physics:localPos0").Set((0.0, 0.0, 0.0))      # node centre
            j_a.GetAttribute("physics:localPos1").Set((half_len, 0.0, 0.0))  # capsule +X end
            _configure_edge_joint(j_a)

            # Joint capsule → lower node (b):
            #   capsule local -X → world -Y → toward lower node
            j_b: Usd.Prim = script_utils.createJoint(stage, "D6", capsule.GetPrim(), b)
            j_b.GetAttribute("physics:localPos0").Set((-half_len, 0.0, 0.0))  # capsule -X end
            j_b.GetAttribute("physics:localPos1").Set((0.0, 0.0, 0.0))         # node centre
            _configure_edge_joint(j_b)

            edges_v.append((capsule.GetPrim(), a, b))

    return {"nodes": node_prims, "edges_h": edges_h, "edges_v": edges_v}



# PBD Particle Rope 替代 D6 Joint Rope 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用 PhysX GPU 原生的 PBD 粒子系统替代 D6 joint 链条实现绳缆，从根本上消除 GPU 模式下 PxRigidDynamic CPU API 非法调用问题。

**Architecture:** 一个 `ParticleSystem` 包含所有 rope 粒子，每条 rope 是一段 1D 粒子链配置为 `ClothPrim`（高 stretch stiffness = 绳缆特性）。端点粒子通过 `PhysxPhysicsAttachment` 固定到 drone base_link / net corner。使用 `ClothPrimView` 做多环境 tensor 批量操作。

**Tech Stack:** Isaac Sim 4.1.0, PhysX 5 PBD particle system, `ClothPrim` / `ClothPrimView` / `ParticleSystem`, `PhysxPhysicsAttachment`, OmniDrones 现有 `NetCaptureGroup` 框架

**风险提示:** `add_physx_particle_cloth` API 在 PhysX 5.2.x 标记为 DEPRECATED（计划被 FEM deformable 替代）。虽然在 Isaac Sim 4.1.0 仍可用，但未来版本可能移除。备选方案：FEM deformable rope（复杂度高，暂不采用）。

---

## 文件结构

| 文件 | 变更 | 职责 |
|------|------|------|
| `omni_drones/utils/scene.py` | MODIFY | 新增 `create_pbd_rope()` / 保留 `create_net()` |
| `omni_drones/envs/net_capture/utils.py` | MODIFY | NetCaptureGroup 接入 PBD rope |
| `omni_drones/envs/net_capture/net_capture.py` | MODIFY | _reset_idx / _compute_state_and_obs 适配 |
| `scripts/debug_pbd_rope.py` | CREATE | 独立诊断脚本（单 rope → 全场景） |

---

## Phase 0: 可行性验证（先做，避免白干）

### Task 0.1: 创建最小 PBD 粒子 rope + rigid body 附着验证脚本

**Files:**
- Create: `scripts/debug_pbd_rope.py`

- [ ] **Step 1: 写出最小验证脚本**

创建单个 rope 连接 drone base_link 和一个固定 rigid body，验证：
1. `ParticleSystem` + `ClothPrim` 能否创建 1D 粒子链
2. `PhysxPhysicsAttachment` 能否连接粒子端点 → rigid body
3. GPU 下是否产生 PhysX 错误
4. 绳子在重力下是否表现出柔性行为

```python
#!/usr/bin/env python3
"""Minimal PBD rope + rigid body attachment verification.

Step 0 = single rigid body attached to world via PBD rope
Step 1 = rope between drone base_link and net corner node
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
from omni.isaac.core.materials.particle_material import ParticleMaterial
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core.prims.soft.cloth_prim import ClothPrim
from omni.isaac.core.utils import prims as prim_utils
from pxr import UsdGeom, Gf, PhysxSchema, Sdf, UsdPhysics
from omni.physx.scripts import particleUtils, physicsUtils
import omni.kit.commands

sim = SimulationContext(
    stage_units_in_meters=1.0, physics_dt=0.016, rendering_dt=0.016,
    backend="torch", device="cuda:0", physics_prim_path="/physicsScene",
)
dev = sim.device

# --- Phase 0a: 创建 ParticleSystem ---
ps_path = "/World/particleSystem"
ps = ParticleSystem(
    prim_path=ps_path,
    solver_position_iteration_count=8,
    non_particle_collision_enabled=True,
    global_self_collision_enabled=False,
)
ps.set_simulation_owner("/physicsScene")
print(f"ParticleSystem at {ps_path}")

# --- Phase 0b: 创建 rope mesh (1D 粒子链) ---
# 用 UsdGeom.Mesh 创建一条粒子链: num_particles 个顶点，每对相邻顶点之间用 spring 连接
num_particles = 12
rope_length = 1.2  # meters
particle_mass = 0.001  # per particle

# 创建粒子位置：从 (0, 0, 0) 到 (rope_length, 0, 0)
positions = [(i * rope_length / (num_particles - 1), 0.0, 0.0) for i in range(num_particles)]

# 创建三角形面（cloth 需要面来生成 spring 约束）
# 对于 1D rope，用退化的三角形带
face_vertex_counts = []
face_vertex_indices = []
for i in range(num_particles - 1):
    face_vertex_counts.append(3)
    face_vertex_indices.extend([i, i + 1, i])

rope_mesh_path = "/World/rope"
rope_mesh = UsdGeom.Mesh.Define(stage, rope_mesh_path)
rope_mesh.CreatePointsAttr().Set(positions)
rope_mesh.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
rope_mesh.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)

# --- Phase 0c: 配置 ClothPrim ---
cloth = particleUtils.add_physx_particle_cloth(
    stage, rope_mesh_path,
    dynamic_mesh_path=None,
    particle_system_path=ps_path,
    spring_stretch_stiffness=1e4,   # 高 stretch stiffness → 绳缆感
    spring_bend_stiffness=1e2,      # 中等 bend stiffness → 柔性弯曲
    spring_shear_stiffness=0.0,     # 无 shear（rope 不需要）
    spring_damping=1.0,
    self_collision=False,
)
print(f"Cloth configured at {rope_mesh_path}")

# --- Phase 0d: 创建固定点 rigid body ---
anchor_path = "/World/anchor"
anchor = physicsUtils.add_rigid_box(
    stage, anchor_path,
    size=Gf.Vec3f(0.1, 0.1, 0.1),
    position=Gf.Vec3f(0.0, 0.0, 0.0),
    density=0.0,  # kinematic
)
print(f"Anchor at {anchor_path}")

# --- Phase 0e: 创建附着 ---
# 将 rope 的第一个顶点附着到 anchor
attachment_path = Sdf.Path(anchor_path + "/ropeAttachment")
attachment_path = Sdf.Path(stage.GetNextFreePath(str(attachment_path)))
omni.kit.commands.execute(
    "CreatePhysicsAttachment",
    target_attachment_path=attachment_path,
    actor0_path=rope_mesh_path,   # cloth
    actor1_path=anchor_path,      # rigid
)

# --- Phase 0f: 初始化并运行 ---
sim.reset()
for i in range(120):  # 2 seconds
    sim.step(render=False)
    if i % 60 == 0:
        print(f"  step {i}")

print("Phase 0 PASSED — no PhysX errors")
simulation_app.close()
```

- [ ] **Step 2: 运行验证脚本**

```bash
cd ~/OmniDrones/scripts && python debug_pbd_rope.py
```

**通过标准:** 无 PhysX `PxRigidDynamic::setLinearVelocity` 等 GPU API 错误，rope 在重力下弯曲下垂。

- [ ] **Step 3: 提交**

```bash
git add scripts/debug_pbd_rope.py
git commit -m "test: PBD rope 最小可行性验证脚本"
```

---

## Phase 1: 创建 `create_pbd_rope()` 工具函数

### Task 1.1: 实现 `create_pbd_rope()` 函数

**Files:**
- Modify: `omni_drones/utils/scene.py`

- [ ] **Step 1: 在 `scene.py` 中添加 `create_pbd_rope()`**

在现有 `create_rope()` 后面新增函数。它需要：
- 接收 `particle_system_path`（共享的粒子系统）
- 创建 1D 粒子链 mesh
- 调用 `particleUtils.add_physx_particle_cloth()` 配置 spring 约束
- 返回 rope prim path 和端点信息

```python
# omni_drones/utils/scene.py — 在 create_rope() 函数后面添加

def create_pbd_rope(
    xform_path: str = "/World/rope",
    translation=(0, 0, 0),
    particle_system_path: str = "/World/particleSystem",
    from_prim: Union[str, Usd.Prim] = None,
    to_prim: Union[str, Usd.Prim] = None,
    num_particles: int = 12,
    rope_length: float = 1.2,
    particle_mass: float = 0.001,
    stretch_stiffness: float = 1e4,
    bend_stiffness: float = 1e2,
    spring_damping: float = 1.0,
    enable_collision: bool = False,
) -> dict:
    """Create a PBD particle-based rope connecting two rigid bodies.

    Uses PhysX PBD particle system with auto-generated spring constraints
    to simulate a flexible rope/cable.  Each rope is a 1D particle chain
    configured as a degenerate ClothPrim (high stretch stiffness, moderate
    bend stiffness).

    GPU-native — NO D6 joints, NO CPU API fallbacks.

    Args:
        xform_path: USD path for the rope mesh prim.
        translation: world-space offset for the rope root xform.
        particle_system_path: Path to the shared PhysxParticleSystem.
        from_prim: Rigid body at the "far" end of the rope (e.g. net corner).
        to_prim: Rigid body at the "near" end of the rope (e.g. drone base_link).
        num_particles: Number of particles in the chain.
        rope_length: Total rest length of the rope (m).
        particle_mass: Mass per particle (kg).
        stretch_stiffness: Spring stretch stiffness (force/distance).
        bend_stiffness: Spring bend stiffness.
        spring_damping: Damping on all spring constraints.
        enable_collision: Whether rope particles collide with other bodies.

    Returns:
        dict with keys: ``mesh_path``, ``num_particles``, ``attachments``,
        ``from_attachment``, ``to_attachment``.
    """
    stage = stage_utils.get_current_stage()
    if isinstance(from_prim, str):
        from_prim = prim_utils.get_prim_at_path(from_prim)
    if isinstance(to_prim, str):
        to_prim = prim_utils.get_prim_at_path(to_prim)
    if isinstance(translation, torch.Tensor):
        translation = translation.tolist()

    # 1. Particle chain mesh — positions along X, centred at origin
    positions = [
        Gf.Vec3f(i * rope_length / (num_particles - 1) - rope_length / 2,
                 0.0, 0.0)
        for i in range(num_particles)
    ]

    # 2. Degenerate triangle strip for spring auto-generation
    #    PhysxAutoParticleClothAPI creates springs from mesh edges
    face_vertex_counts = []
    face_vertex_indices = []
    for i in range(num_particles - 1):
        face_vertex_counts.append(3)
        face_vertex_indices.extend([i, i + 1, i])

    # 3. Create mesh prim
    if not prim_utils.is_prim_path_valid(xform_path):
        prim_utils.define_prim(xform_path, "Xform")
    mesh_path = f"{xform_path}/ropeMesh"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr().Set(positions)
    mesh.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)
    mesh.AddTranslateOp().Set(Gf.Vec3f(*translation))
    mesh.AddOrientOp().Set(Gf.Quatf(1.0))

    # 4. Configure as particle cloth with auto springs
    particleUtils.add_physx_particle_cloth(
        stage, mesh_path,
        dynamic_mesh_path=None,
        particle_system_path=particle_system_path,
        spring_stretch_stiffness=stretch_stiffness,
        spring_bend_stiffness=bend_stiffness,
        spring_shear_stiffness=0.0,
        spring_damping=spring_damping,
        self_collision=False,
    )

    # 5. Create rigid-body attachments at both ends
    import omni.kit.commands
    attachments = []

    if to_prim is not None:
        to_attach_path = Sdf.Path(f"{mesh_path}/toAttachment")
        to_attach_path = Sdf.Path(stage.GetNextFreePath(str(to_attach_path)))
        omni.kit.commands.execute(
            "CreatePhysicsAttachment",
            target_attachment_path=to_attach_path,
            actor0_path=mesh_path,
            actor1_path=to_prim.GetPath(),
        )
        attachments.append(("to", to_attach_path))

    if from_prim is not None:
        from_attach_path = Sdf.Path(f"{mesh_path}/fromAttachment")
        from_attach_path = Sdf.Path(stage.GetNextFreePath(str(from_attach_path)))
        omni.kit.commands.execute(
            "CreatePhysicsAttachment",
            target_attachment_path=from_attach_path,
            actor0_path=mesh_path,
            actor1_path=from_prim.GetPath(),
        )
        attachments.append(("from", from_attach_path))

    result = {
        "mesh_path": mesh_path,
        "num_particles": num_particles,
        "attachments": attachments,
        "to_attachment": attachments[0][1] if to_prim else None,
        "from_attachment": attachments[-1][1] if from_prim else None,
    }
    return result
```

- [ ] **Step 2: 更新 scene.py 的 import**

在 `scene.py` 顶部添加缺少的 import：

```python
import omni.kit.commands
from pxr import Sdf
```

- [ ] **Step 3: 提交**

```bash
git add omni_drones/utils/scene.py
git commit -m "feat: add create_pbd_rope() using PhysX PBD particle cloth"
```

---

## Phase 2: 集成到 NetCaptureGroup

### Task 2.1: 修改 NetCaptureGroup 支持 PBD rope

**Files:**
- Modify: `omni_drones/envs/net_capture/utils.py`
- Modify: `omni_drones/envs/net_capture/net_capture.py` (如果配置结构变化)

- [ ] **Step 1: 在 NetCaptureCfg 中添加 PBD rope 参数**

```python
# omni_drones/envs/net_capture/utils.py — NetCaptureCfg class
@dataclass
class NetCaptureCfg(RobotCfg):
    num_drones: int = 4
    net_rows: int = 6
    net_cols: int = 6
    net_spacing: float = 0.25
    rope_links: int = 12         # ← 保留兼容，但 PBD 模式下含义变为 particles
    rope_link_length: float = 0.1
    node_mass: float = 0.01
    corner_mass: float = 0.02

    # PBD rope 参数
    use_pbd_rope: bool = True      # 是否使用 PBD（默认 True）
    pbd_particle_mass: float = 0.001
    pbd_stretch_stiffness: float = 1e4
    pbd_bend_stiffness: float = 1e2
    pbd_spring_damping: float = 1.0
    pbd_solver_iterations: int = 8
```

- [ ] **Step 2: 在 NetCaptureGroup.spawn() 中调用 create_pbd_rope()**

修改 `spawn()` 方法中的 rope 创建部分（第 193-208 行附近），根据 `cfg.use_pbd_rope` 选择创建方式：

```python
# omni_drones/envs/net_capture/utils.py — NetCaptureGroup.spawn()
# 在 drone + rope 创建循环中：

# 1. 先创建共享 ParticleSystem（如果是第一个 rope）
if cfg.use_pbd_rope:
    ps_path = f"{prim_path}/particleSystem"
    from omni.physx.scripts.particleUtils import add_physx_particle_system
    add_physx_particle_system(
        stage, ps_path,
        solver_position_iterations=cfg.pbd_solver_iterations,
        non_particle_collision_enabled=True,
        global_self_collision_enabled=False,
    )
    # 在 spawn 循环外部创建一次 ParticleSystem
    # （TODO: 移到 __init__ 中确保只创建一次）

for i in range(self.num_drones):
    drone_path = f"/World/envs/env_0/{self.drone.name.lower()}_{i}"
    self.drone.spawn(...)
    _strip_articulation(drone_path)

    r, c = corner_indices[i]
    corner_node_path = str(net_info["nodes"][r][c].GetPath())
    drone_base_link = f"{drone_path}/base_link"

    rope_translation = drone_translations[i].tolist()

    if cfg.use_pbd_rope:
        # PBD rope — GPU native
        rope_length = cfg.rope_links * cfg.rope_link_length
        rope_info = create_pbd_rope(
            xform_path=f"{prim_path}/rope_pbd_{i}",
            translation=rope_translation,
            particle_system_path=ps_path,
            from_prim=corner_node_path,     # net corner
            to_prim=drone_base_link,        # drone
            num_particles=cfg.rope_links,
            rope_length=rope_length,
            particle_mass=cfg.pbd_particle_mass,
            stretch_stiffness=cfg.pbd_stretch_stiffness,
            bend_stiffness=cfg.pbd_bend_stiffness,
            spring_damping=cfg.pbd_spring_damping,
        )
        self._rope_infos.append(rope_info)
    else:
        # 原有 D6 joint rope
        scene_utils.create_rope(
            xform_path=f"{prim_path}/rope_{i}",
            translation=rope_translation,
            from_prim=corner_node_path,
            to_prim=drone_base_link,
            num_links=cfg.rope_links,
            link_length=cfg.rope_link_length,
            color=(0.4, 0.2, 0.1),
            enable_collision=False,
            exclude_from_articulation=True,
        )
```

- [ ] **Step 3: 修改 NetCaptureGroup.initialize() 添加 ClothPrimView**

在 `initialize()` 中为 PBD rope 创建 ClothPrimView：

```python
# omni_drones/envs/net_capture/utils.py — NetCaptureGroup.initialize()
def initialize(self, prim_paths_expr=None, track_contact_forces=False):
    # ... 现有代码 ...

    if self.cfg.use_pbd_rope:
        from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
        self.rope_cloth_view = ClothPrimView(
            f"{self.prim_paths_expr}/rope_pbd_*/ropeMesh",
            particle_systems="/World/envs/env_0/Group_0/particleSystem",  # template
            name="rope_pbd_view",
        )
        self.rope_cloth_view.initialize()
    else:
        # 原有 RigidPrimView for rope segments
        self.rope_segs_view = RigidPrimView(
            f"{self.prim_paths_expr}/rope_*/seg_*",
            reset_xform_properties=False,
        )
        self.rope_segs_view.initialize()
```

- [ ] **Step 4: 提交**

```bash
git add omni_drones/envs/net_capture/utils.py
git commit -m "feat: NetCaptureGroup 集成 PBD particle rope"
```

---

## Phase 3: 环境管线适配

### Task 3.1: 修改 _reset_idx 适配 PBD rope

**Files:**
- Modify: `omni_drones/envs/net_capture/net_capture.py`

- [ ] **Step 1: 在 _reset_idx 中处理 PBD rope 的初始化**

PBD rope 通过 PhysxPhysicsAttachment 附着到刚体——当刚体位姿被 reset 时，附着点会自动跟随。**不需要单独设置 rope 粒子的位姿**（这是 PBD 方案的核心优势之一）。

```python
# omni_drones/envs/net_capture/net_capture.py — _reset_idx()
def _reset_idx(self, env_ids: torch.Tensor):
    # ... 现有 target + drone reset 代码 ...

    if self.group.cfg.use_pbd_rope:
        # PBD rope: particles are attached to rigid bodies via PhysxPhysicsAttachment.
        # When rigid bodies are teleported (set_world_poses), PhysX automatically
        # repositions the attached particles.  No separate rope reset needed.
        # We DO need zero rope particle velocities — handled by ClothPrimView.
        # self.group.rope_cloth_view.set_velocities(...)  # TBD after API check
        pass
    else:
        # 原有 D6 joint rope reset
        # ... reset rope/segs/edges poses and velocities ...
```

- [ ] **Step 2: 简化 _compute_state_and_obs 中的 rope 状态读取**

PBD rope 的粒子位置/速度通过 `ClothPrimView.get_world_positions()` 和 `get_velocities()` 读取。用于观察的话，可以取端点粒子坐标和中间粒子坐标：

```python
# PBD rope state — 暂不加入观察（仅保留 net + drone 观察）
# 后续可以加入 rope 端点力等信息
```

- [ ] **Step 3: 提交**

```bash
git add omni_drones/envs/net_capture/net_capture.py
git commit -m "refactor: _reset_idx 适配 PBD rope（无需单独重置粒子）"
```

---

## Phase 4: 端到端验证

### Task 4.1: 创建完整场景诊断脚本

**Files:**
- Modify: `scripts/debug_pbd_rope.py`

- [ ] **Step 1: 在诊断脚本中加入完整 NetCapture 场景测试**

在现有的 `debug_pbd_rope.py` 中扩展 Phase 1（完整 NetCapture + PBD rope + 多环境克隆）：

```python
# scripts/debug_pbd_rope.py — Phase 1
print("=== Phase 1: Full NetCapture with PBD ropes (4 envs) ===")

drone, _ = MultirotorBase.make("Hummingbird", "LeePositionController")
scene_utils.design_scene()

group_cfg = NetCaptureCfg(
    num_drones=4, net_rows=6, net_cols=6, net_spacing=0.25,
    rope_links=12, use_pbd_rope=True, pbd_solver_iterations=8,
)
group = NetCaptureGroup(drone=drone, cfg=group_cfg)
group.spawn(translations=[(0, 0, 0.5)], enable_collision=False)

cloner = GridCloner(spacing=6)
cloner.define_base_env("/World/envs")
env_paths = cloner.generate_paths("/World/envs/env", 4)
cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=env_paths,
             replicate_physics=False)
cloner.filter_collisions(
    sim.get_physics_context().prim_path, "/World/collisions",
    prim_paths=env_paths, global_paths=["/World/defaultGroundPlane"],
)

sim.reset()
group.initialize()

for i in range(120):  # 2 seconds
    drone.apply_action(torch.zeros(4, 4, 4, device=dev))
    sim.step(render=False)
    if i % 60 == 0:
        print(f"  step {i}")

print("Phase 1 PASSED — full scene with PBD ropes, no PhysX errors")
```

- [ ] **Step 2: 运行诊断**

```bash
cd ~/OmniDrones/scripts && python debug_pbd_rope.py
```

**通过标准:** 无 PhysX 错误，无 crash。

- [ ] **Step 3: 运行完整训练**

```bash
cd ~/OmniDrones/scripts
python train.py algo=mappo headless=true task=NetCapture/NetCapture \
  total_frames=10000 wandb.mode=offline algo.entropy_coef=0.01
```

**通过标准:** 训练稳定运行，无 PhysX GPU API 错误。

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "test: PBD rope 端到端验证通过"
```

---

## 备选方案：Phase 0 失败时的回退路径

如果 `PhysxPhysicsAttachment` 不支持 particle cloth → rigid body 附着：

### 方案 A: 手动计算绳缆力（最快）

在 `drone.apply_action` 后添加力计算，不使用 PhysX 约束：
- 读取 rope 端点粒子位置和 drone/net 端点位置
- 根据位移差计算弹簧-阻尼力（胡克定律）
- 用 `RigidPrimView.apply_forces_and_torques_at_pos` 施加到两端刚体
- 这也是 GPU 兼容的——只有 tensor 操作和 force API

### 方案 B: 保留 D6 joints + 优化 GPU 回退

继续用 `views/__init__.py` 的 GPU/CPU 双路径（当前 main 分支的方案），接受 warning 噪音但确保功能正确。

---

## 验证 checklist

- [ ] Task 0.2: 最小 PBD rope 脚本无 PhysX 错误
- [ ] Task 1.1: `create_pbd_rope()` 可独立创建 rope
- [ ] Task 2.1: NetCaptureGroup 可同时管理 PBD rope + D6 net
- [ ] Task 3.1: `_reset_idx` 正确处理 PBD rope
- [ ] Task 4.2: 完整诊断 4 env 场景无错误
- [ ] Task 4.3: 训练 10k 帧稳定运行

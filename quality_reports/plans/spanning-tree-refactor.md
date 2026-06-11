# NetCapture 生成树 + 闭环关节 重构方案

> 2026-06-05 | Status: DRAFT

## 1. 目标

将所有需要复位的物理体（无人机、绳索段、网节点、网边）纳入 PhysX Articulation 树，使其可通过 `ArticulationView.set_world_poses()` 完成 GPU 兼容的批量复位。2D 网格的闭合环路用 `excludeFromArticulation=True` 的独立 D6 关节处理。

## 2. 新物理架构

### 2.1 三大 Articulation

```
1. drone_0 (ArticulationRoot)
   ├── base_link
   ├── rotor_0..3
   └── rope_0/
       ├── seg_0 ──D6── seg_1 ──...── seg_11
       └── seg_11 ──Fixed(exclude=True)── net/node_0_0

2. drone_1 (ArticulationRoot)  ... same pattern → net/node_0_C-1
3. drone_2 (ArticulationRoot)  ... same pattern → net/node_R-1_0
4. drone_3 (ArticulationRoot)  ... same pattern → net/node_R-1_C-1

5. net (ArticulationRoot) — 唯一包含网格的 Articulation
   ├── spanning tree: 35 条 articulation 内 D6 边连接全部 36 节点
   │   (根: node_0_0, 水平边全保留, 仅首行保留垂直边)
   │
   └── 25 条 loop-closing D6 关节 (excludeFromArticulation=True)
       连接树上已有 link 对, 闭合其余四边形
```

### 2.2 生成树构造

```
6×6 网格 (36 节点, row-major: node_{r}_{c})

Articulation 树边 (35 条, 在 articulation 内):
  - 所有水平边:     (r, c) ──D6── (r, c+1)   for r=0..5, c=0..4   → 30 条
  - 第 0 列垂直边:   (r, 0) ──D6── (r+1, 0)  for r=0..4          →  5 条
  Total: 35 条 ← 连接全部 36 节点, 无环路

闭环关节 (25 条, excludeFromArticulation=True):
  - 其余垂直边:     (r, c) ──D6── (r+1, c)   for r=0..4, c=1..5  → 25 条
```

**Articulation 树根节点**：`node_0_0`

### 2.3 无人机 + 绳索

每架无人机是独立 Articulation，绳索段（seg_0..seg_N）作为 articulation 内的 link：

```
hummingbird_i (ArticulationRoot)
├── base_link
├── rotor_0..3 (revolute joints)
└── rope_i/
    ├── seg_0  ← D6 →  seg_1  ← D6 →  ...  ← D6 →  seg_11
    └── seg_11 ← Fixed(excludeFromArticulation=True) → net/node_corner
```

`seg_0` 通过 `Fixed(excludeFromArticulation=False)` 连接 `base_link`，属于同一 articulation 树内。

`seg_11` 通过 `Fixed(excludeFromArticulation=True)` 连接 `net/node_corner`——跨 articulation 边界。

### 2.4 复位机制

```
_reset_idx(env_ids):
  1. drone._reset_idx(env_ids)                          # 油门 → 悬停
  2. for i in 0..3:
       drone_i.set_world_poses(init[i], env_ids)        # GPU 兼容 ✅
  3. net_articulation.set_world_poses(init_net, env_ids) # GPU 兼容 ✅
  4. target 重新采样
```

**不再需要任何 RigidPrimView.set_world_poses！**

## 3. 代码修改清单

### 3.1 `omni_drones/utils/scene.py` — 重写 `create_net()`

**输入不变，输出改为返回 `(net_prim, corner_nodes)`。**

核心逻辑：

```python
def create_net(xform_path, rows, cols, spacing, ...):
    stage = stage_utils.get_current_stage()

    # ---- Step 1: 创建 ArticulationRoot Xform ----
    net_root = UsdGeom.Xform.Define(stage, xform_path)
    UsdPhysics.ArticulationRootAPI.Apply(net_root.GetPrim())
    PhysxSchema.PhysxArticulationAPI.Apply(net_root.GetPrim())

    # ---- Step 2: 创建节点 (Sphere + RigidBody, 同之前) ----
    nodes = [[None]*cols for _ in range(rows)]
    for r, c in product(range(rows), range(cols)):
        sphere = UsdGeom.Sphere.Define(stage, f"{xform_path}/node_{r}_{c}")
        # ... set radius, position, mass, rigid body (同之前) ...
        nodes[r][c] = sphere.GetPrim()

    # ---- Step 3: Articulation 内边 (生成树) ----
    # (a) 所有水平边
    for r in range(rows):
        for c in range(cols - 1):
            _make_articulation_edge(xform_path, nodes[r][c], nodes[r][c+1],
                                    "h", spacing, horizontal=True)
    # (b) 第 0 列垂直边 (连接行)
    for r in range(rows - 1):
        _make_articulation_edge(xform_path, nodes[r][0], nodes[r+1][0],
                                "v_tree", spacing, horizontal=False)

    # ---- Step 4: 闭环 D6 关节 (excludeFromArticulation) ----
    for r in range(rows - 1):
        for c in range(1, cols):
            _make_loop_closing_joint(xform_path, nodes[r][c], nodes[r+1][c],
                                     "v_loop", spacing)

    # ---- Step 5: 设置 articulation 属性 ----
    kit_utils.set_articulation_properties(
        xform_path,
        enable_self_collisions=False,
        solver_position_iteration_count=12,  # D6-heavy, use more iterations
        solver_velocity_iteration_count=4,
    )

    return {"net_prim": net_root.GetPrim(), "nodes": nodes}
```

**辅助函数 `_make_articulation_edge`**：

```python
def _make_articulation_edge(parent_path, node_a, node_b, tag, spacing, horizontal):
    """Create a Capsule + two D6 joints (IN articulation tree)."""
    edge_xform = UsdGeom.Xform.Define(stage, f"{parent_path}/edge_{tag}_{r}_{c}")
    # position at midpoint
    capsule = UsdGeom.Capsule.Define(stage, f"{edge_path}/capsule")
    # ... set height=spacing, radius, RigidBody, MassAPI ...
    # D6 joint a → capsule (articulation internal)
    j_a = script_utils.createJoint(stage, "D6", node_a, capsule.GetPrim())
    _configure_edge_joint(j_a, half_len)
    # D6 joint capsule → b (articulation internal)
    j_b = script_utils.createJoint(stage, "D6", capsule.GetPrim(), node_b)
    _configure_edge_joint(j_b, half_len)
```

**辅助函数 `_make_loop_closing_joint`**：

```python
def _make_loop_closing_joint(parent_path, node_a, node_b, tag, spacing):
    """Create a D6 joint OUTSIDE the articulation (excludeFromArticulation=True)."""
    j = script_utils.createJoint(stage, "D6", node_a, node_b)
    j.GetAttribute("physics:excludeFromArticulation").Set(True)
    _configure_loop_joint(j)
```

**关键差异**：
- Articulation 内边：Capsule 作为 RigidBody link 参与 articulation tree，两端 D6 关节不 exclude
- 闭环边：直接 D6 关节连接两个已存在的 articulation link，`excludeFromArticulation=True`

### 3.2 `omni_drones/utils/scene.py` — 修改 `create_rope()`

绳索段需要成为**无人机 Articulation 的一部分**：

```python
def create_rope(xform_path, from_prim, to_prim, num_links, link_length,
                exclude_from_articulation=True,  # 现有参数
                rope_in_articulation=False,       # NEW: 绳索段本身是否在 articulation 内
                articulation_root_path=None,       # NEW: 如果 rope_in_articulation, 指定 root
                ):
    # ... 同之前创建 seg 段 ...

    # 端连接
    if from_prim:
        joint = script_utils.createJoint(stage, "Fixed", from_prim, links[-1])
        joint.GetAttribute("physics:excludeFromArticulation").Set(exclude_from_articulation)
    if to_prim:
        joint = script_utils.createJoint(stage, "Fixed", links[0], to_prim)
        joint.GetAttribute("physics:excludeFromArticulation").Set(exclude_from_articulation)

    return links
```

对于无人机内的绳索：`exclude_from_articulation=False`（无人机-绳索同树），`exclude=True`（绳索-网跨树）。

### 3.3 `omni_drones/envs/net_capture/utils.py` — 重写 `NetCaptureGroup`

#### 配置

```python
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
```

#### `spawn()`

```python
def spawn(self, translations, prim_paths, enable_collision=False):
    for prim_path, translation in zip(prim_paths, translations):
        xform = prim_utils.create_prim(prim_path, translation=translation)

        # 1. 创建网 Articulation
        net_info = scene_utils.create_net(f"{prim_path}/net", ...)
        self.net_root = net_info["net_prim"]
        self.corner_nodes = [
            net_info["nodes"][0][0],
            net_info["nodes"][0][-1],
            net_info["nodes"][-1][0],
            net_info["nodes"][-1][-1],
        ]

        # 2. 创建无人机 Articulation (每架独立, drone.is_articulation=True)
        for i in range(self.num_drones):
            drone_path = f"/World/envs/env_0/{self.drone.name.lower()}_{i}"
            drone_prim = self.drone.spawn(
                translations=translation + drone_pos[i],
                prim_paths=[drone_path],
            )[0]
            # 不要 Unapply ArticulationRoot!

            # 3. 连接绳索 (在 drone articulation 内)
            scene_utils.create_rope(
                xform_path=f"{drone_path}/rope",
                from_prim=self.corner_nodes[i],   # net corner
                to_prim=f"{drone_path}/base_link", # drone
                exclude_from_articulation=True,     # 跨 articulation
                ...
            )
```

#### `initialize()`

```python
def initialize(self, prim_paths_expr=None, ...):
    # 无人机作为 articulation
    self.drone.initialize(f"/World/envs/.*/{self.drone.name.lower()}_*")

    # 网作为 articulation
    self.net_articulation = ArticulationView(
        f"{prim_paths_expr}/net",
        reset_xform_properties=False,
        shape=(-1, 1),  # 1 articulation per env
    )
    self.net_articulation.initialize()

    # RigidPrimView 仅用于读取网节点状态 (不用于 set)
    self.net_nodes_view = RigidPrimView(
        f"{prim_paths_expr}/net/node_*", ...
    )
    self.net_nodes_view.initialize()

    # 绳索段也在 drone articulation 内, 不需要独立 view
```

### 3.4 `omni_drones/envs/net_capture/net_capture.py` — 简化 `_reset_idx`

```python
def __init__(self, cfg, headless):
    # ... 同之前 ...
    super().__init__(cfg, headless)
    self.group.initialize()

    # Cache drone + net initial poses
    self.init_drone_pos, self.init_drone_rot = self.drone.get_world_poses(clone=True)
    self.init_net_pos, self.init_net_rot = self.group.net_articulation.get_world_poses(clone=True)
    # 不再需要 rope_view, init_rope_pos 等

def _reset_idx(self, env_ids):
    # Sample target
    target_pos = self.target_pos_dist.sample(env_ids.shape)
    self.target_pos[env_ids] = target_pos

    # Reset drones (ArticulationView — GPU 兼容)
    self.drone._reset_idx(env_ids)
    n = self.drone.n
    init_pos = self.init_drone_pos.reshape(self.num_envs, n, 3)[env_ids]
    init_rot = self.init_drone_rot.reshape(self.num_envs, n, 4)[env_ids]
    self.drone.set_world_poses(init_pos.reshape(-1, 3), init_rot.reshape(-1, 4), env_ids)
    self.drone.set_velocities(torch.zeros(len(env_ids)*n, 6, device=self.device), env_ids)

    # Reset net (ArticulationView — GPU 兼容) ← 新!
    net_pos = self.init_net_pos[env_ids].reshape(-1, 3)
    net_rot = self.init_net_rot[env_ids].reshape(-1, 4)
    self.group.net_articulation.set_world_poses(net_pos, net_rot, env_ids)
    # ArticulationView.set_velocities 和 set_joint_positions 也 GPU 兼容
    self.group.net_articulation.set_joint_positions(
        self.init_net_joint_pos[env_ids], env_ids)

    # Target heading + visual markers (同之前)
    ...
```

**关键变化**：reset 从 ~50 行删减到 ~20 行，全部走 ArticulationView GPU 路径。

### 3.5 需要新增的方法

| 文件 | 新增 | 说明 |
|------|------|------|
| `scene.py` | `_make_articulation_edge()` | 创建 articulation 内的 Capsule + D6 边 |
| `scene.py` | `_make_loop_closing_joint()` | 创建 cross-tree 闭环 D6 关节 |
| `utils.py` | `NetCaptureGroup._setup_drone_articulation()` | 无人机+绳索 articulation 组装 |

### 3.6 需要删除的代码

| 文件 | 删除 | 原因 |
|------|------|------|
| `utils.py` | `rope_view` + 初始化 | 绳索在 drone articulation 内，不需要独立 view |
| `net_capture.py` | `init_rope_pos/rot/vel` | 不再需要独立绳索状态缓存 |
| `net_capture.py` | 所有 `net_nodes_view.set_world_poses/set_velocities` | 替换为 `net_articulation` 操作 |
| `net_capture.py` | 所有 `rope_view.set_world_poses/set_velocities` | 绳索在 drone articulation 内自动复位 |
| `scene.py` | `_lock_d6_trans_and_rotx()` | 改为内联或保留用在新边创建中 |

### 3.7 不需要修改的代码

| 文件/方法 | 原因 |
|-----------|------|
| `_compute_state_and_obs` | 仍通过 RigidPrimView 读取网节点位置 (只读，不 set) |
| `_compute_reward_and_done` | 奖励计算不变 |
| `_pre_sim_step` | 动作应用逻辑不变 |
| `_set_specs` | 观测/动作空间定义不变 |
| `_design_scene` (target marker) | 视觉 marker 不变 |
| `NetCapture.yaml` | 配置参数不变 |

## 4. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 生成树选择影响网面对称性 | 中 | 水平边全覆盖 + 首列垂直边保证对称 |
| 闭环 D6 关节不参与 articulation 求解 → 刚度不一致 | 低 | 同一关节配置，阻尼为主；调大 solver iterations |
| 无人机 articulation 内塞 rope 段 → DOF 激增 | 低 | 12 segs × 2 DOF (rotY/Z) × 4 drones = 96 extra DOF, 可承受 |
| net articulation 跨 articulation FixedJoint 的 `exclude` 行为异常 | 低 | 当前已用 `exclude=True`，验证通过 |

## 5. 实现顺序

| # | 文件 | 内容 | 估计行数 |
|---|------|------|---------|
| 1 | `scene.py` | 添加 `_make_articulation_edge`, `_make_loop_closing_joint` | +60 |
| 2 | `scene.py` | 重写 `create_net()` | ~150 (替换) |
| 3 | `scene.py` | `create_rope()` 添加 `rope_in_articulation` 参数 | +10 |
| 4 | `utils.py` | 重写 `spawn()`, `initialize()` | ~200 (替换) |
| 5 | `net_capture.py` | 简化 `__init__` (删 rope cache), 简化 `_reset_idx` | ~50 (删多增少) |
| 6 | 验证 | 跑 test_net_capture_scene.py 确认场景加载 | - |
| 7 | 验证 | 跑 train.py total_frames=100 确认 reset 无 GPU 错误 | - |

## 6. 验证标准

- [ ] `test_net_capture_scene.py` 成功导出 USD — 网面平整，绳索自然下垂
- [ ] 训练启动无 `PxRigidDynamic::setGlobalPose illegal` 错误
- [ ] `_reset_idx` 后无人机+网回到初始位置，无满天乱飞
- [ ] Episode 长度从 ~0 增长到有意义的值

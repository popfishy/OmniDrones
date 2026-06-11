# 多无人机绳网协同捕捉场景 — 实现方案

## Context

用户已完成 TransportTrack 等多机任务训练，现需搭建新任务场景：4 架无人机通过绳索连接绳网，协同控制绳网对静态目标进行捕捉。需要从 USD 建模到 Python 环境代码的完整实现方案。

## 绳网建模方案

### 方案选择

| 方案 | 描述 | 优点 | 缺点 | 推荐度 |
|------|------|------|------|--------|
| **A. 离散节点+边网格** | M×N 个轻质小球节点 + 相邻节点间用 Capsule+D6关节连接 | 复用 `create_rope()` 模式，代码简洁；PhysX 原生兼容，RL 训练可行 | 刚体数量随网格增大（3×3=9节点+12边） | ★★★★★ |
| **B. 4条交叉绳索简化** | 4条绳索各自连到 4 个角点，角点间用 4 条边连成方形 | 刚体少，调试简单 | 网面形变不真实 | ★★★ |
| **C. PhysX Cloth** | PhysX deformable body API | 形变最真实 | Isaac Sim 4.1 支持有限，可能与 GPU 模式冲突 | ★★ |

**选方案 A**。现有 `create_rope()` 已用 Capsule + D6关节（rotY/rotZ 自由+damping/stiffness）实现绳索动力学，直接泛化到 2D 网格即可。

### 网结构设计

```
         drone_0                    drone_1
            \                          /
             \    rope_0      rope_1  /
              \                      /
         n00───e00───n01───e01───n02
          │          │          │
         e03        e04        e05
          │          │          │
         n10───e06───n11───e07───n12
          │          │          │
         e08        e09        e10
          │          │          │
         n20───e11───n21───e12───n22
              /                      \
             /    rope_2      rope_3  \
            /                          \
         drone_2                    drone_3
```

- **节点** (n00~n22)：小球体 (radius=0.02)，RigidBody，碰撞启用
- **角节点** (n00, n02, n20, n22)：质量稍高 (0.02kg)，承载绳索拉力
- **内部节点**：mass=0.01kg
- **边** (e00~e12)：短 Capsule 段，两端 D6 关节连到相邻节点（`rotY`/`rotZ` 自由，damping=10.0, stiffness=1.0）
- **绳索** (rope_0~3)：每条 ~12 个 Capsule 段（复用 `create_rope()`），一端 Fixed Joint 连无人机 `base_link`，另一端 Fixed Joint 连网角节点
- **网格参数**：默认 3×3（可配置 M×N）

## 系统连接设计

### Articulation 结构

参考 `TransportationGroup` 模式，整系统为**单个 Articulation**：

```
/World/envs/env_0/NetCaptureGroup_0  ← ArticulationRoot (在 spawn 末尾 Apply)
├── differential_uav_0/    ─┐
├── differential_uav_1/     ├─ 无人机 (ArticulationRootAPI 已 Unapply)
├── differential_uav_2/     │
├── differential_uav_3/    ─┘
├── rope_0/  ← Fixed Joint → 无人机0 base_link   ─┐
├── rope_1/  ← Fixed Joint → 无人机1 base_link    ├─ 绳索 (create_rope, 两端 FixedJoint 均在 articulation 内)
├── rope_2/  ← Fixed Joint → 无人机2 base_link    │   exclude_from_articulation=False
├── rope_3/  ← Fixed Joint → 无人机3 base_link   ─┘
├── net/
│   ├── node_0_0/  (Sphere + RigidBody, mass=0.02 — 角节点)
│   ├── node_0_1/  (Sphere + RigidBody, mass=0.01 — 边缘)
│   ├── node_0_2/  (Sphere + RigidBody, mass=0.02 — 角节点)
│   ├── ...        (M×N 个节点)
│   ├── edge_h_0_0/  (Capsule Xform: 水平边, 无旋转)
│   ├── edge_v_0_0/  (Capsule Xform: 垂直边, 绕Z旋转90°)
│   └── ...
└── target/  (Cube + RigidBody, mass=1.0 — 独立，不属于 Articulation)
```

关节链：`drone.base_link` → Fixed → `rope segs` → Fixed → `net_corner_node` → D6 → `edge_capsule` → D6 → `adjacent_node` → ...

### 网-绳索连接

- 4 个角节点 (n00, n02, n20, n22) 各通过一条 `create_rope()` 连接到对应无人机
- 绳索两端 FixedJoint 都在 articulation 内（`exclude_from_articulation=False`）
- 如用 6 架无人机：额外绳索连到边中点节点 (n01, n10, n12, n21)

## 实现文件清单

### 0. `omni_drones/utils/scene.py` — 修改 `create_rope()` （~5 行改动）

添加 `exclude_from_articulation: bool = True` 参数。将第 170 行硬编码改为参数控制。

### 1. `omni_drones/utils/scene.py` — 新增 `create_net()` 函数

```
create_net(
    xform_path: str,            # 父 Xform 路径
    rows: int = 3,              # 行数
    cols: int = 3,              # 列数
    spacing: float = 0.5,       # 节点间距 (m)
    node_radius: float = 0.02,  # 节点球半径 (m)
    node_mass: float = 0.01,    # 默认节点质量 (kg)
    corner_mass: float = 0.02,  # 角节点质量 (kg)
    edge_damping: float = 10.0, # 边 D6 关节阻尼
    edge_stiffness: float = 1.0,# 边 D6 关节刚度
    color: tuple = (0.3, 0.3, 0.3),
    enable_collision: bool = True,
) -> dict:
    # 返回 {
    #   "nodes": [[prim, ...], ...],  # rows × cols 网格
    #   "edges_h": [(prim, node_above, node_below), ...],  # 水平边
    #   "edges_v": [(prim, node_left, node_right), ...],   # 垂直边
    # }
```

实现步骤：
1. 创建 M×N 个球体节点，每个节点：
   - `UsdGeom.Sphere` + `script_utils.setRigidBody("convexHull", False)`
   - `UsdPhysics.MassAPI` 设置质量（角节点用 corner_mass，其余用 node_mass）
   - 位置：`(-cols/2 * spacing + col * spacing, rows/2 * spacing - row * spacing, 0)` 使网中心在原点
2. 创建水平边（M 行 × (N-1) 列）：
   - 每边一个父 Xform（无旋转，Capsule 沿 X 轴即水平方向）
   - Capsule 长度 = spacing
   - 两端 D6 Joint 连到左右节点（复用 `create_rope()` 的 D6 配置）
3. 创建垂直边（(M-1) 行 × N 列）：
   - 每边一个父 Xform，`AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 90))` 绕 Z 轴旋转
   - Capsule 长度 = spacing（旋转后沿 Y 轴）
   - 两端 D6 Joint 连到上下节点
4. 返回完整的节点网格和边列表供后续引用

### 2. `omni_drones/envs/net_capture/__init__.py`

```python
from .net_capture import NetCapture
```

### 3. `omni_drones/envs/net_capture/utils.py` — `NetCaptureGroup` 类

参考 `TransportationGroup`（`omni_drones/envs/transport/utils.py`），核心方法：

```python
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

class NetCaptureGroup(RobotBase):
    def spawn(self, translations, prim_paths, enable_collision=False): ...
    def initialize(self, prim_paths_expr=None, track_contact_forces=False): ...
    def apply_action(self, actions): ...
    def get_state(self, env=True): ...
    def _reset_idx(self, env_ids): ...
```

`spawn()` 的核心逻辑：
1. 对每个 env，创建 xform prim
2. 计算无人机位置（均匀分布在网四周，高度略高于网面）
3. 调用 `create_net()` 创建网（在 xform 下）
4. 对每架无人机：
   - spawn 无人机 prim（在 xform 下）
   - Unapply `ArticulationRootAPI` / `PhysxArticulationAPI`
   - 调用 `create_rope(from_prim=drone_base_link, to_prim=net_corner_node, exclude_from_articulation=False)` 连接
5. 创建 target prim（DynamicCuboid + RigidBody，gravity enabled，不属于 articulation）
6. 在 xform 上 Apply `ArticulationRootAPI` / `PhysxArticulationAPI`
7. 设置 articulation 属性（`enable_self_collisions=False`, `solver_position_iteration_count=8`）

`initialize()` 的核心逻辑：
```python
def initialize(self, prim_paths_expr=None, track_contact_forces=False):
    super().initialize(prim_paths_expr)
    self.drone.initialize(f"{self.prim_paths_expr}/{self.drone.name.lower()}_*")
    self.drone.articulation = self
    self.drone.articulation_indices = torch.arange(self.drone.n, device=self.device)

    # 网节点 RigidPrimView（按行展开，shape 可索引 grid）
    self.net_nodes_view = RigidPrimView(
        f"{self.prim_paths_expr}/net/node_*",
        reset_xform_properties=False,
        shape=(-1, self.drone.n, self.cfg.net_rows, self.cfg.net_cols),  # FIXME: 维度需验证
    )
    self.net_nodes_view.initialize()

    # 目标视图
    self.target_view = RigidPrimView(
        f"{self.prim_paths_expr}/target",
        reset_xform_properties=False,
    )
    self.target_view.initialize()
```

**注意**：RigidPrimView 的 shape 参数与 prim 数量匹配。由于每 env 有 M×N 个节点，节点在 `/World/envs/env_*/Group_*/net/node_*`。需在实际集成时验证 dimension 推导。

### 4. `omni_drones/envs/net_capture/net_capture.py` — `NetCapture` 环境

```python
class NetCapture(IsaacEnv):
    """
    多无人机绳网协同捕捉任务。
    目标：控制绳网覆盖并捕获地面上的静态目标物体。
    """
```

#### `__init__`
- 加载 drone_model、num_drones、net 参数、reward 权重、`action_scale`
- 创建 `NetCaptureGroup`
- 初始化目标位置/状态分布、reward 统计

#### `_design_scene()`
```python
def _design_scene(self):
    drone_model_cfg = self.cfg.task.drone_model
    self.drone, self.controller = MultirotorBase.make(
        drone_model_cfg.name, drone_model_cfg.controller
    )
    group_cfg = NetCaptureCfg(
        num_drones=self.cfg.task.num_drones,
        net_rows=self.cfg.task.get("net_rows", 3),
        net_cols=self.cfg.task.get("net_cols", 3),
        net_spacing=self.cfg.task.get("net_spacing", 0.5),
        rope_links=self.cfg.task.get("rope_links", 12),
    )
    self.group = NetCaptureGroup(drone=self.drone, cfg=group_cfg)
    scene_utils.design_scene()
    self.group.spawn(translations=[(0, 0, 3.0)], enable_collision=True)
    return ["/World/defaultGroundPlane"]
```

#### `_set_specs()`

**Observation（去中心化）:**
- `obs_self`：自身状态 + identity one-hot（与 TransportTrack 一致）
- `obs_others`：其他无人机相对位置 + 状态
- `obs_net`：网中心 + 4 角节点相对位置/速度（扁平化后 ~(1+4)×6=30 维，而非全部 M×N 节点）
- `obs_target`：目标相对位置

**Observation Central（集中式 Critic）:**
- `state_drones`：所有无人机状态
- `state_net`：全部 M×N 节点位置 + 速度（因为 Critic 只在训练时使用，维度大不是问题）
- `state_target`：目标位置

**Action:** 每架无人机转子推力（与 TransportTrack 相同，维度 = `num_rotors`）

**Reward:** 单值（共享 reward），reshape 为 `(num_envs, num_drones, 1)`

#### `_reset_idx(env_ids)`
- 随机采样目标位置（地面上的 x/y 坐标，z=target_half_height）
- 重置无人机到初始位置（网正上方均匀分布）
- 重置网节点姿态为平展状态（水平面）
- 重置绳索初始关节位置
- 清零速度

#### `_compute_state_and_obs()`
- 读取无人机状态 (`self.drone.get_state()`)

- 读取网节点位置、速度（通过 `self.group.net_nodes_view`）

- 计算网中心：`net_center = net_positio`根因：现有 create_rope() 在 to_prim 端的 FixedJoint 上设置 physics:excludeFromArticulation = True（scene.py:170）。这是有意设计——原用途是让绳索末端物体（如 payload）作为独立刚体不受 articulation 约束。但在 NetCapture 中，网节点彼此通过 D6 关节连接，必须全部在同一 articulation 内。

  后果：如果直接使用 create_rope(drone_base_link, net_node)，网节点会被排除在 articulation 外，导致：

  ArticulationView 无法读取网节点状态

  网与无人机构成的 articulation 树断裂

  PhysX 可能报 articulation 循环错误

  修复：给 create_rope() 添加 exclude_from_articulation: bool = True 参数，在连接网节点时传 False。只改 scene.py 一处，向后兼容。

  ￼
  # scene.py create_rope() 签名改动
  def create_rope(
      ...,
      exclude_from_articulation: bool = True,  # 默认 True 保持向后兼容
  ):
      ...
      if to_prim is not None:
          joint = script_utils.createJoint(stage, "Fixed", links[0], to_prim)
          joint.GetAttribute("physics:excludeFromArticulation").Set(exclude_from_articulation)
  在 NetCaptureGroup.spawn() 中调用 create_rope() 时传 exclude_from_articulation=False。

  问题 2 (MEDIUM): 角节点质量过轻导致 PhysX 不稳定
  根因：计划中所有节点 mass=0.005kg（5g）。角节点需要承载绳索总重（~12 段 × 0.01kg = 0.12kg）+ 无人机几牛顿的拉力。质量比 > 300:1 会导致 PhysX 关节求解器难以收敛。

  修复：

  角节点（连绳索的）：mass=0.02kg（20g）

  边缘节点（n01, n10, n12, n21 等）：mass=0.01kg

  内部节点（n11 等）：mass=0.01kg

  提高 articulation 的 solver position iteration count（默认 4→8）以改善 D6 关节稳定性

  问题 3 (MEDIUM): 网边方向旋转实现模糊
  根因：计划说"旋转父 Xform 90°"，未指定轴。Capsule 沿 X 轴放置，水平边（东西方向）沿 X 轴无需旋转，垂直边（南北方向）需旋转。两种边界造方法不同，计划未区分。

  修复：水平边 Capsule 父 Xform 不做额外旋转；垂直边 Capsule 父 Xform 绕 Z 轴旋转 90°。同时将 create_rope() 的 Xform 旋转模式明确复用：网边父 Xform 在创建 Capsule 前先 AddRotateXYZOp().Set()。

  问题 4 (MEDIUM): 观测空间维度爆炸
  根因：计划为每架无人机提供全部 M×N 个节点的相对位置。3×3=9 节点 × (3 位置 + 3 速度) = 54 维，乘以 4 架无人机 = 216 维仅用于网观测。9×9 时 81×6×N_drones 将不可承受。

  修复：观测只保留：

  网中心（质心）相对每架无人机位置

  4 个角节点相对每架无人机位置（或直接使用角节点，不包含全部节点）

  网面的法向量（用于判断覆盖姿态）

  目标相对每架无人机位置

  同时 observation_central（集中式 Critic）保留全网节点完整状态，因为 Critic 只在训练时使用。

  问题 5 (LOW): 未指定目标物体创建方式
  根因：计划只说"target (Cube + RigidBody — 独立，不属于 Articulation)"，缺少具体创建方式。

  修复：用 DynamicCuboid（Isaac Sim API，与 TransportHover 中的 payloadTargetVis 一致），或手动创建 Cube prim + RigidBodyAPI + MassAPI。

  问题 6 (LOW): 网节点 RigidPrimView 索引方案
  根因：需要读取特定节点（如角节点 vs 内部节点）的状态来计算奖励/观测。RigidPrimView 按 prim path 字母序排列 prims。

  修复：命名时保证字母序与网格顺序一致：node_0_0, node_0_1, ..., node_2_2。这样 RigidPrimView 返回的张量维度 (-1, rows, cols, ...) 可以直接索引。

  📋 代码审查要点（实现时检查）
  以下是基于现有代码库模式识别的潜在陷阱：

- `ns.mean(dim=(-3, -2))`

- 提取角节点位置（索引 [0,0], [0,-1], [-1,0], [-1,-1]）

- 读取目标位置

- 组装观测 TensorDict

#### `_compute_reward_and_done()`
奖励设计（更精确的定义）：

| 奖励项 | 公式 | 权重 | 说明 |
|--------|------|------|------|
| `reward_coverage` | `exp(-distance(net_center_xy, target_xy) * scale)` | 1.0 | 网水平面接近目标 |
| `reward_descend` | `exp(-max(0, net_center_z - target_top_z - margin))` | 0.5 | 网下降至目标高度附近 |
| `reward_capture` | 节点低于目标顶面 + 网中心覆盖的 bonus | 2.0 | 触发条件：至少 3 个节点 z < target_top_z 且 xy 距离 < 阈值 |
| `reward_up` | `(drone_up · world_up + 1) / 2` 的平方 | 0.1 | 无人机保持 upright |
| `reward_effort` | `exp(-effort).mean()` | 0.05 | 能量消耗惩罚 |

终止条件：
- **成功**：网覆盖目标持续 N 步（可选，先不做自动终止）
- **失败**：任一无人机高度 < 0.2m（crash）
- **失败**：网中心偏离目标超过 `reset_thres`（如 2.0m）
- **保护**：NaN 检测

### 5. `cfg/task/NetCapture/NetCapture.yaml`

```yaml
name: NetCapture

defaults:
  - /base/env_base@_here_
  - /base/sim_base@_here_

env:
  num_envs: 32
  max_episode_length: 600

drone_model:
  name: Hummingbird
  controller: LeePositionController

num_drones: 4

# 绳网参数
net_rows: 3
net_cols: 3
net_spacing: 0.5        # 节点间距 (m)
rope_links: 12           # 每条绳索段数
rope_link_length: 0.06   # 每段长度 (m)

# 目标参数
target_size: [0.3, 0.3, 0.3]  # 目标物尺寸 (m)

# 奖励权重
reward_coverage_weight: 1.0
reward_capture_weight: 2.0
reward_descend_weight: 0.5
reward_up_weight: 0.1
reward_effort_weight: 0.05
reward_action_smoothness_weight: 0.0

# 奖励 scale
reward_distance_scale: 2.0

# 终止条件
reset_thres: 2.0         # 网中心偏离目标太远则 reset
capture_thres: 0.3       # 网中心距目标 xy < thres 视为覆盖

# 无人机碰撞安全距离
safe_distance: 0.5

action_scale: 1.0

ravel_obs: true
ravel_obs_central: true
action_transform: null
```

### 6. `omni_drones/envs/__init__.py` — 注册

```python
from .net_capture import NetCapture
```

## 实现顺序

| 步骤 | 文件 | 内容 | 估计行数 | 依赖 |
|------|------|------|---------|------|
| 0 | `scene.py` | 修改 `create_rope()` 添加 `exclude_from_articulation` 参数 | ~5 | 无 |
| 1 | `scene.py` | 新增 `create_net()` 函数 | ~150 | 步骤 0 |
| 2 | `net_capture/utils.py` | 新建 `NetCaptureGroup` 类 | ~280 | 步骤 0,1 |
| 3 | `net_capture/net_capture.py` | 新建 `NetCapture` 环境 | ~350 | 步骤 0-2 |
| 4 | `net_capture/__init__.py` | 包初始化 | ~1 | 步骤 3 |
| 5 | `cfg/task/NetCapture/NetCapture.yaml` | Hydra 配置 | ~40 | 步骤 3 |
| 6 | `envs/__init__.py` | 注册环境 | +1 | 步骤 4 |

## 潜在风险 & 应对

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 大量 D6 关节导致 PhysX 不稳定 | 训练崩溃 | 提高 solver_position_iteration_count 到 8-12；使用阻尼为主的关节配置（damping >> stiffness） |
| 网格节点过多导致训练慢 | 低帧率 | 默认 3×3（12 边）；后续按需增加，监控 FPS |
| 网与地面碰撞导致弹跳/穿透 | 不真实物理 | 调高 rest_offset，增加接触阻尼；可在地面使用 patch friction |
| 绳索初始化时姿态异常 | 关节初始冲突 | 在 `_reset_idx` 中不重置关节位置（让物理稳定后再开始 episode） |
| 多关节 articulation GPU OOM | 无法训练 | 从 32 envs 开始，`record_video=false`，必要时降至 16 envs |
| D6 关节 rotY/rotZ limits 冲突 | PhysX 报错 | limits 设置对称 ±110 度，与 `create_rope()` 一致，已验证无问题 |
| 角节点受力过大 | 关节断裂 | 角节点质量 0.02kg 为内部节点的 2 倍；后续可加 linear damping |

## 训练命令（预期）

```bash
# 阶段 1: 最小验证 — 确认场景加载和物理无 crash
python train.py algo=mappo headless=true task=NetCapture/NetCapture \
  total_frames=100 wandb.mode=offline

# 阶段 2: 初始训练 — Hummingbird + 低帧数
python train.py algo=mappo headless=true task=NetCapture/NetCapture \
  total_frames=10_000_000 wandb.mode=offline \
  algo.entropy_coef=0.01

# 阶段 3: DifferentialUAV + action_scale
python train.py algo=mappo headless=true task=NetCapture/NetCapture \
  task.drone_model.name=DifferentialUAV \
  total_frames=20_000_000 wandb.mode=offline \
  algo.actor.lr=0.0001 algo.critic.lr=0.0001 \
  algo.max_grad_norm=1.0 algo.entropy_coef=0.01 \
  task.action_scale=0.5
```

## 验证步骤

1. `python train.py ... total_frames=10` — 确认 USD stage 加载无错误
2. `python train.py ... total_frames=100` — 确认 PhysX 无 D6 joint limit 冲突报错
3. 检查 articulation DOF count / body count 是否符合预期
4. 在有 GUI 模式下检查网结构视觉正确（无重叠/穿透、绳索/节点方向正确）
5. 训练几个 epoch 后检查 reward 是否上升、entropy 是否正常（> 0.5）
6. 评估视频中确认网能与目标交互

## 后续扩展路径

- **更多节点**：修改 `net_rows` / `net_cols` → 5×5, 7×7 等，`create_net()` 通用
- **更多无人机**：`num_drones=6/8`，额外绳索连到边中点节点
- **移动目标**：参考 `TransportTrack._compute_traj()` 添加目标轨迹
- **动态网张力控制**：通过 D6 drive stiffness/damping 参数调节网面刚度（环境级参数）

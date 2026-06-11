# NetCapture 环境代码详解

> 供人工审阅用。逐方法解释 `net_capture.py` 的数据流和物理状态。

---

## 0. 空间坐标系

训练时 `_step` 中调用 `self.drone.get_state()` 和 `get_env_poses()` 等，返回的是 **env 帧坐标**（减去各 env 的 GridCloner 偏移）。下面全部用 env 帧坐标分析，不再每次注明。

### 物理 Hierarchy

```
/World/envs/env_0/
├── Group_0/                          (普通 Xform, 无 Articulation, z=3.0)
│   ├── net/                          (net Xform, z=0 in group → env z=3.0)
│   │   ├── node_0_0/                 (Sphere, RigidBody, mass=0.01~0.02)
│   │   ├── node_0_1/ ...
│   │   ├── edge_h_0_0/capsule        (Xform+Capsule, RigidBody, D6→左右节点)
│   │   └── edge_v_0_0/capsule        (Xform+Capsule, RigidBody, D6→上下节点)
│   ├── rope_0/seg_0 .. seg_11        (Capsule chain, Fixed→drone, Fixed→corner)
│   ├── rope_1/...
│   ├── rope_2/...
│   └── rope_3/...
├── hummingbird_0/                    (独立 Articulation, z=3.0+0.825=3.825)
├── hummingbird_1/
├── hummingbird_2/
├── hummingbird_3/
├── target/                           (Sphere, RigidBody, no gravity, no collision)
└── target_heading/                   (Capsule arrow, RigidBody, no gravity)
```

### 关键坐标（env 帧）

| 对象 | Z 坐标 | 计算 |
|------|--------|------|
| Group Xform | 3.0 | `spawn(translations=[(0,0,3.0)])` |
| 网节点 | 3.0 | group z + net z = 3.0 + 0 = 3.0 |
| 无人机 | 3.825 | group z + z_drone = 3.0 + 0.825 = 3.825 |
| 绳索有效长度 | 0.825m | `(12-1) * 0.1 * 0.75 = 0.825` 恰好 = drone 高度差 |
| 目标 | [2.0, 3.5] | 随机分布，略低于网面，靠近网面 |
| Target heading | 单位向量, z ≥ 0.5 | 球坐标 θ∈[0, π/3], φ∈[0, 2π] |

### 无人机-网角对应

| 无人机 | 网角 (r, c) | 组内 (x, y) 坐标 |
|--------|-------------|-----------------|
| drone_0 | (0, 0) | (-x_offset, +y_offset) |
| drone_1 | (0, C-1) | (+x_offset, +y_offset) |
| drone_2 | (R-1, 0) | (-x_offset, -y_offset) |
| drone_3 | (R-1, C-1) | (+x_offset, -y_offset) |

无人机水平位置与对应网角完全对齐（正上方）。

---

## 1. `__init__` — 初始化

### 1.1 读取配置 (L94-103)

从 `cfg.task.*` 读取奖励权重、阈值、动作缩放系数。部分配置键名保留历史遗迹（如 `reward_descend_weight` 实际是 heading alignment 权重）。

### 1.2 缓存观测维度 (L105-108)

`net_rows`, `net_cols`, `n_nodes` 必须在 `super().__init__()` 之前赋值，因为 `IsaacEnv.__init__` 内会立即调用 `_set_specs()` 构造观测/动作空间。

### 1.3 `super().__init__(cfg, headless)` (L110)

`IsaacEnv.__init__` 内部调用链：
1. `_design_scene()` → 创建 USD 场景
2. GridCloner.clone() → 复制 env_0 到 N 个 env
3. `sim.reset()` → 物理引擎初始化
4. `_set_specs()` → 定义 observation/action/reward spec

### 1.4 创建视图 (L112-126)

- `self.group.initialize()`: 初始化 drone ArticulationView + net_nodes RigidPrimView
- `self.target`: 目标球体的 RigidPrimView
- `self.target_heading_view`: 方向箭头的 RigidPrimView

### 1.5 缓存初始状态 (L128-135)

```
init_drone_pos: (N*4, 3) — 所有 env 中所有无人机的世界坐标（spawn 后即读取）
init_drone_rot: (N*4, 4)
init_drone_vels: (N*4, 6) — 全零
init_net_pos: (N, n_nodes, 3) — 所有网节点世界坐标
init_net_vel: (N, n_nodes*6) — 全零
```

这些缓存用于 `_reset_idx` 将系统恢复到初始状态。

**潜在问题**：绳索段 (seg_0~seg_11)、边胶囊 (edge capsules) 的初始位置没有缓存——它们的复位依赖两端 D6 关节约束自动拉回。在两端（无人机+网节点）都被复位到初始位置后，关节约束会逐步恢复中间体位置，但初帧可能存在残余速度/位移。

### 1.6 目标分布 (L137-144)

- `target_pos_dist`: XYZ ∈ [[-1,-1,2], [1,1,3.5]] — 网面高度 3.0，目标在网面附近
- `target_heading_vec`: 在 `_reset_idx` 中球坐标采样，不在 `__init__` 中定义分布

---

## 2. `_design_scene` — 场景创建

### 2.1 创建 Group (L148-161)

```python
self.group = NetCaptureGroup(drone=self.drone, cfg=group_cfg)
```

`NetCaptureGroup.__init__`:
- `drone.is_articulation = False` — 无人机作为独立 RigidBody，通过 `apply_action` 施力，不通过 ArticulationView 控制
- 存储 net 参数（rows, cols, spacing, rope_links, etc.）

### 2.2 `self.group.spawn(translations=[(0,0,3.0)])` (L165)

在 `env_0` 下创建完整的 USD 层级：
1. Group Xform at z=3.0
2. `create_net()` — 节点+边（D6 关节锁定 trans+rotX，rotY/Z 有 angular drive）
3. 4 架无人机 spawn + connect via `create_rope()`（rope xform 在无人机位置，FixedJoint 两端都 `excludeFromArticulation=True`）

**物理连接方式**：绳索两端 FixedJoint 连接 `drone base_link` ↔ `net corner node`。中间 12 个 Capsule 通过 D6 关节串接（rotY/Z 自由 + damping/stiffness drive）。

`from_prim=corner_node`（网角—接链条底端 links[-1]）, `to_prim=drone_base_link`（无人机—接链条顶端 links[0]）。

链条在 xform 内沿 +X 展开，xform 旋转 (0,90,0) 使 +X→世界-Z（向下）。links[0] 在 xform 原点=无人机位置，links[-1] 在下方 `(N-1)*0.75*L` ≈ 0.825m 处=网角位置。

**与 TransportTrack 的区别**：TransportTrack 的 Group 是单一 Articulation（所有 drone+payload 在一个 articulation tree 内）。NetCapture 的 Group 不是 articulation——网节点间有闭合环路（2D 网格的四边形），PhysX articulation 要求树形拓扑。

### 2.3 目标 Marker (L167-195)

纯视觉（RigidBody 但 disable_gravity + no collision）：
- 小球 Sphere (r=0.06) → 目标位置指示
- 长 Capsule (h=0.3, axis=X) → 方向指示，旋转 quat 在 reset 时更新

### 2.4 返回值 (L196)

`["/World/defaultGroundPlane"]` — 全局共享的地面，GridCloner 不会为每个 env 复制。

---

## 3. `_set_specs` — 观测/动作空间

### 3.1 观测维度

```
drone_state_dim = drone.state_spec[-1] + n_drones
    = (19 + n_rotors) + 4     Hummingbird: 19+4+4 = 27
    drone.get_state() 返回 [pos(3), rot(4), vel(6), heading(3), up(3), throttle(4)] = 23
    实际是 23 + 4(identity) = 27  ← 确认
```

| 观测键 | 维度 | 内容 |
|--------|------|------|
| `obs_self` | (1, 27) | 网中心相对位置(3) + drone状态(23) + identity(4) |
| `obs_others` | (3, 14) | 其他无人机相对位置(3) + 距离(1) + 状态(10) |
| `obs_net` | (1, 30) | 网中心(6) + 4角(4×6) |
| `obs_target` | (1, 6) | 目标位置(3) + 方向(3) |

集中式 Critic 观测：

| 状态键 | 维度 | 内容 |
|--------|------|------|
| `state_drones` | (4, 27) | 全部无人机状态 |
| `state_net` | (36, 6) | 全部网节点位置+线速度 |
| `state_target` | (1, 6) | 目标位置+方向 |

### 3.2 动作空间

```
action: (4, 4) — 4 架无人机 × 4 个旋翼 cmd ∈ [-1, 1]
```

共享 reward：`reward: (4, 1)` — 每无人机相同值。

---

## 4. `_reset_idx` — Episode 复位

**调用时机**：
1. 训练开始时（首次 reset，env_ids = 全部 env）
2. Episode 终止时（只有终止的环境被 reset）

### 4.1 采样目标 (L256-258)

```python
target_pos = self.target_pos_dist.sample(env_ids.shape)
self.target_pos[env_ids] = target_pos
```

只随机化目标点——无人机和网的初始位置不随机化。

### 4.2 无人机复位 (L260-277)

1. `drone._reset_idx(env_ids)` — 内部状态：油门设为悬停值（gravity/KF），重新随机化物理参数（如果配置了 randomization）
2. `drone.set_world_poses(init)` — 位置回到 spawn 时的世界坐标
3. `drone.set_velocities(zeros)` — 速度归零

### 4.3 网节点复位 (L279-292)

1. 位置回到 `init_net_pos`
2. 速度归零

`net_nodes_view` 是 1D 视图 `(N×n_nodes,)`，使用 `env_indices` 时需要构造扁平索引：
```python
node_env_ids = (env_ids * n_nodes + arange(n_nodes)).reshape(-1)
```

**⚠️ 未复位的组件**：绳索段、边胶囊——它们的初始位置未显式设置。复位后第 0 帧，它们可能处于上一轮 episode 结束时的位置/速度。但由于两端（无人机+网节点）已复位到 spawn 位置，D6 关节约束会施加恢复力，物理系统在若干步内收敛到稳定状态。

### 4.4 目标方向 (L294-303)

球坐标采样，θ（与竖直夹角）∈[0, π/3]，φ（方位角）∈[0, 2π]。方向始终在上半球（z ≥ 0.5）。

### 4.5 视觉标记 (L305-329)

- 目标球体：移到位
- 方向箭头：计算从 +Z 到 heading 的旋转 quaternion，箭头偏移 target + 0.15*heading（箭尾在目标，箭头前伸）

---

## 5. `_pre_sim_step` — 动作应用

```python
actions = tensordict[("agents", "action")]      # (N, K, 4)
self.effort = self.drone.apply_action(actions * self.action_scale)
```

`drone.apply_action` (MultirotorBase, `is_articulation=False`):
1. 将 cmd∈[-1,1] 转换为 throttle 和 thrust：`throttle = sqrt(clamp((cmd+1)/2, 0, 1))`, `thrust = throttle²/KF`
2. 通过 `rotors_view.apply_forces_and_torques_at_pos()` 施加旋翼力
3. 通过 `base_link.apply_forces_and_torques_at_pos()` 施加力矩和阻力

**不会设置旋翼关节速度**（`is_articulation=False` 时跳过），但物理正确——力直接施加在 RigidBody 上。

`action_scale=0.1`：将策略输出的幅度压缩到 ±0.1。这意味着随机策略（输出 ≈ N(0, σ)）只会在悬停油门基础上做 ±10% 的微调，防止随机动作把网"扯散"。

---

## 6. `_compute_state_and_obs` — 观测组装

### 6.1 读取网节点状态 (L340-347)

```python
net_pos_world, _ = self.group.net_nodes_view.get_world_poses()
# flat (N*n_nodes, 3) → reshape (N, n_nodes, 3)
net_pos, _ = self.get_env_poses((net_pos_world, None))
```

### 6.2 网中心和法向量 (L353-376)

- 中心：所有节点位置的均值
- 法向量：4 个角节点叉积 `(n02-n00) × (n20-n00)`

**⚠️ 网法向量的方向取决于角节点顺序**。当网折叠或极为倾斜时，叉积可能退化。至少需要 3 个角节点不共线。

### 6.3 观测组装 (L392-429)

- `obs_self`: 无人机自身状态，位置用**网中心**作为参考点（而不是世界原点或 group 原点）
- `obs_others`: 其他无人机的相对位置+状态
- `obs_net`: 网中心(3+3 vel) + 4 角节点(3+3 lin-vel each) → 30D
- `obs_target`: 目标相对位置(3) + 方向向量(3) → 6D

集中式 critic 状态不展开 per-drone 维度，直接给出全局网节点状态和目标。

### 6.4 统计量 (L442-452)

指数移动平均 (α=0.8)：`stat_new = stat_old + (1-α) * (current - stat_old)` = `stat_old + 0.2 * delta`。

---

## 7. `_compute_reward_and_done` — 奖励+终止

### 7.1 奖励项

| 项 | 公式 | 范围 | 含义 |
|----|------|------|------|
| r_pos | `exp(-β·dist)` | (0, 1] | 越大越接近目标 |
| r_head | `((n·v+1)/2)²` | [0, 1] | 法向量与目标方向对齐度 |
| r_phead | `r_pos × r_head` | [0, 1] | 耦合：靠近后才关注朝向 |
| r_up | `mean(((uz+1)/2)²)` | [0, 1] | 无人机保持水平 |
| r_eff | `-0.05 × mean(throttle)` | (-0.05, 0] | 能耗惩罚 |
| r_survive | `0.02 × sigmoid((z-1.0)×2)` | (0, 0.02) | 网高度 >1m 时存活奖励 |
| r_sep | `min(1, min_dist/0.8)²` | [0, 1] | 无人机碰撞安全因子（乘法） |

总奖励：`r = r_sep × (1.5·r_pos + 1.0·r_phead + 0.1·r_up + r_eff + r_smooth + r_survive)`

### 7.2 终止条件

```
terminated: drone z < 0.2    (任意无人机坠地)
           ‖p_net - p_tgt‖ > 8.0  (网偏离目标超 8m)
           NaN in drone_states
truncated:  step ≥ 600
```

---

## 8. 已知问题 & 待验证

### 8.1 绳索/边胶囊未显式复位

`_reset_idx` 不直接设置绳索段和边胶囊的位置。两端物体复位后，D6 关节约束产生恢复力，中间体在若干步后稳定。极端情况（上一轮完全绞缠）可能导致复位后初帧出现短暂的超大关节力。

**解决方案**（未实现）：为绳索段和边胶囊各自创建 RigidPrimView，在 reset 时也设置为初始位置。

### 8.2 RigidPrimView 导入混用

`utils.py` 最初导入 `omni.isaac.core.prims.RigidPrimView`（基类，不支持 `env_indices`），后来改为 `omni_drones.views.RigidPrimView`（自定义，支持 `env_indices`）。确保两个地方一致。

### 8.3 网节点 1D 视图索引

`net_nodes_view` 没有显式 shape → 默认 1D `(N×n_nodes,)`。reset 时需要构造正确的扁平索引来选中某个 env 的所有节点。

### 8.4 法向量接近奇异

当网折叠、四个角节点共线或极度靠近时，叉积退化 → 法向量接近零 → `torch.nn.functional.normalize` 可能产生 NaN。当前未做 NaN 防护。

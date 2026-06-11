# PBD Particle Rope 替代 D6 Joint Rope 实现计划

> **Status:** IMPLEMENTED — 场景搭建成功，训练验证通过（5k 帧，零 PhysX 错误）
> **Branch:** `feature/pbd-particle-rope`
> **Last updated:** 2026-06-11

**Goal:** 用 PhysX GPU 原生的 PBD 粒子系统替代 D6 joint 链条实现绳缆，从根本上消除 GPU 模式下 PxRigidDynamic CPU API 非法调用问题。

**Architecture:** 一个 `ParticleSystem` 包含所有 rope 粒子，每条 rope 是窄带 2-row mesh 配置为 `ClothPrim`（高 stretch stiffness = 绳缆特性）。端点粒子通过 `PhysxPhysicsAttachment` + `PhysxAutoAttachmentAPI` 固定到 drone base_link / net corner。使用 `ClothPrimView` 做多环境 tensor 批量操作。

**Tech Stack:** Isaac Sim 4.1.0, PhysX 5 PBD particle system, `ClothPrimView` / `ParticleSystem`, `PhysxPhysicsAttachment`, OmniDrones `NetCaptureGroup`

**风险提示:** `add_physx_particle_cloth` API 已标记 DEPRECATED。虽然在 Isaac Sim 4.1.0 仍可用，未来版本可能移除。备选方案：手动胡克定律力计算（GPU 兼容）。

---

## 核心实现要点

### 1. 窄带 2-row mesh 替代退化三角形

**问题:** 1D 粒子链（退化三角形 `[i, i+1, i]`）无法让 `PhysxAutoParticleClothAPI` 生成有效 spring 约束。

**解决:** 使用 2-row 窄带 mesh（宽 0.02m），每个 quad 拆成 2 个有效三角形 `[a, c, b, a, d, c]`。Auto-spring API 从 mesh 边生成 stretch/bend/shear springs。

```python
# 2-row strip: N lengthwise × 2 widthwise = 2N vertices
positions = [
    Gf.Vec3f(start + dir * (i * length / (N-1)) + offset) for i in range(N)
] + [
    Gf.Vec3f(start + dir * (i * length / (N-1)) - offset) for i in range(N)
]
# Quads → triangles
for k in range(N - 1):
    a, b, c, d = k, k + 1, N + k, N + k + 1
    face_indices.extend([a, c, b, a, d, c])
```

### 2. 粒子位置必须与附着目标精确重合（⚠️ 最易出错的点）

**关键 Bug:** 第一版实现将 rope 粒子沿 X 轴放置，然后整体平移。但 drone → net corner 的实际方向因角点不同而异（上左/上右/下左/下右），导致 rope 端点偏离目标 1m+。`PhysxAutoAttachmentAPI` 的 proximity search 找不到附着目标。

**Fix:** `create_pbd_rope()` 接收 `start_pos`/`end_pos` 参数，粒子沿实际向量插值：

```python
# 计算从 drone (start_pos) 到 net corner (end_pos) 的实际方向
direction = end_pos - start_pos
# 粒子沿此方向等距排布
for i in range(N):
    positions[i] = start_pos + dir_norm * (i * actual_length / (N - 1))
```

**结果:** 端点偏差从 1m+ 降至 <0.01m（1cm），proximity search 能够匹配。

### 3. 坐标系：粒子在 Group_0 本地空间

rope mesh 是 `Group_0` Xform 的子节点。粒子位置必须在 Group_0 本地坐标系中，而非世界坐标。

```python
# ✅ 正确：Group_0 本地坐标
start_pos = drone_offset.tolist()     # (cx, cy, z_drone)
end_pos = [cx, cy, 0.]                # net z=0 in group-local frame

# ❌ 错误：世界坐标（会导致位置偏移 translation）
start_pos = (translation + drone_offset).tolist()
```

### 4. sim.reset() 必须在 ClothPrimView.initialize() 之前

PBD 粒子位置只能通过 `ClothPrimView.get_world_positions()` 读取（GPU 动态状态），USD mesh `GetPointsAttr()` 返回静态属性。

```python
sim.reset()                    # ✅ 先创建 physics scene
cv.initialize()                # ✅ 再初始化 tensor view
positions = cv.get_world_positions()  # GPU 粒子位置
```

### 5. PhysxPhysicsAttachment 使用 PhysxAutoAttachmentAPI

`omni.kit.commands.execute("CreatePhysicsAttachment", ...)` 创建 `PhysxPhysicsAttachment` + `PhysxAutoAttachmentAPI`。Auto-attachment 基于空间 proximity 自动选择顶点。**必须确保 rope 端点粒子与目标刚体质心重叠**（偏差 < 几厘米），否则自动附着失败。

### 6. D6 joint net edges 的 GPU 兼容性

PBD rope 只替代了 rope（绳缆），**net edges 仍使用 D6 joints**。这些 D6 joints 在 GPU 模式下可能触发 CPU API 回退。`views/__init__.py` 的 GPU/CPU 双路径修复处理了此问题（`_invalidate_physics_handle_callback` 清空 `_physics_view` 时走 USD fallback）。

---

## 实施记录

### Phase 0: 可行性验证 ✅

| Test | 内容 | 结果 |
|------|------|------|
| 0a | 粒子链自由下落 | z: 3.00→-15.83 (120 步) |
| 0b | rope + 2 个 kinematic 刚体 | 端点 z=2.00 不动，中点下垂至 1.92 |
| 0c | rope + drone + net corner | 端点跟随 drone 从 1.32→0.01 |

**关键发现:** 窄带 2-row mesh 可行；`sim.reset()` 必须在 `cv.initialize()` 之前；USD `GetPointsAttr()` 读不到 GPU 动态数据。

### Phase 1: `create_pbd_rope()` ✅

文件: `omni_drones/utils/scene.py`

`create_pbd_rope(start_pos, end_pos, particle_system_path, from_prim, to_prim, ...)`:
- 2-row mesh strip 沿 start→end 向量
- 创建 `PhysxParticleSystem` + material（幂等，避免重复）
- `particleUtils.add_physx_particle_cloth()` auto-spring
- `omni.kit.commands.execute("CreatePhysicsAttachment")` × 2

### Phase 2: NetCaptureGroup 集成 ✅

文件: `omni_drones/envs/net_capture/utils.py`

- `NetCaptureCfg.use_pbd_rope: bool = True`（默认 PBD）
- `spawn()`: 条件分支 → `create_pbd_rope()` vs `create_rope()`
- `initialize()`: `ClothPrimView` vs `RigidPrimView`
- 配置写入 `cfg/task/NetCapture/NetCapture.yaml`

### Phase 3: 环境管线适配 ✅

文件: `omni_drones/envs/net_capture/net_capture.py`

- `_reset_idx`: PBD rope 跳过 rope 位姿/速度重置（attachment 自动跟随刚体）
- `__init__`: rope segs init 缓存条件化
- Rope 状态暂不加入 observation

### Phase 4: 端到端验证 ✅

| 测试 | 结果 |
|------|------|
| 场景 USD 导出 | ✅ USDA 102MB，含 4 条 PBD rope |
| 5k 帧训练 | ✅ episode_len=599, return=155.46, 零 PhysX 错误 |
| rope 端点精度 | ✅ 偏差 < 0.01m（target < 0.1m） |

---

## 文件结构（实际）

| 文件 | 变更 | 职责 |
|------|------|------|
| `omni_drones/utils/scene.py` | MODIFY | `create_pbd_rope()` — PBD rope 工厂函数 |
| `omni_drones/envs/net_capture/utils.py` | MODIFY | `NetCaptureCfg` PBD 参数, `spawn()`/`initialize()` 双模式 |
| `omni_drones/envs/net_capture/net_capture.py` | MODIFY | `_reset_idx` 条件化 rope 重置 |
| `omni_drones/views/__init__.py` | MODIFY | GPU/CPU 双路径 fallback（D6 net edges 兼容） |
| `omni_drones/envs/isaac_env.py` | MODIFY | `_reset` 中 `sim.step()` after reset |
| `cfg/task/NetCapture/NetCapture.yaml` | MODIFY | PBD rope 配置项 |
| `cfg/base/sim_base.yaml` | MODIFY | `use_flatcache: false` |
| `scripts/debug_pbd_rope.py` | CREATE | Phase 0 增量诊断脚本 |
| `scripts/test_net_capture_scene.py` | MODIFY | USDA 导出 + export 时机修复 |
| `quality_reports/debug/netcapture_physx_diagnosis.md` | CREATE | D6 joint GPU 兼容性诊断报告 |

---

## 剩余问题

### 1. PhysxAutoAttachmentAPI 在小刚体上的可靠性

Net corner node 是半径 0.02m 的球体。Auto-attachment 依赖空间 proximity，小球体可能在某些情况下被跳过。

**改进方案:**
- 增大 corner node 碰撞体半径
- 或使用显式 `vertexIndices0` 代替 AutoAttachment
- 或改用备选方案 A（手动计算绳缆力）

### 2. PBD cloth 在长时间仿真中的数值稳定性

PBD 粒子在极端拉伸/压缩时可能发散。

**缓解措施:**
- `solver_position_iterations` 从 16 提高到 32
- 监控 `ClothPrimView.get_world_positions()` 中的 NaN

### 3. ClothPrimView 的 multi-env 支持

当前 `ClothPrimView` 的 `particle_systems` 参数指向模板环境的 particle system。克隆后的 env_1/2/3 的 particle system 路径不同，可能需要每个环境独立的 `ClothPrimView`。

**当前状态:** 已验证 1-env 场景（`test_net_capture_scene.py`）。4-env 训练只验证了初始化流程（未验证 rope 状态读取）。

---

## 备选方案

### 方案 A: 手动计算绳缆力（最可靠）

不依赖 PhysX joint/attachment 约束：
- 每 step 读取 `ClothPrimView.get_world_positions()` 获取 rope 端点粒子位置
- 读取 `RigidPrimView` 获取 drone/net 端点位置
- 位移差 → 胡克定律 `F = -k * (x - x_rest)` + 阻尼 `-d * v`
- 用 `RigidPrimView.apply_forces_and_torques_at_pos()` 施加到两端刚体
- **100% GPU 兼容** — 只有 tensor 操作和 force API

### 方案 B: 保留 D6 joints + 优化 GPU 回退（当前 main 分支方案）

```yaml
use_pbd_rope: false
```
使用 D6 joint rope + `views/__init__.py` GPU/CPU 双路径，接受 PhysX warning 噪音。

---

## 验证 checklist

- [x] Phase 0: 最小 PBD rope 脚本无 PhysX 错误
- [x] Phase 1: `create_pbd_rope()` 可独立创建 rope
- [x] Phase 2: NetCaptureGroup 可同时管理 PBD rope + D6 net
- [x] Phase 3: `_reset_idx` 正确处理 PBD rope
- [x] Phase 4: 训练 5k 帧稳定运行，零 PhysX 错误
- [x] Scene USD 导出正确（USDA 102MB）
- [x] Rope 端点位置偏差 < 0.01m
- [ ] 4-env full training (5M frames) — 待验证
- [ ] Net corner attachment 在长时间仿真中的可靠性 — 待观察

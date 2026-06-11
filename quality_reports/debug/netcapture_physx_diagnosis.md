# NetCapture PhysX GPU API 问题诊断报告

**日期**: 2026-06-11
**Isaac Sim 版本**: 4.1.0
**结论**: 环境逻辑正确，GPU 模式下 PhysX timeline STOP 事件导致 `RigidPrimView._physics_view` 被清空，GPU API 回退到 CPU 路径触发 PxRigidDynamic 非法 API 调用。

---

## 1. 增量诊断结果

诊断脚本 `scripts/debug_netcapture.py` 分 5 步测试，**全部通过**：

| Step | 内容 | 结果 | PhysX 错误 |
|------|------|------|------------|
| 1 | 单 drone (articulation), 1 env | PASS | 无 |
| 2 | drone + net (无连接), 1 env | PASS | 无 |
| 3 | drone + net + rope D6 joints, 1 env | PASS | 无 |
| 4 | 完整场景, 4 env (GridCloner) | PASS | 无 |
| 5 | 完整训练级 reset + apply_action + mass randomization | PASS | 无 |

**关键结论**: 环境 USD 结构、D6 rope joints、多环境克隆、GPU dynamics 本身都没有问题。

---

## 2. 训练错误根因分析

### 2.1 错误链条

```
训练启动
  → IsaacEnv.__init__()
    → SimulationContext.__init__(sim_params={...})
    → _design_scene() → group.spawn() → 包含 _strip_articulation()
    → cloner.clone() → 4 环境
    → sim.reset()          # timeline PLAY
  → NetCapture.__init__()
    → group.initialize()   # 创建 5 个 RigidPrimView
      → RigidPrimView.initialize()
        → base._RigidPrimView.initialize()
          → 创建 _physics_sim_view
          → 创建 _physics_view (GPU tensor handle)
          → 订阅 timeline event → _invalidate_physics_handle_callback
          → post_reset() → set_velocities() → 此时 _physics_view 有效
    → 缓存 init poses (get_world_poses 成功)
  → env.reset() (torchrl 触发)
    → _reset() → _reset_idx(env_ids)
      → drone.set_world_poses(d_pos, d_rot, env_ids)
        → RigidPrimView.set_world_poses()
          → self._physics_view is None?  ← 关键判断点
```

### 2.2 根本原因

**Isaac Sim 基类 `_RigidPrimView` (rigid_prim_view.py:285-288):**

```python
def _invalidate_physics_handle_callback(self, event):
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        self._physics_view = None  # ← GPU 句柄被清除
        self._invalidate_physics_handle_event = None
```

训练管线中某处触发了 timeline STOP 事件（可能是 `sim.reset()` 或 SimulationContext 初始化流程），导致所有 `RigidPrimView` 的 `_physics_view` 被置为 `None`。

### 2.3 为何会触发 PhysX 错误

当 `_physics_view is None` 时，`RigidPrimView.set_world_poses()` / `set_velocities()` 回退到基类的 CPU 路径：

```
RigidPrimView.set_world_poses()  [views/__init__.py:463]
  → _RigidPrimView.set_world_poses()  [rigid_prim_view.py:350]
    → XFormPrimView.set_world_poses()  [xform_prim_view.py:939]
      → 写 USD transform 属性
        → PhysX USD 插件拦截
          → PxRigidDynamic::setGlobalPose()  ← CPU API, GPU 模式非法!
```

同理 `set_velocities()` → `set_linear_velocities()` → 写 `physics:velocity` 属性 → `PxRigidDynamic::setLinearVelocity/AngularVelocity()`。

### 2.4 为何诊断通过但训练失败

诊断脚本中 `SimulationContext` 使用最简参数创建，timeline 行为可能与训练不同。训练中 `IsaacEnv` 传递完整 `sim_params`（包括 `solver_type`, GPU buffer sizes 等），可能触发额外的 timeline STOP 事件。

---

## 3. 已实施的修复

### 3.1 `omni_drones/views/__init__.py` — RigidPrimView GPU/CPU 双路径

```python
def set_world_poses(self, positions=None, orientations=None, env_indices=None):
    indices = self._resolve_env_indices(env_indices)
    if self._physics_view is not None:
        # GPU 路径：直接操作 tensor
        with disable_warnings(self._physics_sim_view):
            poses = self._physics_view.get_transforms()
            if positions is not None:
                poses[indices, :3] = positions.reshape(-1, 3)
            if orientations is not None:
                poses[indices, 3:] = orientations.reshape(-1, 4)[:, [1, 2, 3, 0]]
            self._physics_view.set_transforms(poses, indices)
            return
    # CPU 回退：_physics_view 被清除时走 USD 路径
    with disable_warnings(self._physics_sim_view):
        return _RigidPrimView.set_world_poses(
            self,
            positions=positions.reshape(-1, 3) if positions is not None else None,
            orientations=orientations.reshape(-1, 4) if orientations is not None else None,
            indices=indices,
        )
```

`set_velocities()` 同理。

### 3.2 `omni_drones/envs/isaac_env.py` — reset 后 sim.step()

```python
def _reset(self, tensordict, **kwargs):
    ...
    self._reset_idx(env_ids)
    self.sim.step(render=False)  # ← 取消注释，让 GPU pipeline 结算约束
    ...
```

### 3.3 `omni_drones/envs/net_capture/net_capture.py` — _reset_idx 顺序优化

base_link 和 rotors 的位姿/速度重置紧邻执行，避免 FixedJoint 约束不一致窗口：

```
1. drone._reset_idx(env_ids)
2. drone.set_world_poses(base_link)     ← 紧邻
3. rotors_view.set_world_poses(rotors)   ← 紧邻
4. drone.set_velocities(...)            ← 紧邻
5. rotors_view.set_velocities(...)      ← 紧邻
6. net nodes/edges/rope 重置
```

### 3.4 `cfg/base/sim_base.yaml` — 关闭 flatcache

```yaml
use_flatcache: false  # 原为 true，会阻止 RigidPrimView GPU 视图创建
```

### 3.5 其他修复

- `~/.local/share/ov/pkg/isaac-sim-4.1.0/exts/omni.isaac.ml_archive/pip_prebundle/torch/_C/` → `_C.bak`（修复 PyTorch `_C` 导入错误）
- `omni.pip.cloud/pip_prebundle/typing_extensions.py` 更新为 v4.15.0（修复 pydantic Sentinel 缺失）

---

## 4. 仍有残留的问题

### 4.1 "Physics Simulation View is not created yet" 警告

**来源**: Isaac Sim 基类 `_RigidPrimView.post_reset()` 在 init 阶段调用 `set_velocities()` 时，`_physics_view` 可能尚未创建。这是初始化时序问题，不影响训练（训练中 `_physics_view` 有效时走 GPU 路径）。

**当前处理**: 代码已改为 GPU/CPU 双路径，CPU 回退用 `disable_warnings` 包裹，抑制了 PhysX 噪音。但 CARB 日志级别的警告（`carb.log_warn`）不会被 `disable_warnings` 抑制。

**未来改进方向**: 
- 在 `RigidPrimView.initialize()` 中传入共享的 `physics_sim_view`
- 或在 `post_reset` 前确保 `_physics_view` 已创建

### 4.2 旋翼不转

`is_articulation=False` + `_strip_articulation` 后，`multirotor.py:257-262` 的 `set_joint_velocities` 被跳过。旋翼推力通过 `rotors_view.apply_forces_and_torques_at_pos` 施加，物理正确但视觉旋翼不转。纯视觉效果，不影响飞行物理。

---

## 5. 未来开发建议

### 5.1 简化任务路径

建议从 "hover at target" 开始（类似 Hover 环境但带绳网），逐步增加难度：

1. **NetHover**: 多机绳网系统悬停在固定目标点
2. **NetTrack**: 跟踪移动目标
3. **NetCapture**: 最终捕捉任务

### 5.2 is_articulation 架构选择

| 方案 | 优点 | 缺点 |
|------|------|------|
| `is_articulation=False` (当前) | 全部用 RigidPrimView，API 统一 | rotors 不转 (视觉), `_strip_articulation` 是 hack |
| `is_articulation=True` (尝试过) | 标准架构, rotors 自动转 | training 中 ArticulationView.get_root_transforms 崩溃 |

当前 `is_articulation=False` 是唯一能跑通训练的路径。

### 5.3 GPU Dynamics 兼容性

D6 joints (rope) + GPU dynamics 是已知的 PhysX 限制区域。如果后续遇到更多 GPU API 问题，考虑：
- 用胡克定律手动计算绳缆力（在 `apply_action` 中），完全绕过 PhysX 约束求解
- 或降级到 CPU physics（`sim.use_gpu=false`）

---

## 6. 关键文件索引

| 文件 | 作用 |
|------|------|
| `omni_drones/views/__init__.py` | RigidPrimView GPU/CPU 双路径 override |
| `omni_drones/envs/net_capture/net_capture.py` | NetCapture 环境, _reset_idx 顺序优化 |
| `omni_drones/envs/net_capture/utils.py` | NetCaptureGroup, _strip_articulation, 视图初始化 |
| `omni_drones/envs/isaac_env.py` | IsaacEnv 基类, _reset 中 sim.step() |
| `omni_drones/utils/scene.py` | create_net, create_rope, _make_compliant_cross_joint |
| `cfg/base/sim_base.yaml` | sim 参数 (use_flatcache 已改为 false) |
| `scripts/debug_netcapture.py` | 增量诊断脚本 |

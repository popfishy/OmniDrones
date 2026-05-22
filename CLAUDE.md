# OmniDrones 项目知识总结

## 项目概述

OmniDrones 是基于 Nvidia Isaac Sim 4.1.0 的多旋翼无人机强化学习平台。使用 Hydra 管理配置，TorchRL 的 EnvBase 作为环境基类，PPO 为主要训练算法。

## 核心架构

### IsaacEnv 基类 (`omni_drones/envs/isaac_env.py`)

所有任务环境继承 `IsaacEnv`，需实现 5 个抽象方法：

| 方法 | 作用 |
|------|------|
| `_design_scene()` | 创建模板场景（无人机、物体等），返回全局碰撞 prim 路径列表 |
| `_set_specs()` | 定义 observation/action/reward 的 TensorSpec |
| `_reset_idx(env_ids)` | 重置指定环境的状态 |
| `_compute_state_and_obs()` | 物理步进后计算观测 |
| `_compute_reward_and_done()` | 计算奖励 + terminated/truncated 标志 |

执行流程：`_design_scene()` → GridCloner 复制 N 个环境 → `_set_specs()` → 训练循环 (`_reset_idx` → `_pre_sim_step` → `sim.step()` → `_compute_state_and_obs` → `_compute_reward_and_done`)

任务通过 `IsaacEnv.REGISTRY` 自动注册（`__init_subclass__` 机制），只需在 `omni_drones/envs/__init__.py` 中 import 即可。

### 训练命令示例

```bash
python train.py algo=ppo headless=true task=Hover drone_model.name=DifferentialUAV wandb.mode=offline
python train.py algo=ppo headless=true task=Transport/TransportTrack total_frames=300_000 wandb.mode=offline
```

## 无人机模型体系

### 注册机制

`RobotBase.__init_subclass__` 自动将所有子类注册到 `RobotBase.REGISTRY`（类名和类名小写双 key）。只需在 `robots/drone/__init__.py` 中 import 新类即可生效。

### 创建自定义无人机（最简方式）

只需定义一个 8 行子类 + 一个 YAML 参数文件：

**Python 类** (`omni_drones/robots/drone/xxx.py`)：
```python
from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH

class XxxDrone(MultirotorBase):
    usd_path: str = ASSET_PATH + "/usd/xxx.usd"
    param_path: str = ASSET_PATH + "/usd/xxx.yaml"
```

所有物理参数（质量、惯量、桨叶布局）从 YAML 读取，无需在 Python 中硬编码。

**YAML 参数文件** (`omni_drones/robots/assets/usd/xxx.yaml`)：
```yaml
name: xxx
inertia: {xx: ..., yy: ..., zz: ...}
mass: 1.8
l: 0.157
drag_coef: 0.2
rotor_configuration:
  num_rotors: 4
  rotor_angles: [-0.88, -2.26, 2.26, 0.88]   # atan2(y, x) 弧度
  arm_lengths: [0.157, ...]
  directions: [1.0, -1.0, 1.0, -1.0]          # 对角同号
  force_constants: [2.0e-05, ...]
  moment_constants: [2.0e-07, ...]
  max_rotation_velocities: [838, ...]
```

### USD 模型的命名要求（关键）

`MultirotorBase.initialize()` 中有硬编码的 prim 名称约定：

- 机体链接：必须是 `base_link`
- 桨叶链接：必须是 `rotor_0`, `rotor_1`, `rotor_2`, `rotor_3`（匹配模式 `rotor_*`）
- 桨叶关节：必须以 `rotor` 开头（如 `rotor_0_joint`），用于 `startswith("rotor")` 匹配

如从 SolidWorks URDF 导出，默认名为 `7045_3_R1_link` 等，必须在 Isaac Sim 中手动重命名。

### 控制器配置 (`omni_drones/controllers/cfg/`)

```yaml
position_gain: [6, 6, 6]
velocity_gain: [4.7, 4.7, 4.7]
attitude_gain: [1.0, 1.0, 0.05]
angular_rate_gain: [0.18, 0.18, 0.06]
```

控制器通过 `ControllerBase.REGISTRY` 同样自动注册。

## 项目配置结构

- `cfg/base/env_base.yaml` — 环境基础参数（num_envs, env_spacing, max_episode_length）
- `cfg/base/sim_base.yaml` — 仿真参数（dt, substeps, PhysX GPU buffer 等）
- `cfg/task/` — 任务配置（继承 base 配置，指定 drone_model, 奖励权重等）
- `cfg/robot/` — 机器人选择配置（name 指向 REGISTRY key）
- `cfg/algo/` — 算法配置（PPO 参数）

配置通过 Hydra 的 `defaults` 机制组合。

## 本项目现有无人机

**DifferentialUAV** (`omni_drones/robots/drone/differential_uav.py`)：4 旋翼，质量 1.8kg，臂长 0.157m，桨叶角 [-0.88, -2.26, 2.26, 0.88]。USD 和参数文件在 `robots/assets/usd/differential_uav.{usd,yaml}`，控制器配置在 `controllers/cfg/lee_controller_differential_uav.yaml`。wurenji_3 与此模型几何一致。

## 自定义无人机训练注意事项

### 推力灵敏度差异导致梯度爆炸

PayLoadTrack / TransportTrack 等任务中，RL 策略**直接控制旋翼转速**（不经过控制器）。不同无人机的 KF（最大推力/旋翼）差异导致策略对动作的灵敏度不同：同一动作值 `cmd` 映射为 `clamp((cmd+1)/2, 0, 1) × KF`，即 **∂thrust/∂cmd = KF/2**。

| 指标 | Hummingbird | DifferentialUAV |
|------|-------------|-----------------|
| KF (单旋翼最大推力) | 6.0 N | 14.05 N |
| ∂thrust/∂cmd | 3.0 | **7.03** (2.34x) |

DifferentialUAV 的推力对动作敏感 2.34 倍，首次 PPO 更新后策略容易过度补偿 → 极端姿态/速度 → 观测值放大 → 梯度爆炸 → NaN。

**解决：** 降低学习率 + 收紧梯度裁剪 + 减少 PPO 更新轮次。

注意 Hydra 中 `actor`/`critic` 只在 YAML 中定义（不在 `PPOConfig` dataclass），命令行覆盖需用 `+` 前缀：

```bash
python train.py algo=ppo headless=true task=Payload/PayloadTrack \
  task.drone_model.name=DifferentialUAV total_frames=200_000_000 \
  wandb.mode=offline \
  +algo.actor.lr=0.0001 +algo.critic.lr=0.0001 \
  +algo.max_grad_norm=1.0 algo.ppo_epochs=2
```

### 训练中 OOM（Out of Memory）

Eval 阶段 `RenderCallback` 将完整 episode 帧存在内存中（约 600MB），wandb 编码 MP4 额外分配内存，叠加 Isaac Sim 的 GPU 占用容易触发 OOM Killer。解决：添加 `record_video=false` 禁用 eval 视频录制（`scripts/train.py` 已支持）。

```bash
python train.py ... record_video=false
```

## 常见问题

### PhysX GPU Direct API 报错

```
PxRigidDynamic::setAngularVelocity(): it is illegal to call this method if PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled!
```

解决：确保 `replicate_physics: false`（在 sim_base.yaml 中），或无头模式下设置：
```bash
export OMNI_KIT_ALLOW_TELEMETRY=0
unset DISPLAY
```

## 关键文件索引

| 用途 | 路径 |
|------|------|
| 环境基类 | `omni_drones/envs/isaac_env.py` |
| 无人机基类 | `omni_drones/robots/drone/multirotor.py` |
| 机器人基类（注册/spawn/initialize） | `omni_drones/robots/robot.py` |
| 无人机注册入口 | `omni_drones/robots/drone/__init__.py` |
| 任务注册入口 | `omni_drones/envs/__init__.py` |
| DifferentialUAV 类 | `omni_drones/robots/drone/differential_uav.py` |
| 单机 Hover 任务 | `omni_drones/envs/single/hover.py` |
| 协同搬运任务 | `omni_drones/envs/transport/transport_track.py` |
| Platform 协同任务 | `omni_drones/envs/platform/platform_hover.py` |
| 训练入口 | `scripts/train.py` |
| Asset 路径常量 | `omni_drones/robots/robot.py` 中的 `ASSET_PATH` |
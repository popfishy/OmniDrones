![Visualization of OmniDrones](docs/source/_static/visualization.jpg)

---

> **Note:** I greatly appreciate the interest by the community in this project. However, due to several difficulties, this version of the project is hard to maintain and update anymore. I sincerely apologize for the inconvenience. There may or may not be a cleaner refactored version in the future. If you believe it is highly helpful to your research, you are welcomed to contact me by emailing to btx0424@outlook.com.

# OmniDrones

[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-4.1.0-orange.svg)](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://omnidrones.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord Forum](https://dcbadge.vercel.app/api/server/J4QvXR6tQj)](https://discord.gg/J4QvXR6tQj)

*OmniDrones* is an open-source platform designed for reinforcement learning research on multi-rotor drone systems. Built on [Nvidia Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html), it features highly efficient and flexible simulation for various research purposes, with a suite of benchmark tasks and algorithm baselines.

For usage and more details, please refer to the [documentation](https://omnidrones.readthedocs.io/en/latest/). Windows is not supported. Welcome to join our [Discord](https://discord.gg/J4QvXR6tQj) for discussions and questions.

## 快速开始

```bash
# 单机悬停
python train.py algo=ppo headless=true task=Hover total_frames=200_000_000 wandb.mode=offline

# 单机轨迹跟踪
python train.py algo=ppo headless=true task=Track total_frames=200_000_000 wandb.mode=offline

# 多机前往目标点
python train.py algo=mappo headless=true task=Transport/TransportHover \
  task.drone_model.name=DifferentialUAV \
  total_frames=40_000_000 wandb.mode=offline \
  algo.entropy_coef=0.01 \
  task.action_scale=0.5


# 多机协同搬运 + DifferentialUAV
python train.py algo=mappo headless=true task=Transport/TransportTrack \
  total_frames=40_000_000 wandb.mode=offline \
  algo.entropy_coef=0.01


python train.py algo=mappo headless=true task=Transport/TransportTrack \
  task.drone_model.name=DifferentialUAV \
  total_frames=40_000_000 wandb.mode=offline \
  algo.entropy_coef=0.01

# 多无人机绳网捕捉
python train.py algo=mappo headless=true task=NetCapture/NetCapture \
  total_frames=100 wandb.mode=offline



```

更多训练命令见 [tasks.md](docs/source/demo/tasks.md)。

## 项目架构

### IsaacEnv 基类

所有任务环境继承 `IsaacEnv`（`omni_drones/envs/isaac_env.py`），需实现 5 个方法：

| 方法 | 作用 |
|------|------|
| `_design_scene()` | 创建模板场景 |
| `_set_specs()` | 定义 observation/action/reward 的 TensorSpec |
| `_reset_idx(env_ids)` | 重置指定环境 |
| `_compute_state_and_obs()` | 物理步进后计算观测 |
| `_compute_reward_and_done()` | 计算奖励 + terminated/truncated |

执行流程：`_design_scene()` → GridCloner 复制 N 环境 → `_set_specs()` → 训练循环。任务通过 `IsaacEnv.REGISTRY` 自动注册（`__init_subclass__`），在 `envs/__init__.py` import 即可。

### 配置结构

```
cfg/
├── base/env_base.yaml      # num_envs, max_episode_length
├── base/sim_base.yaml       # dt, substeps, PhysX GPU buffer
├── task/<Task>/             # 任务配置 → defaults: /base/env_base
├── algo/mappo.yaml          # 算法参数
└── robot/                   # 机器人选择
```

Hydra `defaults` 组合配置，命令行可覆盖任意键。

### 无人机模型体系

`RobotBase.__init_subclass__` 自动注册。创建新无人机只需 8 行子类 + YAML 参数文件。

**USD 命名要求**（硬编码在 `MultirotorBase.initialize()`）：
- 机体链接：`base_link`
- 桨叶链接/关节：`rotor_0`, `rotor_1`, ...（匹配 `rotor_*`）

**DifferentialUAV**：4 旋翼，1.8kg，臂长 0.157m。参数在 `robots/assets/usd/differential_uav.{usd,yaml}`。

### 推力计算链

```
策略输出 cmd ∈ [-1,1]
  → throttle = sqrt(clamp((cmd+1)/2, 0, 1))
  → thrust = throttle² × KF
  → KF = max_rot_vel² × force_constants
```

| 指标 | Hummingbird | DifferentialUAV |
|------|-------------|-----------------|
| KF (单旋翼) | 6.0 N | 14.05 N |
| ∂thrust/∂cmd | 3.0 | **7.03** (2.34×) |

## 训练指南

### 算法选择

| 算法 key | 类 | Critic | 适用 |
|----------|-----|--------|------|
| `mappo` | `MAPPO` (`ppo/mappo.py`) | **集中式** (`observation_central`) | 多机共享奖励 |
| `ppo` | `PPOPolicy` (`ppo/ppo.py`) | 去中心化 (局部观测) | 单机任务 |
| `happo` | `HAPPOPolicy` | 可选集中式 | 异构多智能体 |

**多机任务（Transport/Platform 等）必须用 `algo=mappo`**。`algo=ppo` 的去中心化 Critic 在共享奖励下信用分配失效。

### Hydra 命令行覆盖

```bash
# YAML 中已存在的键直接覆盖
algo.entropy_coef=0.01

# Dataclass 中不存在的键需 + 前缀
+algo.actor.lr=0.0001

# mappo.yaml 已定义 actor/critic 段，不需要 + 前缀
algo.actor.lr=0.0001
```

### DifferentialUAV 训练：action_scale 方案

DifferentialUAV 推力灵敏度是 Hummingbird 的 2.34 倍，直接训练易导致梯度爆炸。在环境层加 `action_scale` 压缩动作范围，不修改物理参数，不影响 sim2real。

`action_scale` 默认 1.0，已在 `transport_hover.py`、`transport_track.py`、`transport_fly_through.py` 中支持。

### 多智能体训练注意事项

- `entropy_coef=0.01` 必需（默认 0.001 对多智能体太低，策略熵崩塌）
- `num_envs≥32`
- `record_video=false` 可减少 GPU 显存，降低 OOM 风险

## 关键文件索引

| 用途 | 路径 |
|------|------|
| 环境基类 | `omni_drones/envs/isaac_env.py` |
| 无人机基类 | `omni_drones/robots/drone/multirotor.py` |
| 推力计算 | `omni_drones/actuators/rotor_group.py` |
| MAPPO (集中式 Critic) | `omni_drones/learning/ppo/mappo.py` |
| PPO (去中心化 Critic) | `omni_drones/learning/ppo/ppo.py` |
| 训练入口 | `scripts/train.py` |
| 场景工具（rope/net/bar） | `omni_drones/utils/scene.py` |
| TransportTrack | `omni_drones/envs/transport/transport_track.py` |
| TransportHover | `omni_drones/envs/transport/transport_hover.py` |
| DifferentialUAV | `omni_drones/robots/drone/differential_uav.py` |

## 已知问题 & 故障排查

### GPU 掉卡

长时间训练后 Isaac Sim 可能丢失 CUDA 设备。

```bash
sudo nvidia-smi -r    # 重置 GPU（推荐）
# 或彻底重启驱动
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia_uvm
```

预防：`record_video=false`、`save_interval=1000` 定期保存 checkpoint。

### 训练后残留进程

上一次训练结束但进程未清理干净时，再次训练可能失败。

```bash
pkill -9 python
pkill -9 kit
```

### S3 地面资源无法加载

Isaac Sim 启动时从 `http://omniverse-content-production.s3-us-west-2.amazonaws.com/` 加载地面纹理。若 HTTP 被防火墙阻断，`design_scene()` 会崩溃。可改为本地 ground plane fallback。

### PhysX GPU Direct API 报错

```
PxRigidDynamic::setAngularVelocity(): it is illegal to call this method
if PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled!
```

确保 `replicate_physics: false`，无头模式下：

```bash
export OMNI_KIT_ALLOW_TELEMETRY=0
export DISABLE_TELEMETRY=1
unset DISPLAY
```

## Citation

Please cite [this paper](https://arxiv.org/abs/2309.12825) if you use *OmniDrones* in your work:

```bibtex
@misc{xu2023omnidrones,
    title={OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control},
    author={Botian Xu and Feng Gao and Chao Yu and Ruize Zhang and Yi Wu and Yu Wang},
    year={2023},
    eprint={2309.12825},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

## 衍生项目

1. [VolleyBots](https://github.com/thu-uav/VolleyBots) — 无人机排球平台 [[website](https://volleybots.github.io/)]
2. [HCSP](https://github.com/thu-uav/HCSP) — 无人机排球分层协同自博弈算法 [[website](https://hi-co-self-play.github.io/)]
3. [Multi-UAV Formation](https://github.com/thu-uav/multi-UAV-formation) — 多无人机避障
4. [JuggleRL](https://github.com/thu-uav/JuggleRL_train) — 无人机颠球
5. [SimpleFlight](https://github.com/thu-uav/SimpleFlight) — 无人机零样本 sim2real
6. [Multi-UAV Pursuit-Evasion](https://github.com/thu-uav/Multi-UAV-pursuit-evasion) — 多无人机追逃
7. [NavRL](https://github.com/Zhefan-Xu/NavRL) — 无人机复杂环境安全导航

## 版本说明

The initial release of **OmniDrones** was developed on Isaac Sim 2022.2.0 ([release branch](https://github.com/btx0424/OmniDrones/tree/release)). The current version is developed on Isaac Sim 4.1.0.

The project has gone through significant API changes (Isaac Sim 2023.1.0 brought new sensor support). The `release` branch is maintained for backward compatibility.

## Acknowledgement

Heavily inspired by [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

# 算法注册表与使用路径

> 自动生成于 2026-06-17 | 基于 `omni_drones/learning/__init__.py` 中 `ALGOS` 字典

## 入口脚本总览

| 入口脚本 | 配置路径 | 默认算法 | 算法加载方式 |
|----------|---------|---------|-------------|
| `scripts/train.py` | `scripts/train.yaml` | `ppo` | `ALGOS` 字典动态查找 |
| `scripts/play.py` | `scripts/train.yaml` | `ppo` | `ALGOS` 字典动态查找 |
| `scripts_paper/train.py` | `cfg/train.yaml` | `mappo` | **硬编码** `algos` 字典（不共享 ALGOS） |
| `examples/demo_task.py` | `cfg/train.yaml` (via artifact) | 取决于 artifact | `ALGOS` 字典动态查找 |
| `scripts/train_lidar.py` | N/A | N/A | **自定义本地 PPOPolicy**（独立于 ALGOS） |

---

## 算法详细清单

### 1. `mappo` — MAPPO (ppo/mappo.py)

| 维度 | 详情 |
|------|------|
| 配置文件 | `cfg/algo/mappo.yaml` |
| `scripts/train.py` | ✅ 通过 ALGOS |
| `scripts/play.py` | ✅ 通过 ALGOS |
| `scripts_paper/train.py` | ✅ **硬编码字典第 95 行** |
| `examples/demo_task.py` | ✅ 通过 ALGOS |
| 实验配置 | `experiments/ma_onpolicy.yaml`, `onpolicy.yaml`, `platform.yaml`, `scalability.yaml`, `drone_model.yaml` |
| Wandb Sweep | `wandb_sweep/platform_mappo.yaml`, `transport_mappo.yaml`, `spread_mappo.yaml` |
| 说明 | **项目中使用最广泛的算法**；也是 `cfg/train.yaml` 的默认值 |

### 2. `mappo_old` — MAPPOPolicy (mappo.py)

| 维度 | 详情 |
|------|------|
| 配置文件 | `cfg/algo/mappo_old.yaml` |
| `scripts/train.py` | ✅ 通过 ALGOS |
| `scripts/play.py` | ✅ 通过 ALGOS |
| `scripts_paper/train.py` | ❌ **不在**硬编码字典中 |
| 实验配置 | 无 |
| Wandb Sweep | 无 |
| 说明 | 历史遗留版本，无实验或 sweep 使用 |

### 3. `happo` — HAPPOPolicy (happo.py)

| 维度 | 详情 |
|------|------|
| 配置文件 | `cfg/algo/happo.yaml` |
| `scripts/train.py` | ✅ 通过 ALGOS |
| `scripts/play.py` | ✅ 通过 ALGOS |
| `scripts_paper/train.py` | ✅ **硬编码字典第 96 行** |
| 实验配置 | `experiments/ma_onpolicy.yaml`（grid: `[mappo, happo]`） |
| Wandb Sweep | 无 |
| 说明 | 实验配置 `platform.yaml` 中有注释掉的 `happo` 项 |

### 4. `ppo` — PPOPolicy (ppo/ppo.py)

| 维度 | 详情 |
|------|------|
| 配置文件 | ❌ **无专用配置文件** |
| `scripts/train.py` | ✅ 通过 ALGOS（**默认算法**） |
| `scripts/play.py` | ✅ 通过 ALGOS（**默认算法**） |
| `scripts_paper/train.py` | ❌ **不在**硬编码字典中 |
| 实验配置 | `scripts/experiments/baselines.yaml`（grid: `[ppo, ppo_priv, ppo_priv_critic]`） |
| Wandb Sweep | 无 |
| 说明 | 无 `cfg/algo/ppo.yaml`；`train_lidar.py` 中有**自定义 PPO 实现** |

### 5. `ppo_rnn` — PPORNNPolicy (ppo/ppo_rnn.py)

| 维度 | 详情 |
|------|------|
| 配置文件 | ❌ **无专用配置文件** |
| `scripts/train.py` | ✅ 可通过 ALGOS 访问（无默认配置） |
| `scripts_paper/train.py` | ❌ **不在**硬编码字典中 |
| 实验配置 | 无 |
| Wandb Sweep | 无 |
| 说明 | **从未在任何实际运行中使用** |

### 6. `ppo_adapt` — PPOAdaptivePolicy (ppo/ppo_adapt.py)

| 维度 | 详情 |
|------|------|
| 配置文件 | ❌ **无专用配置文件** |
| `scripts/train.py` | ✅ 可通过 ALGOS 访问（无默认配置） |
| `scripts_paper/train.py` | ❌ **不在**硬编码字典中 |
| 实验配置 | 无 |
| Wandb Sweep | 无 |
| 说明 | **从未在任何实际运行中使用** |

### 7. `sac` — SACPolicy (sac.py)

| 维度 | 详情 |
|------|------|
| 配置文件 | `cfg/algo/sac.yaml` |
| `scripts/train.py` | ✅ 通过 ALGOS |
| `scripts/play.py` | ✅ 通过 ALGOS |
| `scripts_paper/train.py` | ✅ **硬编码字典第 99 行** |
| 实验配置 | `experiments/offpolicy.yaml`（grid: `[sac, td3]`） |
| Wandb Sweep | `wandb_sweep/hover_sac.yaml` |
| 说明 | **数值稳定性已修复**（来自 VolleyBots 的移植） |

### 8. `td3` — TD3Policy (td3.py)

| 维度 | 详情 |
|------|------|
| 配置文件 | `cfg/algo/td3.yaml` |
| `scripts/train.py` | ✅ 通过 ALGOS |
| `scripts/play.py` | ✅ 通过 ALGOS |
| `scripts_paper/train.py` | ✅ **硬编码字典第 100 行** |
| 实验配置 | `experiments/offpolicy.yaml`（grid: `[sac, td3]`） |
| Wandb Sweep | 无 |
| 说明 | **已修复 Critic 多智能体维度**（`num_agents` 从硬编码 1 改为 `agent_spec.n`） |

### 9. `maddpg` — MADDPGPolicy (maddpg.py) 🆕

| 维度 | 详情 |
|------|------|
| 配置文件 | `cfg/algo/maddpg.yaml` |
| `scripts/train.py` | ✅ 通过 ALGOS（需手动 `algo=maddpg`） |
| `scripts/play.py` | ✅ 通过 ALGOS |
| `scripts_paper/train.py` | ❌ **不在**硬编码字典中 |
| 实验配置 | 无 |
| Wandb Sweep | 无 |
| 说明 | 来自 VolleyBots 的新移植算法；**尚无实际实验调用** |

### 10. `mappo_mask` — MAPPOPolicyMask (mappo_mask.py) 🆕

| 维度 | 详情 |
|------|------|
| 配置文件 | `cfg/algo/mappo_mask.yaml`（继承 `mappo`） |
| `scripts/train.py` | ✅ 通过 ALGOS（需手动 `algo=mappo_mask`） |
| `scripts/play.py` | ✅ 通过 ALGOS |
| `scripts_paper/train.py` | ❌ **不在**硬编码字典中 |
| 实验配置 | 无 |
| Wandb Sweep | 无 |
| 说明 | 来自 HCSP 的新移植算法；在 `update_actor`/`update_critic` 中支持 action masking；需 `cfg.mask_name` |

### 11. `mat` — MATPolicy (mat.py) 🆕

| 维度 | 详情 |
|------|------|
| 配置文件 | `cfg/algo/mat.yaml` |
| `scripts/train.py` | ✅ 通过 ALGOS（需手动 `algo=mat`） |
| `scripts/play.py` | ✅ 通过 ALGOS |
| `scripts_paper/train.py` | ❌ **不在**硬编码字典中 |
| 实验配置 | 无 |
| Wandb Sweep | 无 |
| 说明 | 来自 VolleyBots 的新移植算法；Transformer encoder-decoder 架构；支持连续/离散动作 |

### 12. `psro` — PSROPolicy (psro/psro.py) 🆕

| 维度 | 详情 |
|------|------|
| 配置文件 | `cfg/algo/psro.yaml`（继承 `mappo`） |
| `scripts/train.py` | ✅ 通过 ALGOS（需手动 `algo=psro`） |
| `scripts/play.py` | ✅ 通过 ALGOS |
| `scripts_paper/train.py` | ❌ **不在**硬编码字典中 |
| 实验配置 | 无 |
| Wandb Sweep | 无 |
| 说明 | 来自 VolleyBots 的新移植算法；双玩家群体训练 + meta-game 分析；要求 `agent_spec.n % 2 == 0` |

---

## 不在 ALGOS 中但在 `learning/` 内的算法

以下算法仅在 `scripts_paper/train.py` 的硬编码字典中使用，**不在** `learning/__init__.py` 的 `ALGOS` 字典中：

| 类 | 配置文件 | `scripts_paper/train.py` | 实验 |
|----|---------|--------------------------|------|
| `QMIXPolicy` | `cfg/algo/qmix.yaml` | 第 97 行 | `experiments/qmix.yaml` |
| `DQNPolicy` | `cfg/algo/dqn.yaml` | 第 98 行 | `experiments/dqn.yaml` |
| `MATD3Policy` | `cfg/algo/matd3.yaml` | 第 101 行 | `experiments/ma_offpolicy.yaml` |
| `TDMPCPolicy` | `cfg/algo/tdmpc.yaml` | 第 102 行 | `experiments/model_based.yaml` |

---

## 关键发现

1. **两个独立的算法加载路径**：`scripts/train.py` 使用 `ALGOS` 字典，而 `scripts_paper/train.py` 使用**硬编码的本地字典**。两者未同步。

2. **新移植的 4 个算法**（`maddpg`, `mappo_mask`, `mat`, `psro`）仅在 `ALGOS` 中注册，`scripts_paper/train.py` **不包含它们**。

3. **`ppo_rnn` 和 `ppo_adapt`** 缺少 `cfg/algo/` 配置文件，**从未被任何运行使用**。

4. **`scripts/train.py` 的构造函数签名不匹配**：传入 4 个位置参数，而 ALGOS 中大多数策略的 `__init__` 只接受 3 个参数（`cfg`, `agent_spec`, `device`）。这是**预存缺陷**——`scripts_paper/train.py` 中正确的 `AgentSpec` 构造方式证实了这一点。

5. **`mappo` 是使用最广泛的算法**，在 5 个实验配置 + 3 个 wandb sweep 中被引用。

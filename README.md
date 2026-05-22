![Visualization of OmniDrones](docs/source/_static/visualization.jpg)

---

# Future of this Project

I greatly appreciate the interest by the community in this project. However, due to several difficulties, this version of the project is hard to maintain and update anymore. I sincerely apologize for the inconvenience. There may or may not be a clearner refactored version in the future. If you believe it is highly helpful to your research, you are welcomed to contact me by emailing to btx0424@outlook.com.

# OmniDrones

[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-4.1.0-orange.svg)](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://omnidrones.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord Forum](https://dcbadge.vercel.app/api/server/J4QvXR6tQj)](https://discord.gg/J4QvXR6tQj)

*OmniDrones* is an open-source platform designed for reinforcement learning research on multi-rotor drone systems. Built on [Nvidia Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html), *OmniDrones* features highly efficient and flexible simulation that can be adopted for various research purposes. We also provide a suite of benchmark tasks and algorithm baselines to provide preliminary results for subsequent works.

For usage and more details, please refer to the [documentation](https://omnidrones.readthedocs.io/en/latest/). Unfortunately, it does not support Windows.

Welcome to join our [Discord](https://discord.gg/J4QvXR6tQj) for discussions and questions!

## Notice

The initial release of **OmniDrones** is developed based on Isaac Sim 2022.2.0. It can be found at the [release](https://github.com/btx0424/OmniDrones/tree/release) branch. The current version is developed based on Isaac Sim 4.1.0.

## Announcement 2023-09-25

The initial release of **OmniDrones** is developed based on Isaac Sim 2022.2.0. As the next version of
Isaac Sim (2023.1.0) is expected to bring substantial changes but is not yet available, the APIs and usage
of **OmniDrones** are subject to change. We will try our best to keep the documentation up-to-date.

## Announcement 2023-10-25

The new release of Isaac Sim (2023.1.0) has brought substantial changes as well as new possibilities, among
which the most important is new sensors. We are actively working on it at the `devel` branch. The `release`
branch will still be maintained for compatibility. Feel free to raise issues if you encounter any problems
or have ideas to discuss.

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

## Acknowledgement

Some of the abstractions and implementation was heavily inspired by [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

## Train

详细训练数据见 [tasks.md](docs/source/demo/tasks.md)

```
python train.py algo=ppo headless=true task=Transport/TransportTrack total_frames=300_000 save_interval=2000 wandb.mode=offline

python train.py algo=ppo headless=true task=Transport/TransportTrack wandb.mode=offline

python train.py algo=ppo headless=true task=Track total_frames=200_000_000 wandb.mode=offline

python train.py algo=ppo headless=true task=Payload/PayloadTrack total_frames=200_000_000 wandb.mode=offline

python train.py algo=ppo headless=true task=Pinball total_frames=200_000_000 wandb.mode=offline

python train.py algo=ppo headless=true task=InvPendulum/InvPendulumHover total_frames=200_000_000 wandb.mode=offline

python train.py algo=ppo headless=true task=Hover task.drone_model.name=DifferentialUAV total_frames=200_000_000 wandb.mode=offline

python train.py algo=ppo headless=true task=Payload/PayloadTrack \
  task.drone_model.name=DifferentialUAV total_frames=200_000_000 \
  wandb.mode=offline \
  +algo.actor.lr=0.0001 +algo.critic.lr=0.0001 \
  +algo.max_grad_norm=1.0 algo.ppo_epochs=2

```

2026-03-19 02:21:25 [14,269ms] [Error] [omni.physx.plugin] PhysX error: PxRigidDynamic::setAngularVelocity(): it is illegal to call this method if PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled!, FILE /builds/omniverse/physics/physx/source/physx/src/NpRigidDynamic.cpp, LINE 267
2026-03-19 02:21:25 [14,269ms] [Error] [omni.physx.plugin] PhysX error: PxRigidDynamic::setGlobalPose(): it is illegal to call this method if PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled!, FILE /builds/omniverse/physics/physx/source/physx/src/NpRigidDynamic.cpp, LINE 102
2026-03-19 02:21:25 [14,269ms] [Error] [omni.physx.plugin] PhysX error: PxRigidDynamic::setLinearVelocity(): it is illegal to call this method if PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled!, FILE /builds/omniverse/physics/physx/source/physx/src/NpRigidDynamic.cpp, LINE 239
2026-03-19 02:21:25 [14,269ms] [Error] [omni.physx.plugin] PhysX error: PxRigidDynamic::setAngularVelocity(): it is illegal to call this method if PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled!, FILE





## DEBUG

#### 报错一

图形化界面问题

#### 第一步：彻底禁用 NVIDIA 遥测插件

```
export OMNI_KIT_ALLOW_TELEMETRY=0
export DISABLE_TELEMETRY=1
```

#### 第二步：掐断虚假的显示器信号

（在无头模式下，告诉显卡直接在后台显存里算，别去碰操作系统的窗口界面）

```
unset DISPLAY
```

#### 第三步：再次运行训练代码

```
python train.py algo=ppo headless=true
```



#### 报错二

训练一次之后，再次训练其他代码，可能遇到：

```
pkill -9 python
pkill -9 kit
```

# 衍生项目

1.  无人机排球平台：VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play   https://github.com/thu-uav/VolleyBots    https://volleybots.github.io/
2.  无人机排球算法：Mastering Multi-Drone Volleyball through Hierarchical Co-Self-Play Reinforcement Learning  https://github.com/thu-uav/HCSP   https://hi-co-self-play.github.io/
3. 多无人机避障：Multi-UAV Formation Control with Static and Dynamic Obstacle Avoidance via Reinforcement Learning  https://github.com/thu-uav/multi-UAV-formation
4.  无人机颠球算法：JuggleRL: Mastering Ball Juggling with a Quadrotor via Deep Reinforcement Learning   https://github.com/thu-uav/JuggleRL_train
5. 无人机零样本sim2real: What Matters in Learning A Zero-Shot Sim-to-Real RL Policy for Quadrotor Control? A Comprehensive Study  https://github.com/thu-uav/SimpleFlight （时间比较远）
6. 多无人机轨迹跟踪：Multi-UAV Pursuit-Evasion with Online Planning in Unknown Environments by Deep Reinforcement Learning  https://github.com/thu-uav/Multi-UAV-pursuit-evasion   （时间比较远）
7. 无人机复杂环境安全导航：NavRL: Learning Safe Flight in Dynamic Environments   https://github.com/Zhefan-Xu/NavRL 




























import ctypes
ctypes.CDLL("libX11.so.6").XInitThreads()

import logging
import os
import time

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
# import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    # 解析Hydra配置（支持命令行覆盖，如algo=ppo）
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    # 初始化Isaac Sim仿真进程（必须先启动，否则环境无法加载）
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    # 环境变换（TorchRL封装，支持观测展平、动作离散化）
    transforms = [InitTracker()]

    # 展平观测（CompositeSpec → 一维Tensor，适配MLP编码器）
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # optionally discretize the action space or use a controller
    # 动作离散化（如multidiscrete:5 → 动作空间分为5个离散区间）
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        # 从算法注册表加载策略（如ppo→PPOPolicy，mappo→MAPPOPolicy）
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,  # 观测空间规格
            env.action_spec,       # 动作空间规格
            env.reward_spec,       # 奖励空间规格
            device=base_env.device  # 设备（默认cuda）
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    # 批次配置（环境数 × 训练间隔 = 每个批次的帧数）
    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1) # 评估间隔（每N批次评估一次）
    save_interval = cfg.get("save_interval", -1) # 模型保存间隔

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    # 数据收集器（TorchRL同步收集，按批次获取环境交互数据）
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE
    ):
        record_video = cfg.get("record_video", True)

        base_env.enable_render(record_video)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2) if record_video else None

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        if record_video:
            info["recording"] = wandb.Video(
                render_callback.get_video_array(axes="t c h w"),
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                format="mp4"
            )

        return info

    # 训练循环（进度条可视化）
    pbar = tqdm(collector, total=total_frames//frames_per_batch)
    env.train()
    for i, data in enumerate(pbar):
        # 1. 收集训练指标（奖励、episode长度等）
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # 2. 执行算法训练（如PPO的clip更新）
        info.update(policy.train_op(data.to_tensordict()))

        # 3. 评估（生成视频，上传到WandB）
        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
            env.train()
            base_env.train()

        # 4. 保存模型
        if save_interval > 0 and i % save_interval == 0:
            try:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        # 5. 日志写入WandB
        run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate())
    run.log(info)

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()

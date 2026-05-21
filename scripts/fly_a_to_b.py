"""
从 A 点飞行到 B 点的演示脚本 —— Headless + 视频录制版

运行方法 (conda sim 环境):
    python scripts/fly_a_to_b.py
"""

import os
import torch
import hydra
from omegaconf import OmegaConf

from omni_drones import init_simulation_app


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    cfg.headless = True
    cfg.task.drone_model.name = "DifferentialUAV"
    cfg.task.drone_model.controller = "LeePositionController"
    cfg.env.num_envs = 1

    init_simulation_app(cfg)

    from omni_drones.envs.single.hover import Hover
    env = Hover(cfg, headless=cfg.headless)

    # 启用 offscreen 渲染 (headless 模式下 viewport render product 仍然可用)
    env.enable_render(True)

    point_A = torch.tensor([[0.0, 0.0, 0.5]], device=env.device)
    point_A_rot = torch.tensor([[1., 0., 0., 0.]], device=env.device)
    point_B = torch.tensor([[5.0, 3.0, 2.5]], device=env.device)
    target_yaw = torch.zeros(1, 1, device=env.device)

    env.reset()
    env.drone.set_world_poses(point_A, point_A_rot)
    env.drone.set_velocities(torch.zeros(1, 1, 6, device=env.device))

    # 帧率 = 1 / (sim.dt * substeps)
    fps = 1.0 / (cfg.sim.dt * cfg.sim.substeps)

    print("-" * 50)
    print(f"无人机已在起点就绪: {point_A[0].tolist()}")
    print(f"目标坐标设定为:   {point_B[0].tolist()}")
    print(f"录制帧率: {fps:.1f} fps (采集间隔: 1 帧)")
    print("-" * 50)

    frames = []
    total_steps = 1000

    for i in range(total_steps + 1):
        # 控制指令
        action = torch.cat([point_B, target_yaw], dim=-1).unsqueeze(1)
        env.drone.apply_action(action)

        # 物理 + 渲染步进
        env.sim.step(render=True)

        # 采集渲染帧 (gather rendered frame after step)
        frame = env.render(mode="rgb_array")
        frames.append(frame)

        if i % 50 == 0:
            current_pos, _ = env.drone.get_world_poses(clone=True)
            x, y, z = current_pos[0, 0, :3].tolist()
            dist_error = torch.norm(current_pos[0, 0] - point_B[0]).item()
            print(f"步数 [{i:4d}/{total_steps}] | 坐标: X={x:>5.2f}, Y={y:>5.2f}, Z={z:>5.2f} | 距目标: {dist_error:.2f}m")

    print("-" * 50)
    print("飞行任务结束！正在保存视频...")

    # 使用 imageio 将帧序列保存为 MP4
    output_dir = "results_video"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fly_a_to_b.mp4")

    import imageio
    import numpy as np

    frames_array = np.stack(frames)  # (T, H, W, 3)
    imageio.mimsave(output_path, frames_array, fps=fps, codec="libx264")
    print(f"视频已保存至: {os.path.abspath(output_path)}")
    print(f"共 {len(frames)} 帧, {fps:.1f} fps")

    env.close()


if __name__ == "__main__":
    main()

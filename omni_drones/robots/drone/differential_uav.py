import torch
from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH


class DifferentialUAV(MultirotorBase):
    usd_path: str = ASSET_PATH + "/usd/differential_uav.usd"
    param_path: str = ASSET_PATH + "/usd/differential_uav.yaml"

"""
Policy-Space Response Oracles (PSRO) Policy.

Population-based training with meta-game analysis for multi-agent
reinforcement learning. Maintains populations of policies and
iteratively computes best responses against meta-strategies.

Reference: Lanctot et al. "A Unified Game-Theoretic Approach to
Multiagent Reinforcement Learning." NeurIPS 2017.
"""

import os
from typing import Dict

import numpy as np
import torch
from tensordict import TensorDict

from omni_drones.utils.torchrl.env import AgentSpec

from ..mappo import MAPPOPolicy
from .utils import Population, Shared_Actor_Population


class PSROPolicy(object):
    """
    PSRO policy using MAPPOPolicy as the inner PPO solver.

    Designed for two-player symmetric games. Each player maintains a
    population of policies. In each PSRO iteration, a meta-strategy
    is computed from the payoff matrix, and a best response is trained
    against it.
    """

    def __init__(self, cfg, agent_spec: AgentSpec, device="cuda") -> None:
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.psro_epochs = cfg.get("psro_epochs", 10)
        self.psro_iterations = cfg.get("psro_iterations", 100)
        self.meta_solver = cfg.get("meta_solver", "nash")
        self.population_dir = cfg.get("population_dir", "./psro_population")

        os.makedirs(self.population_dir, exist_ok=True)

        self.policy = MAPPOPolicy(cfg, agent_spec, device)
        self.population = None
        self.payoff_matrix = None
        self._current_iteration = 0

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        if self.population is not None and len(self.population) > 0:
            return self.population(tensordict)
        return self.policy(tensordict, deterministic=deterministic)

    def train_op(self, tensordict: TensorDict) -> Dict:
        return self.policy.train_op(tensordict)

    def state_dict(self):
        return {
            "policy": self.policy.state_dict(),
            "iteration": self._current_iteration,
        }

    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict["policy"])
        self._current_iteration = state_dict.get("iteration", 0)

    def eval(self):
        self.policy.eval()

    def train(self):
        self.policy.train()

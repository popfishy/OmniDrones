"""
Population management utilities for PSRO.

Provides Population and Shared_Actor_Population classes for managing
collections of policies with on-disk checkpoint storage.
"""

import os
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn, vmap
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictParams


_policy_t = Callable[[TensorDict], TensorDict]


class uniform_policy(nn.Module):
    """Uniform random policy for population initialization."""

    def forward(self, tensordict: TensorDict):
        observation: torch.Tensor = tensordict["agents", "observation"]
        action_dim = 4
        action_shape = observation.shape[:-1] + (action_dim,)
        action = 2 * torch.rand(action_shape, device=observation.device) - 1

        action_log_prob = torch.log(
            torch.ones(size=action_shape, device=observation.device) / 2
        )
        action_log_prob = torch.sum(action_log_prob, dim=-1, keepdim=True)

        action_entropy = torch.ones(
            size=action_shape, device=observation.device
        ) * torch.log(torch.tensor(2.0, device=observation.device))
        action_entropy = torch.sum(action_entropy, dim=-1, keepdim=True)

        tensordict.set(("agents", "action"), action)
        tensordict.set("drone.action_logp", action_log_prob)
        tensordict.set("drone.action_entropy", action_entropy)
        return tensordict


class Population:
    """Manages a collection of policies with on-disk checkpoint storage."""

    def __init__(
        self,
        dir: str,
        module: TensorDictModule,
        initial_policy: Union[uniform_policy, dict] = None,
        device="cuda",
    ):
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        self._module_idx = -1
        self._module = module
        self._current_module_idx = -1
        self.device = device

        self.policy_sets: List[Union[_policy_t, int]] = []
        self._func = None
        self._params = None

        if initial_policy is None:
            initial_policy = uniform_policy()

        if callable(initial_policy):
            self.policy_sets.append(initial_policy)
            self._module_idx += 1
        elif isinstance(initial_policy, dict):
            self.add_actor(initial_policy)
        else:
            raise ValueError("Invalid initial_policy")

        self.sample(meta_policy=np.array([1.0]))

    def __len__(self) -> int:
        return len(self.policy_sets)

    def add_actor(self, actor_dict: dict):
        if len(self.policy_sets) == 1 and callable(self.policy_sets[0]):
            self._module_idx = 0
            torch.save(actor_dict, os.path.join(self.dir, f"{self._module_idx}.pt"))
            self.policy_sets = [self._module_idx]
            self._current_module_idx = -1
        else:
            self._module_idx += 1
            torch.save(actor_dict, os.path.join(self.dir, f"{self._module_idx}.pt"))
            self.policy_sets.append(self._module_idx)
        self.set_latest_policy()

    def _set_policy(self, index: int):
        if self._current_module_idx == index:
            return

        if not isinstance(self.policy_sets[index], int):
            self._func = self.policy_sets[index]
        else:
            assert self._module is not None
            checkpoint = torch.load(
                os.path.join(self.dir, f"{self.policy_sets[index]}.pt")
            )
            self._params = checkpoint["actor_params"].detach()
            self._func = lambda tensordict: vmap(
                self._module, in_dims=(1, 0), out_dims=1, randomness="error"
            )(tensordict, self._params, deterministic=True)

        self._current_module_idx = index

    def set_latest_policy(self):
        self._set_policy(self._module_idx)

    def set_second_latest_policy(self):
        self._set_policy(self._module_idx - 1)

    def set_behavioural_strategy(self, index: int):
        self._set_policy(index)

    def sample(self, meta_policy: np.array):
        if len(meta_policy) == len(self.policy_sets):
            self._set_policy(np.random.choice(len(self.policy_sets), p=meta_policy))
        elif len(meta_policy) == len(self.policy_sets) - 1:
            prob = np.append(meta_policy, 0.0)
            self._set_policy(np.random.choice(len(self.policy_sets), p=prob))
        else:
            raise ValueError("Invalid meta_policy")

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        tensordict = tensordict.to(self.device)
        return self._func(tensordict)

    def _get_policy_checkpoint(self, index: int) -> dict:
        if not isinstance(self.policy_sets[index], int):
            raise ValueError("The policy params are not saved in the population")
        return torch.load(os.path.join(self.dir, f"{self.policy_sets[index]}.pt"))

    def get_latest_policy_checkpoint(self) -> dict:
        return self._get_policy_checkpoint(self._module_idx)


class Shared_Actor_Population(Population):
    """Population for the case where actors share parameters."""

    def _set_policy(self, index):
        if self._current_module_idx == index:
            return

        if not isinstance(self.policy_sets[index], int):
            self._func = self.policy_sets[index]
        else:
            assert self._module is not None
            checkpoint = torch.load(
                os.path.join(self.dir, f"{self.policy_sets[index]}.pt")
            )
            self._params = checkpoint["actor_params"].detach()
            self._func = lambda tensordict: self._module(
                tensordict, self._params, deterministic=True
            )

        self._current_module_idx = index

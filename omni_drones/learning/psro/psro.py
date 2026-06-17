# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Policy-Space Response Oracles (PSRO) Policy.

Population-based training with meta-game analysis for two-player
multi-agent reinforcement learning. Maintains populations of policies
and iteratively computes best responses against meta-strategies.

Reference: Lanctot et al. "A Unified Game-Theoretic Approach to
Multiagent Reinforcement Learning." NeurIPS 2017.
"""

import os
import traceback
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictParams, make_functional
from torch import vmap
from torch.optim import lr_scheduler
from torchrl.data import (
    BoundedTensorSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
)

from omni_drones.utils.torchrl.env import AgentSpec

from ..common import make_encoder
from ..modules.distributions import (
    CustomDiagGaussian,
    MultiCategoricalModule,
)
from ..utils import valuenorm
from ..utils.gae import compute_gae
from .utils import Population

LR_SCHEDULER = lr_scheduler._LRScheduler


class PSROPolicy(object):
    """
    PSRO with PPO as the inner solver.

    Designed for two-player tasks. Agent indices are split into two teams.
    Each team maintains its own population of policies. In each PSRO iteration,
    one team trains a best response against the other team's population.
    """

    def __init__(self, cfg, agent_spec: AgentSpec, device="cuda") -> None:
        super().__init__()
        self._deterministic = False

        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        assert agent_spec.n % 2 == 0, "PSRO requires an even number of agents (two teams)"

        print(self.agent_spec.observation_spec)
        print(self.agent_spec.action_spec)

        self.clip_param = cfg.clip_param
        self.ppo_epoch = int(cfg.ppo_epochs)
        self.num_minibatches = int(cfg.num_minibatches)
        self.normalize_advantages = cfg.normalize_advantages

        self.entropy_coef = cfg.entropy_coef
        self.gae_gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda

        self.act_dim = agent_spec.action_spec.shape[-1]

        self.obs_name = ("agents", "observation")
        self.state_name = ("agents", "state")
        self.act_name = ("agents", "action")
        self.reward_name = ("agents", "reward")

        self.make_actor()
        self.make_critic()

        self.current_player_id = 0
        self.eval_payoff = False

        self.population_0 = Population(
            dir=os.path.join(wandb.run.dir, "population_0"), module=self.actor
        )
        self.population_1 = Population(
            dir=os.path.join(wandb.run.dir, "population_1"), module=self.actor
        )

        self.train_in_keys = list(
            set(
                self.actor_in_keys
                + self.actor_out_keys
                + self.critic_in_keys
                + self.critic_out_keys
                + [
                    "next",
                    self.act_logps_name,
                    ("reward", self.reward_name),
                    "state_value",
                ]
                + ["progress", ("collector", "traj_ids")]
            )
        )

        self.n_updates = 0

    @property
    def act_logps_name(self):
        return f"{self.agent_spec.name}.action_logp"

    def make_actor(self):
        cfg = self.cfg.actor

        self.actor_in_keys = [self.obs_name, self.act_name]
        self.actor_out_keys = [
            self.act_name,
            self.act_logps_name,
            f"{self.agent_spec.name}.action_entropy",
        ]

        if cfg.get("output_dist_params", False):
            self.actor_out_keys.append(("debug", "action_loc"))
            self.actor_out_keys.append(("debug", "action_scale"))

        def create_actor_fn():
            return TensorDictModule(
                _make_ppo_actor(
                    cfg, self.agent_spec.observation_spec, self.agent_spec.action_spec
                ),
                in_keys=self.actor_in_keys,
                out_keys=self.actor_out_keys,
            ).to(self.device)

        actors = nn.ModuleList([create_actor_fn() for _ in range(self.agent_spec.n)])

        self.actor = actors[0]

        actor_params = [make_functional(actor) for actor in actors]
        self.actor_params_0 = TensorDictParams(
            torch.stack(actor_params[: self.agent_spec.n // 2]).to_tensordict()
        )
        self.initial_actor_params_0 = self.actor_params_0.clone()
        self.actor_params_1 = TensorDictParams(
            torch.stack(actor_params[self.agent_spec.n // 2 :]).to_tensordict()
        )
        self.initial_actor_params_1 = self.actor_params_1.clone()

        self.actor_opt_0 = torch.optim.Adam(self.actor_params_0.parameters(), lr=cfg.lr)
        self.initial_actor_opt_0_state_dict = self.actor_opt_0.state_dict()
        self.actor_opt_1 = torch.optim.Adam(self.actor_params_1.parameters(), lr=cfg.lr)
        self.initial_actor_opt_1_state_dict = self.actor_opt_1.state_dict()

    def make_critic(self):
        cfg = self.cfg.critic

        if cfg.use_huber_loss:
            self.critic_loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
        else:
            self.critic_loss_fn = nn.MSELoss()

        assert self.cfg.critic_input in ("state", "obs")
        if self.cfg.critic_input == "state" and self.agent_spec.state_spec is not None:
            self.critic_in_keys = [self.state_name]
            self.critic_out_keys = ["state_value"]

            reward_spec = self.agent_spec.reward_spec
            critic = _make_critic(
                cfg, self.agent_spec.state_spec, reward_spec, centralized=True
            )
            self.critic = TensorDictModule(
                critic,
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = self.critic
        else:
            self.critic_in_keys = [self.obs_name]
            self.critic_out_keys = ["state_value"]

            critic = _make_critic(
                cfg,
                self.agent_spec.observation_spec,
                self.agent_spec.reward_spec,
                centralized=False,
            )
            self.critic = TensorDictModule(
                critic,
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = self.critic

        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = cfg.lr_scheduler
        if scheduler is not None:
            scheduler = eval(scheduler)
            self.critic_opt_scheduler: LR_SCHEDULER = scheduler(
                self.critic_opt, **cfg.lr_scheduler_kwargs
            )

        if hasattr(cfg, "value_norm") and cfg.value_norm is not None:
            cls = getattr(valuenorm, cfg.value_norm["class"])
            self.value_normalizer: valuenorm.Normalizer = cls(
                input_shape=self.agent_spec.reward_spec.shape[-2:],
                **cfg.value_norm["kwargs"],
            ).to(self.device)

    def value_op(self, tensordict: TensorDict) -> TensorDict:
        critic_input = tensordict.select(*self.critic_in_keys, strict=False)

        if self.cfg.critic_input == "obs":
            critic_input.batch_size = [*critic_input.batch_size, self.agent_spec.n]
        tensordict = self.value_func(critic_input)
        return tensordict

    def __call__(self, tensordict: TensorDict, deterministic: Optional[bool] = None):
        if deterministic is None:
            deterministic = self._deterministic
        actor_input = tensordict.select(*self.actor_in_keys, strict=False)

        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        actor_input_0, actor_input_1 = torch.split(
            actor_input, [self.agent_spec.n // 2, self.agent_spec.n // 2], dim=1
        )

        if self.eval_payoff:
            actor_output_0 = self.population_0(actor_input_0)
            actor_output_1 = self.population_1(actor_input_1)
        elif self.current_player_id == 0:
            actor_output_0 = vmap(
                self.actor, in_dims=(1, 0), out_dims=1, randomness="different"
            )(actor_input_0, self.actor_params_0, deterministic=deterministic)
            actor_output_1 = self.population_1(actor_input_1)
        elif self.current_player_id == 1:
            actor_output_0 = self.population_0(actor_input_0)
            actor_output_1 = vmap(
                self.actor, in_dims=(1, 0), out_dims=1, randomness="different"
            )(actor_input_1, self.actor_params_1, deterministic=deterministic)

        try:
            actor_output = torch.cat([actor_output_0, actor_output_1], dim=1)
        except Exception:
            traceback.print_exc()
            raise

        tensordict.update(actor_output)
        tensordict.update(self.value_op(tensordict))
        return tensordict

    def update_actor(self, batch: TensorDict, agent_idx: int) -> Dict[str, Any]:
        assert agent_idx in (0, 1)

        advantages = batch["advantages"]
        if agent_idx == 0:
            advantages = advantages[:, : self.agent_spec.n // 2]
        else:
            advantages = advantages[:, self.agent_spec.n // 2 :]

        actor_input = batch.select(*self.actor_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        if agent_idx == 0:
            actor_input = actor_input[:, : self.agent_spec.n // 2]
            actor_param = self.actor_params_0
        else:
            actor_input = actor_input[:, self.agent_spec.n // 2 :]
            actor_param = self.actor_params_1

        actor_output = vmap(
            self.actor, in_dims=(1, 0), out_dims=1, randomness="different"
        )(actor_input, actor_param, eval_action=True)

        log_probs_old = batch[self.act_logps_name]
        if agent_idx == 0:
            log_probs_old = log_probs_old[:, : self.agent_spec.n // 2]
        else:
            log_probs_old = log_probs_old[:, self.agent_spec.n // 2 :]

        log_probs_new = actor_output[self.act_logps_name]
        dist_entropy = actor_output[f"{self.agent_spec.name}.action_entropy"]

        assert advantages.shape == log_probs_new.shape == dist_entropy.shape

        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        policy_loss = -torch.mean(torch.min(surr1, surr2) * self.act_dim)
        entropy_loss = -torch.mean(dist_entropy)

        if agent_idx == 0:
            actor_opt = self.actor_opt_0
        else:
            actor_opt = self.actor_opt_1

        actor_opt.zero_grad()
        (policy_loss + entropy_loss * self.cfg.entropy_coef).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
        )
        actor_opt.step()

        ess = (
            2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)
        ).exp().mean() / ratio.shape[0]
        return {
            "policy_loss": policy_loss.item(),
            "actor_grad_norm": grad_norm.item(),
            "entropy": -entropy_loss.item(),
            "ESS": ess.item(),
        }

    def update_critic(self, batch: TensorDict, agent_idx: int = None) -> Dict[str, Any]:
        assert agent_idx in (0, 1, None)

        critic_input = batch.select(*self.critic_in_keys)
        values = self.value_op(critic_input)["state_value"]
        b_values = batch["state_value"]
        b_returns = batch["returns"]

        if agent_idx == 0:
            values = values[:, : (self.agent_spec.n // 2), :]
            b_values = b_values[:, : (self.agent_spec.n // 2), :]
            b_returns = b_returns[:, : (self.agent_spec.n // 2), :]
        elif agent_idx == 1:
            values = values[:, (self.agent_spec.n // 2) :, :]
            b_values = b_values[:, (self.agent_spec.n // 2) :, :]
            b_returns = b_returns[:, (self.agent_spec.n // 2) :, :]

        assert values.shape == b_values.shape == b_returns.shape
        value_pred_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )

        value_loss_clipped = self.critic_loss_fn(b_returns, value_pred_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.max_grad_norm
        )
        self.critic_opt.step()
        self.critic_opt.zero_grad(set_to_none=True)
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return {
            "value_loss": value_loss.mean(),
            "critic_grad_norm": grad_norm.item(),
            "explained_var": explained_var.item(),
        }

    def _get_dones(self, tensordict: TensorDict):
        env_done = tensordict[("next", "done")].unsqueeze(-1)
        agent_done = tensordict.get(
            ("next", f"{self.agent_spec.name}.done"),
            env_done.expand(*env_done.shape[:-2], self.agent_spec.n, 1),
        )
        done = agent_done | env_done
        return done

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.select(*self.train_in_keys, strict=False)
        next_tensordict = tensordict["next"][:, -1]
        with torch.no_grad():
            value_output = self.value_op(next_tensordict)

        rewards = tensordict.get(("next", *self.reward_name))
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)

        values = tensordict["state_value"]
        next_value = value_output["state_value"].squeeze(0)

        if hasattr(self, "value_normalizer"):
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)

        dones = self._get_dones(tensordict)

        tensordict["advantages"], tensordict["returns"] = compute_gae(
            rewards,
            dones,
            values,
            next_value,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
        )

        advantages_mean = tensordict["advantages"].mean()
        advantages_std = tensordict["advantages"].std()
        if self.normalize_advantages:
            tensordict["advantages"] = (tensordict["advantages"] - advantages_mean) / (
                advantages_std + 1e-8
            )

        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(tensordict["returns"])
            tensordict["returns"] = self.value_normalizer.normalize(
                tensordict["returns"]
            )

        train_info = []
        for ppo_epoch in range(self.ppo_epoch):
            dataset = _make_dataset_naive(
                tensordict,
                int(self.cfg.num_minibatches),
                1,
            )
            for minibatch in dataset:
                train_info.append(
                    TensorDict(
                        {
                            **self.update_actor(
                                minibatch, agent_idx=self.current_player_id
                            ),
                            **self.update_critic(minibatch),
                        },
                        batch_size=[],
                    )
                )

        train_info = {k: v.mean().item() for k, v in torch.stack(train_info).items()}
        train_info["advantages_mean"] = advantages_mean.item()
        train_info["advantages_std"] = advantages_std.item()
        if isinstance(
            self.agent_spec.action_spec, (BoundedTensorSpec, UnboundedTensorSpec)
        ):
            train_info["action_norm"] = (
                tensordict[self.act_name].norm(dim=-1).mean().item()
            )
        if hasattr(self, "value_normalizer"):
            train_info["value_running_mean"] = (
                self.value_normalizer.running_mean.mean().item()
            )

        self.n_updates += 1
        return {f"{self.agent_spec.name}/{k}": v for k, v in train_info.items()}

    def append_actor(self, current_player_id: int, share_population: bool = False):
        actor_params_0 = self.actor_params_0.clone()
        actor_dict_0 = {
            "actor_params": actor_params_0,
            "actor_opt_params": self.actor_opt_0.state_dict(),
        }

        actor_params_1 = self.actor_params_1.clone()
        actor_dict_1 = {
            "actor_params": actor_params_1,
            "actor_opt_params": self.actor_opt_1.state_dict(),
        }

        assert current_player_id in (0, 1)
        if current_player_id == 0:
            if share_population:
                self.population_0.add_actor(actor_dict_0)
                self.population_1.add_actor(actor_dict_0)
            else:
                self.population_0.add_actor(actor_dict_0)
        elif current_player_id == 1:
            if share_population:
                self.population_0.add_actor(actor_dict_1)
                self.population_1.add_actor(actor_dict_1)
            else:
                self.population_1.add_actor(actor_dict_1)

    def switch_current_player_id(self):
        self.current_player_id = 1 - self.current_player_id

    def sample_pure_strategy(
        self, meta_policy_0: Optional[np.ndarray], meta_policy_1: Optional[np.ndarray]
    ):
        if self.current_player_id == 0:
            self.population_1.sample(meta_policy=meta_policy_1)
        else:
            self.population_0.sample(meta_policy=meta_policy_0)

    def both_sample_pure_strategy(
        self, meta_policy_0: Optional[np.ndarray], meta_policy_1: Optional[np.ndarray]
    ):
        self.population_1.sample(meta_policy=meta_policy_1)
        self.population_0.sample(meta_policy=meta_policy_0)

    def set_pure_strategy(self, idx_0: int, idx_1: int):
        self.population_0.set_behavioural_strategy(idx_0)
        self.population_1.set_behavioural_strategy(idx_1)

    def set_latest_strategy(
        self, set_population_0: bool = True, set_population_1: bool = True
    ):
        if set_population_0:
            self.population_0.set_latest_policy()
        if set_population_1:
            self.population_1.set_latest_policy()

    def set_actor_params_with_latest_policy(self):
        if self.current_player_id == 0:
            checkpoint = self.population_0.get_latest_policy_checkpoint()
            self.actor_params_0 = checkpoint["actor_params"]
            self.actor_opt_0 = torch.optim.Adam(
                self.actor_params_0.parameters(), lr=self.cfg.actor.lr
            )
            if "actor_opt_params" in checkpoint:
                self.actor_opt_0.load_state_dict(checkpoint["actor_opt_params"])
        else:
            checkpoint = self.population_1.get_latest_policy_checkpoint()
            self.actor_params_1 = checkpoint["actor_params"]
            self.actor_opt_1 = torch.optim.Adam(
                self.actor_params_1.parameters(), lr=self.cfg.actor.lr
            )
            if "actor_opt_params" in checkpoint:
                self.actor_opt_1.load_state_dict(checkpoint["actor_opt_params"])

    def set_actor_params_with_initial_random_policy(self):
        if self.current_player_id == 0:
            self.actor_params_0 = self.initial_actor_params_0.clone()
            self.actor_opt_0 = torch.optim.Adam(
                self.actor_params_0.parameters(), lr=self.cfg.actor.lr
            )
            self.actor_opt_0.load_state_dict(self.initial_actor_opt_0_state_dict)
        else:
            self.actor_params_1 = self.initial_actor_params_1.clone()
            self.actor_opt_1 = torch.optim.Adam(
                self.actor_params_1.parameters(), lr=self.cfg.actor.lr
            )
            self.actor_opt_1.load_state_dict(self.initial_actor_opt_1_state_dict)

    def state_dict(self):
        state_dict = {
            "critic": self.critic.state_dict(),
            "actor_params_0": self.actor_params_0,
            "actor_params_1": self.actor_params_1,
            "value_normalizer": self.value_normalizer.state_dict(),
        }
        return state_dict

    def load_actor_dict(self, actor_dict_0: dict, actor_dict_1: dict):
        self.actor_params_0 = actor_dict_0["actor_params"]
        self.actor_opt_0 = torch.optim.Adam(
            self.actor_params_0.parameters(), lr=self.cfg.actor.lr
        )
        self.actor_opt_0.load_state_dict(actor_dict_0["actor_opt_params"])

        self.actor_params_1 = actor_dict_1["actor_params"]
        self.actor_opt_1 = torch.optim.Adam(
            self.actor_params_1.parameters(), lr=self.cfg.actor.lr
        )
        self.actor_opt_1.load_state_dict(actor_dict_1["actor_opt_params"])

        self.population_0 = Population(
            dir=os.path.join(wandb.run.dir, "population_0"),
            module=self.actor,
            initial_policy=actor_dict_0,
        )
        self.population_1 = Population(
            dir=os.path.join(wandb.run.dir, "population_1"),
            module=self.actor,
            initial_policy=actor_dict_1,
        )

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()


def _make_dataset_naive(
    tensordict: TensorDict, num_minibatches: int = 4, seq_len: int = 1
):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]


def _make_ppo_actor(cfg, observation_spec: TensorSpec, action_spec: TensorSpec):
    encoder = make_encoder(cfg, observation_spec)

    if isinstance(action_spec, MultiDiscreteTensorSpec):
        act_dist = MultiCategoricalModule(
            encoder.output_shape.numel(),
            torch.as_tensor(action_spec.nvec.storage().float()).long(),
        )
    elif isinstance(action_spec, DiscreteTensorSpec):
        act_dist = MultiCategoricalModule(
            encoder.output_shape.numel(), [action_spec.space.n]
        )
    elif isinstance(action_spec, (UnboundedTensorSpec, BoundedTensorSpec)):
        action_dim = action_spec.shape[-1]

        create_dist_func = cfg.get("create_dist_func", "default")
        if create_dist_func == "default":
            create_dist_func = None
        else:
            raise NotImplementedError(f"Unknown create_dist_func: {create_dist_func}")

        act_dist = CustomDiagGaussian(
            encoder.output_shape.numel(),
            action_dim,
            False,
            0.01,
            create_dist_func=create_dist_func,
        )
    else:
        raise NotImplementedError(action_spec)

    return PSROActor(
        encoder, act_dist, output_dist_params=cfg.get("output_dist_params", False)
    )


def _make_critic(cfg, state_spec: TensorSpec, reward_spec: TensorSpec, centralized=False):
    from ..modules.networks import MLP

    assert isinstance(reward_spec, (UnboundedTensorSpec, BoundedTensorSpec))
    encoder = make_encoder(cfg, state_spec)

    if centralized:
        v_out = nn.Linear(
            encoder.output_shape.numel(), reward_spec.shape[-2:].numel()
        )
        nn.init.orthogonal_(v_out.weight, cfg.gain)
        return PSROCritic(encoder, v_out, reward_spec.shape[-2:])
    else:
        v_out = nn.Linear(encoder.output_shape.numel(), reward_spec.shape[-1])
        nn.init.orthogonal_(v_out.weight, cfg.gain)
        return PSROCritic(encoder, v_out, reward_spec.shape[-1:])


def _is_independent_normal(dist):
    return isinstance(dist, torch.distributions.Independent) and isinstance(
        dist.base_dist, torch.distributions.Normal
    )


class PSROActor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        act_dist: nn.Module,
        output_dist_params: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.act_dist = act_dist
        self.output_dist_params = output_dist_params

    def forward(
        self,
        obs: Union[torch.Tensor, TensorDict],
        action: torch.Tensor = None,
        deterministic=False,
        eval_action=False,
    ):
        actor_features = self.encoder(obs)
        action_dist = self.act_dist(actor_features)

        if self.output_dist_params and _is_independent_normal(action_dist):
            loc = action_dist.base_dist.loc
            scale = action_dist.base_dist.scale
        else:
            loc = None
            scale = None

        if eval_action:
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            dist_entropy = action_dist.entropy().unsqueeze(-1)
            return action, action_log_probs, dist_entropy, loc, scale
        else:
            if deterministic:
                action = action_dist.mode
            else:
                action = action_dist.sample()
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            dist_entropy = action_dist.entropy().unsqueeze(-1)
            return action, action_log_probs, dist_entropy, loc, scale


class PSROCritic(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        v_out: nn.Module,
        output_shape: torch.Size = torch.Size((-1,)),
    ):
        super().__init__()
        self.base = base
        self.v_out = v_out
        self.output_shape = output_shape

    def forward(self, critic_input: torch.Tensor):
        critic_features = self.base(critic_input)
        values = self.v_out(critic_features)

        if len(self.output_shape) > 1:
            values = values.unflatten(-1, self.output_shape)
        return values

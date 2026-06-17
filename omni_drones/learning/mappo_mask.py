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
MAPPOPolicy with action masking support.

Extends MAPPOPolicy to support environment-provided action masks.
When a mask is present, masked actions receive zero gradient contribution
in both policy loss and value loss.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any, Dict

from .mappo import MAPPOPolicy


class MAPPOPolicyMask(MAPPOPolicy):
    def __init__(self, cfg, agent_spec, device="cuda") -> None:
        super().__init__(cfg, agent_spec, device)
        self.mask_name = cfg.get("mask_name", None)

        if self.mask_name is not None:
            if self.mask_name not in self.train_in_keys:
                self.train_in_keys.append(self.mask_name)

    def update_actor(self, batch: TensorDict) -> Dict[str, Any]:
        advantages = batch["advantages"]
        actor_input = batch.select(*self.actor_in_keys)
        if "is_init" in actor_input.keys():
            from tensordict.utils import expand_right
            actor_input["is_init"] = expand_right(
                actor_input["is_init"], (*actor_input.batch_size, self.agent_spec.n)
            )
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]

        log_probs_old = batch[self.act_logps_name]
        if hasattr(self, "minibatch_seq_len"):
            from torch import vmap
            actor_output = vmap(self.actor, in_dims=(2, 0), out_dims=2)(
                actor_input, self.actor_params, eval_action=True
            )
        else:
            if self.cfg.share_actor:
                actor_output = self.actor(
                    actor_input, self.actor_params, eval_action=True
                )
            else:
                from torch import vmap
                actor_output = vmap(self.actor, in_dims=(1, 0), out_dims=1)(
                    actor_input, self.actor_params, eval_action=True
                )

        log_probs_new = actor_output[self.act_logps_name]
        dist_entropy = actor_output[f"{self.agent_spec.name}.action_entropy"]

        assert advantages.shape == log_probs_new.shape == dist_entropy.shape

        mask = None
        if self.mask_name is not None:
            mask = batch.get(self.mask_name)
            if mask is not None:
                mask = mask.unsqueeze(-1)

        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        if mask is not None:
            policy_loss = -torch.mean(torch.min(surr1, surr2) * self.act_dim * mask)
            entropy_loss = -torch.mean(dist_entropy * mask)
        else:
            policy_loss = -torch.mean(torch.min(surr1, surr2) * self.act_dim)
            entropy_loss = -torch.mean(dist_entropy)

        self.actor_opt.zero_grad()
        (policy_loss + entropy_loss * self.cfg.entropy_coef).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
        )
        self.actor_opt.step()

        ess = (2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)).exp().mean() / ratio.shape[0]
        return {
            "policy_loss": policy_loss.item(),
            "actor_grad_norm": grad_norm.item(),
            "entropy": -entropy_loss.item(),
            "ESS": ess.item()
        }

    def update_critic(self, batch: TensorDict) -> Dict[str, Any]:
        mask = None
        if self.mask_name is not None:
            mask = batch.get(self.mask_name)
            if mask is not None:
                mask = mask.unsqueeze(-1)

        critic_input = batch.select(*self.critic_in_keys)
        values = self.value_op(critic_input)["state_value"]
        b_values = batch["state_value"]
        b_returns = batch["returns"]
        assert values.shape == b_values.shape == b_returns.shape
        value_pred_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )

        if mask is not None:
            value_loss_clipped = self.critic_loss_fn(b_returns * mask, value_pred_clipped * mask)
            value_loss_original = self.critic_loss_fn(b_returns * mask, values * mask)
        else:
            value_loss_clipped = self.critic_loss_fn(b_returns, value_pred_clipped)
            value_loss_original = self.critic_loss_fn(b_returns, values)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.max_grad_norm
        )
        self.critic_opt.step()
        self.critic_opt.zero_grad(set_to_none=True)
        explained_var = 1 - torch.nn.functional.mse_loss(values, b_returns) / b_returns.var()
        return {
            "value_loss": value_loss.mean(),
            "critic_grad_norm": grad_norm.item(),
            "explained_var": explained_var.item()
        }

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["actor_opt"] = self.actor_opt.state_dict()
        state_dict["critic_opt"] = self.critic_opt.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "actor_opt" in state_dict:
            self.actor_opt.load_state_dict(state_dict["actor_opt"])
        if "critic_opt" in state_dict:
            self.critic_opt.load_state_dict(state_dict["critic_opt"])

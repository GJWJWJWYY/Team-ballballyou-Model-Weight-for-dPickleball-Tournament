from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

CHANNEL_MEAN = np.array([0.41899611, 0.60789724, 0.82478420], dtype=np.float32)
CHANNEL_STD = np.array([0.23628990, 0.22181486, 0.15711305], dtype=np.float32)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _repeat_mean_std(frame_stack: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = np.tile(CHANNEL_MEAN, frame_stack).astype(np.float32)
    std = np.tile(CHANNEL_STD, frame_stack).astype(np.float32)
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return mean_t, std_t


@dataclass
class PixelAgentConfig:
    obs_shape: Tuple[int, int, int] = (30, 51, 131)
    action_dims: Tuple[int, int, int] = (3, 3, 3)
    latent_dim: int = 256
    hidden_dim: int = 512
    lr: float = 3e-4
    gamma: float = 0.993
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ppo_epochs: int = 4
    ppo_batch_size: int = 256
    rollout_length: int = 10
    device: torch.device = _default_device()


class ConvEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], latent_dim: int, hidden_dim: int):
        super().__init__()
        channels, height, width = obs_shape
        self.backbone = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            out = self.backbone(dummy)
        self.flatten_dim = out.view(1, -1).size(1)
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(obs)
        x = x.view(x.size(0), -1)
        return self.head(x)


class PolicyHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, action_dims: Tuple[int, int, int]):
        super().__init__()
        self.heads = nn.ModuleList()
        for dim in action_dims:
            self.heads.append(
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.SiLU(inplace=True),
                    nn.Linear(hidden_dim, dim),
                )
            )

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(head(latent) for head in self.heads)


class ValueHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


class PixelPolicy(nn.Module):
    def __init__(self, cfg: PixelAgentConfig):
        super().__init__()
        self.cfg = cfg
        self.frame_stack = cfg.obs_shape[0] // 3

        self.encoder = ConvEncoder(cfg.obs_shape, cfg.latent_dim, cfg.hidden_dim)
        self.policy_head = PolicyHead(cfg.latent_dim, cfg.hidden_dim // 2, cfg.action_dims)
        self.value_head = ValueHead(cfg.latent_dim, cfg.hidden_dim // 2)

        mean, std = _repeat_mean_std(self.frame_stack, torch.device("cpu"))
        self.register_buffer("channel_mean", mean, persistent=False)
        self.register_buffer("channel_std", std, persistent=False)

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.device != self.channel_mean.device:
            self.channel_mean = self.channel_mean.to(obs.device)
            self.channel_std = self.channel_std.to(obs.device)
        obs = obs.float() / 255.0
        return (obs - self.channel_mean) / self.channel_std

    def forward(self, obs: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        norm_obs = self.preprocess(obs)
        latent = self.encoder(norm_obs)
        logits = self.policy_head(latent)
        value = self.value_head(latent)
        return logits, value, latent

    def evaluate_value(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.preprocess(obs)
        latent = self.encoder(norm_obs)
        return self.value_head(latent)


class PPOBuffer:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.obs_list: List[torch.Tensor] = []
        self.next_obs_list: List[torch.Tensor] = []
        self.actions_list: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        next_obs: torch.Tensor,
    ):
        self.obs_list.append(obs.detach())
        self.actions_list.append(action.detach())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.next_obs_list.append(next_obs.detach())

    def compute_advantages(self, policy: PixelPolicy, gamma: float, gae_lambda: float) -> Dict[str, torch.Tensor]:
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        values = torch.stack(self.values)

        if dones[-1] > 0.5:
            last_value = torch.zeros(1, device=self.device)
        else:
            with torch.no_grad():
                last_obs = self.next_obs_list[-1].unsqueeze(0)
                last_value = policy.evaluate_value(last_obs)

        values = torch.cat([values, last_value])

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + gamma * values[step + 1] * mask - values[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[step] = gae
        returns = advantages + values[:-1]

        obs = torch.stack(self.obs_list)
        actions = torch.stack(self.actions_list)
        old_log_probs = torch.stack(self.log_probs)

        return {
            "obs": obs,
            "actions": actions,
            "returns": returns,
            "advantages": advantages,
            "old_log_probs": old_log_probs,
        }


class PixelPolicyAgent(nn.Module):
    def __init__(self, cfg: PixelAgentConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.policy = PixelPolicy(cfg).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float, torch.Tensor]:
        self.policy.eval()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        logits, value, _ = self.policy(obs_tensor)

        actions = []
        log_prob = torch.zeros(1, device=self.device)
        for logit in logits:
            dist = Categorical(logits=logit)
            if deterministic:
                action = torch.argmax(logit, dim=-1)
            else:
                action = dist.sample()
            actions.append(action)
            log_prob += dist.log_prob(action)

        action_tensor = torch.stack(actions, dim=-1)
        return (
            action_tensor.squeeze(0).cpu().numpy().astype(np.int32),
            log_prob.item(),
            value.squeeze(0).item(),
            obs_tensor.squeeze(0).detach(),
        )

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, _ = self.policy(obs)
        log_probs = torch.zeros(actions.size(0), device=self.device)
        entropy = torch.zeros(actions.size(0), device=self.device)
        for i, logit in enumerate(logits):
            dist = Categorical(logits=logit)
            log_probs += dist.log_prob(actions[:, i])
            entropy += dist.entropy()
        return log_probs, entropy, values

    def ppo_update(self, buffer: PPOBuffer):
        data = buffer.compute_advantages(self.policy, self.cfg.gamma, self.cfg.gae_lambda)

        advantages = data["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        obs = data["obs"].to(self.device)
        actions = data["actions"].to(self.device)
        returns = data["returns"].to(self.device)
        old_log_probs = data["old_log_probs"].to(self.device)

        num_samples = obs.size(0)
        batch_size = min(self.cfg.ppo_batch_size, num_samples)

        for _ in range(self.cfg.ppo_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                idx = indices[start : start + batch_size]
                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_old_log_probs = old_log_probs[idx]

                log_probs, entropy, values = self.evaluate_actions(batch_obs, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = entropy.mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

    def save(self, path: str):
        torch.save({"model": self.policy.state_dict()}, path)

    def load(self, path: str, map_location: Optional[str] = None):
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["model"])


class PixelSequenceReplayBuffer:
    def __init__(self, capacity: int = 240_000, horizon: int = 10):
        from collections import deque

        self.capacity = capacity
        self.horizon = horizon
        self.storage = deque(maxlen=capacity)
        self.current_episode = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool, next_obs: np.ndarray):
        entry = (
            obs.astype(np.uint8),
            action.astype(np.int32),
            float(reward),
            1.0 if done else 0.0,
            next_obs.astype(np.uint8),
            self.current_episode,
        )
        self.storage.append(entry)
        if done:
            self.current_episode += 1

    def sample(self, batch_size: int, frame_stack: int, crop: Tuple[int, int, int, int]) -> Dict[str, torch.Tensor]:
        assert len(self.storage) > self.horizon, "Not enough data in buffer"

        obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []

        storage_len = len(self.storage)
        max_start = storage_len - self.horizon
        x1, y1, x2, y2 = crop

        for _ in range(batch_size):
            attempts = 0
            while True:
                start = np.random.randint(0, max_start)
                seq = [self.storage[start + t] for t in range(self.horizon)]
                episode_ids = [item[5] for item in seq]
                if len(set(episode_ids)) == 1 and all(item[3] == 0.0 for item in seq[:-1]):
                    break
                attempts += 1
                if attempts >= 100:
                    break

            obs_seq = []
            actions_seq = []
            rewards_seq = []
            dones_seq = []

            for transition in seq:
                frame = transition[0][:, y1:y2, x1:x2]
                obs_seq.append(frame)
                actions_seq.append(transition[1])
                rewards_seq.append(transition[2])
                dones_seq.append(transition[3])

            terminal_next_obs = seq[-1][4][:, y1:y2, x1:x2]
            stacked_obs = np.stack(obs_seq + [terminal_next_obs], axis=0)

            obs_batch.append(stacked_obs)
            action_batch.append(actions_seq)
            reward_batch.append(rewards_seq)
            done_batch.append(dones_seq)

        obs_tensor = torch.from_numpy(np.array(obs_batch, dtype=np.uint8))

        return {
            "obs": obs_tensor,
            "actions": torch.from_numpy(np.array(action_batch, dtype=np.int64)),
            "rewards": torch.from_numpy(np.array(reward_batch, dtype=np.float32)),
            "dones": torch.from_numpy(np.array(done_batch, dtype=np.float32)),
        }

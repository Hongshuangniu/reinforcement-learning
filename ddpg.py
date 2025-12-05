"""
10kV变压器智能冷却控制系统 - DDPG算法（修复版）
Deep Deterministic Policy Gradient
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple, List
from networks import BaseActorNetwork, BaseCriticNetwork
from config import CONFIG


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """Ornstein-Uhlenbeck噪声，用于探索"""

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class DDPG:
    """DDPG算法（修复版）"""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = CONFIG.device

        # 创建网络
        self.actor = BaseActorNetwork(state_dim, action_dim, CONFIG.train.HIDDEN_DIM).to(self.device)
        self.actor_target = BaseActorNetwork(state_dim, action_dim, CONFIG.train.HIDDEN_DIM).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = BaseCriticNetwork(state_dim, action_dim, CONFIG.train.HIDDEN_DIM).to(self.device)
        self.critic_target = BaseCriticNetwork(state_dim, action_dim, CONFIG.train.HIDDEN_DIM).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=CONFIG.train.LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CONFIG.train.LEARNING_RATE_CRITIC)

        # 经验回放
        self.replay_buffer = ReplayBuffer(CONFIG.train.BUFFER_SIZE)

        # 探索噪声
        self.noise = OUNoise(action_dim, sigma=CONFIG.algo.NOISE_STD)

        # 超参数
        self.gamma = CONFIG.train.GAMMA
        self.tau = CONFIG.train.TAU

        # 训练统计
        self.actor_losses = []
        self.critic_losses = []
        self.training_step = 0
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, evaluate: bool = False,
                      deterministic: bool = False) -> np.ndarray:
        """
        选择动作

        Args:
            state: 状态
            evaluate: 是否使用确定性策略(兼容参数名)
            deterministic: 是否使用确定性策略(兼容参数名)

        Returns:
            动作
        """
        # 统一两种参数名
        is_deterministic = evaluate or deterministic

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor, deterministic=True).cpu().numpy()[0]

        # 添加探索噪声
        if not is_deterministic:
            noise = self.noise.sample()
            action = action + noise

        # 裁剪到有效范围
        action = np.clip(action, -1, 1)

        # 映射到实际动作空间
        real_action = np.array([
            action[0] * 1.5 + 3.5,  # 压力: 2-5 kPa
            (action[1] + 1) / 2,  # 帕尔贴: 0-1
            (action[2] + 1) * 50  # 阀门: 0-100%
        ])

        return real_action

    def update(self):
        """
        更新网络参数（新增方法，用于trainer接口统一）
        """
        if len(self.replay_buffer) < CONFIG.train.BATCH_SIZE:
            return

        # 调用train_step进行实际更新
        self.train_step(CONFIG.train.BATCH_SIZE)

    def train_step(self, batch_size: int = 256) -> Tuple[float, float]:
        """
        训练一步

        Args:
            batch_size: 批次大小

        Returns:
            actor_loss, critic_loss
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0

        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 更新Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states, deterministic=True)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), CONFIG.train.GRAD_CLIP_NORM)
        self.critic_optimizer.step()

        # 更新Actor
        actions_new = self.actor(states, deterministic=True)
        actor_loss = -self.critic.Q1(states, actions_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), CONFIG.train.GRAD_CLIP_NORM)
        self.actor_optimizer.step()

        # 软更新目标网络
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.training_step += 1

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'training_step': self.training_step
        }, filepath)

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.training_step = checkpoint.get('training_step', 0)

    def get_training_stats(self):
        """获取训练统计信息"""
        return {
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'training_step': self.training_step
        }


if __name__ == "__main__":
    print("=" * 60)
    print("DDPG算法测试")
    print("=" * 60)

    # 创建DDPG智能体
    state_dim = CONFIG.env.STATE_DIM
    action_dim = CONFIG.env.ACTION_DIM

    agent = DDPG(state_dim, action_dim)
    print(f"✓ DDPG智能体创建成功")
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {action_dim}")

    # 测试动作选择
    test_state = np.random.randn(state_dim)
    action = agent.select_action(test_state)
    print(f"\n✓ 动作选择测试")
    print(f"  动作: {action}")

    # 测试训练
    for i in range(10):
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False

        agent.store_transition(state, action, reward, next_state, done)

    # 训练几步
    if len(agent.replay_buffer) >= 10:
        actor_loss, critic_loss = agent.train_step(batch_size=10)
        print(f"\n✓ 训练测试")
        print(f"  Actor Loss: {actor_loss:.4f}")
        print(f"  Critic Loss: {critic_loss:.4f}")

    print("\n" + "=" * 60)
    print("✓ DDPG算法测试完成")
    print("=" * 60)
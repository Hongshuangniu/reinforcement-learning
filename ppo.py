"""
10kV变压器智能冷却控制系统 - PPO算法
Proximal Policy Optimization Algorithm
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

from config import CONFIG
from networks import PPOActorNetwork, PPOCriticNetwork


class PPO:
    """
    PPO算法实现
    用作基线对比算法
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = CONFIG.device

        # 创建网络
        self.actor = PPOActorNetwork(
            state_dim, action_dim, CONFIG.train.HIDDEN_DIM
        ).to(self.device)

        self.critic = PPOCriticNetwork(
            state_dim, CONFIG.train.HIDDEN_DIM
        ).to(self.device)

        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=CONFIG.train.LEARNING_RATE_ACTOR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=CONFIG.train.LEARNING_RATE_CRITIC
        )

        # 超参数
        self.gamma = CONFIG.train.GAMMA
        self.clip_epsilon = CONFIG.algo.PPO_CLIP_EPSILON
        self.entropy_coef = CONFIG.algo.PPO_ENTROPY_COEF
        self.value_loss_coef = CONFIG.algo.PPO_VALUE_LOSS_COEF
        self.update_epochs = CONFIG.algo.PPO_UPDATE_EPOCHS
        self.batch_size = CONFIG.train.BATCH_SIZE

        # 缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }

        # 统计
        self.episode_rewards = []
        self.losses = {'actor': [], 'critic': []}
        self.entropies = []
        self.training_step = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        选择动作

        Args:
            state: 当前状态
            evaluate: 是否为评估模式

        Returns:
            动作数组
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                action = self.actor(state)
            else:
                dist = self.actor.get_distribution(state)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)

            value = self.critic(state)

        action_np = action.cpu().numpy().flatten()
        value_np = value.cpu().numpy().item()

        # 计算log_prob用于训练
        if not evaluate:
            with torch.no_grad():
                dist = self.actor.get_distribution(state)
                log_prob = dist.log_prob(action).sum(-1).cpu().numpy().item()
        else:
            log_prob = 0

        # 映射到实际动作空间
        real_action = np.array([
            action_np[0] * 1.5 + 3.5,  # 压力: 2-5 kPa
            (action_np[1] + 1) / 2,  # 帕尔贴: 0-1
            (action_np[2] + 1) * 50  # 阀门: 0-100%
        ])

        return real_action, log_prob, value_np

    def store_transition(self, state, action, reward, log_prob, value, done):
        """存储经验"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
        self.buffer['dones'].append(done)

    def compute_returns_and_advantages(self):
        """计算回报和优势函数"""
        rewards = self.buffer['rewards']
        values = self.buffer['values']
        dones = self.buffer['dones']

        returns = []
        advantages = []

        R = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                R = 0
            R = rewards[i] + self.gamma * R
            returns.insert(0, R)

            # GAE优势估计
            advantage = R - values[i]
            advantages.insert(0, advantage)

        return returns, advantages

    def update(self):
        """更新网络参数"""
        if len(self.buffer['states']) < self.batch_size:
            return

        self.training_step += 1

        # 计算回报和优势
        returns, advantages = self.compute_returns_and_advantages()

        # 转换为张量
        states = torch.FloatTensor(self.buffer['states']).to(self.device)
        actions = torch.FloatTensor(self.buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        advantages = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        for _ in range(self.update_epochs):
            # 评估动作
            log_probs, entropy = self.actor.evaluate_actions(states, actions)
            values = self.critic(states)

            # 计算比率
            ratio = torch.exp(log_probs - old_log_probs)

            # 裁剪的策略损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            # 值函数损失
            critic_loss = F.mse_loss(values, returns)

            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), CONFIG.train.GRAD_CLIP_NORM)
            self.actor_optimizer.step()

            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), CONFIG.train.GRAD_CLIP_NORM)
            self.critic_optimizer.step()

        # 记录损失
        self.losses['actor'].append(actor_loss.item())
        self.losses['critic'].append(critic_loss.item())
        self.entropies.append(entropy.mean().item())

        # 清空缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }

    def get_action_smoothness(self, actions: list) -> float:
        """计算动作平滑度"""
        if len(actions) < 2:
            return 0.0
        actions_array = np.array(actions)
        action_changes = np.diff(actions_array, axis=0)
        return np.mean(np.linalg.norm(action_changes, axis=1))

    def train_step(self, *args, **kwargs):
        """训练一步（update的别名，用于接口一致性）"""
        self.update()
        return 0.0, 0.0  # 返回空损失值以保持接口一致

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
            'entropies': self.entropies,
            'training_step': self.training_step
        }, filepath)

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', {'actor': [], 'critic': []})
        self.entropies = checkpoint.get('entropies', [])
        self.training_step = checkpoint.get('training_step', 0)

    def get_training_stats(self):
        """获取训练统计信息"""
        return {
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.losses['actor'],
            'critic_losses': self.losses['critic'],
            'entropies': self.entropies,
            'training_step': self.training_step
        }


if __name__ == "__main__":
    # 测试PPO
    print("=" * 70)
    print("Testing PPO Algorithm".center(70))
    print("=" * 70)

    state_dim = CONFIG.env.STATE_DIM
    action_dim = CONFIG.env.ACTION_DIM

    agent = PPO(state_dim, action_dim)
    print(f"\n✓ PPO initialized successfully!")
    print(f"  Device: {CONFIG.device}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")

    # 测试动作选择
    test_state = np.random.randn(state_dim)
    action, log_prob, value = agent.select_action(test_state)
    print(f"\n✓ Action selection works!")
    print(f"  Action: {action}")
    print(f"  Log prob: {log_prob:.4f}")
    print(f"  Value: {value:.4f}")

    print("\n" + "=" * 70)
    print("PPO test completed successfully!".center(70))
    print("=" * 70)
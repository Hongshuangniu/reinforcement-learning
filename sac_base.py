"""
10kV变压器智能冷却控制系统 - 基础SAC算法
Base Soft Actor-Critic Algorithm
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random

from config import CONFIG
from networks import BaseActorNetwork, BaseCriticNetwork

# 经验元组
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        """添加经验"""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """随机采样"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class BaseSAC:
    """
    基础SAC算法
    标准Soft Actor-Critic实现
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = CONFIG.device

        # 创建网络
        self.actor = BaseActorNetwork(
            state_dim, action_dim, CONFIG.train.HIDDEN_DIM
        ).to(self.device)

        self.critic = BaseCriticNetwork(
            state_dim, action_dim, CONFIG.train.HIDDEN_DIM
        ).to(self.device)

        self.critic_target = BaseCriticNetwork(
            state_dim, action_dim, CONFIG.train.HIDDEN_DIM
        ).to(self.device)

        # 复制参数到目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=CONFIG.train.LEARNING_RATE_ACTOR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=CONFIG.train.LEARNING_RATE_CRITIC
        )

        # 温度参数
        self.log_alpha = torch.tensor(
            np.log(CONFIG.train.INITIAL_ALPHA),
            requires_grad=True,
            device=self.device
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=CONFIG.train.LEARNING_RATE_ALPHA
        )
        self.target_entropy = -action_dim

        # 超参数
        self.gamma = CONFIG.train.GAMMA
        self.tau = CONFIG.train.TAU
        self.batch_size = CONFIG.train.BATCH_SIZE

        # 经验回放
        self.memory = ReplayBuffer(CONFIG.train.BUFFER_SIZE)

        # 统计
        self.episode_rewards = []
        self.losses = {'actor': [], 'critic': [], 'alpha': []}
        self.entropies = []
        self.training_step = 0
        self.last_action = None

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

        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state, deterministic=False)
                action = torch.tanh(mean)
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state)

        action = action.cpu().numpy().flatten()

        # 映射到实际动作空间
        real_action = np.array([
            action[0] * 1.5 + 3.5,  # 压力: 2-5 kPa
            (action[1] + 1) / 2,  # 帕尔贴: 0-1
            (action[2] + 1) * 50  # 阀门: 0-100%
        ])

        return real_action

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)

    def update(self):
        """更新网络参数"""
        if len(self.memory) < self.batch_size:
            return

        self.training_step += 1

        # 采样
        experiences = self.memory.sample(self.batch_size)

        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device).unsqueeze(1)

        # ========= 更新Critic =========
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

            alpha = self.log_alpha.exp()
            target_q = rewards + (1 - dones) * self.gamma * (q_next - alpha * next_log_probs)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), CONFIG.train.GRAD_CLIP_NORM)
        self.critic_optimizer.step()

        # ========= 更新Actor =========
        actions_new, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, actions_new)
        q_new = torch.min(q1_new, q2_new)

        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), CONFIG.train.GRAD_CLIP_NORM)
        self.actor_optimizer.step()

        # ========= 更新温度参数 =========
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # 记录损失
        self.losses['critic'].append(critic_loss.item())
        self.losses['actor'].append(actor_loss.item())
        self.losses['alpha'].append(alpha_loss.item())

        # 记录熵
        with torch.no_grad():
            entropy = -log_probs.mean().item()
            self.entropies.append(entropy)

    def get_action_smoothness(self, actions: list) -> float:
        """
        计算动作平滑度

        Args:
            actions: 动作序列

        Returns:
            平滑度指标(越小越平滑)
        """
        if len(actions) < 2:
            return 0.0

        actions_array = np.array(actions)
        action_changes = np.diff(actions_array, axis=0)
        smoothness = np.mean(np.linalg.norm(action_changes, axis=1))

        return smoothness

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
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
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', {'actor': [], 'critic': [], 'alpha': []})
        self.entropies = checkpoint.get('entropies', [])
        self.training_step = checkpoint.get('training_step', 0)

    def get_training_stats(self):
        """获取训练统计信息"""
        return {
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.losses['actor'],
            'critic_losses': self.losses['critic'],
            'alpha_losses': self.losses['alpha'],
            'entropies': self.entropies,
            'training_step': self.training_step
        }


if __name__ == "__main__":
    # 测试基础SAC
    print("=" * 70)
    print("Testing Base SAC Algorithm".center(70))
    print("=" * 70)

    state_dim = CONFIG.env.STATE_DIM
    action_dim = CONFIG.env.ACTION_DIM

    agent = BaseSAC(state_dim, action_dim)
    print(f"\n✓ Base SAC initialized successfully!")
    print(f"  Device: {CONFIG.device}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")

    # 测试动作选择
    test_state = np.random.randn(state_dim)
    action = agent.select_action(test_state)
    print(f"\n✓ Action selection works!")
    print(f"  Action shape: {action.shape}")
    print(f"  Action values: {action}")

    # 测试经验存储和更新
    for i in range(300):
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        agent.store_transition(state, action, reward, next_state, done)

    print(f"\n✓ Experience storage works!")
    print(f"  Buffer size: {len(agent.memory)}")

    # 测试更新
    agent.update()
    print(f"\n✓ Network update works!")
    print(f"  Critic loss: {agent.losses['critic'][-1]:.4f}")
    print(f"  Actor loss: {agent.losses['actor'][-1]:.4f}")
    print(f"  Alpha loss: {agent.losses['alpha'][-1]:.4f}")

    print("\n" + "=" * 70)
    print("Base SAC test completed successfully!".center(70))
    print("=" * 70)
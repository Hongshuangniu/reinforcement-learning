"""
10kV变压器智能冷却控制系统 - 神经网络架构
Neural Network Architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import CONFIG


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    用于改进SAC算法
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # 添加序列维度
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        seq_len = x.shape[1]

        # 计算Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 应用注意力
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim)

        # 输出投影和残差连接
        output = self.output_proj(attention_output)
        output = self.layer_norm(output + x)

        # 移除序列维度
        if output.shape[1] == 1:
            output = output.squeeze(1)

        return output


class ImprovedActorNetwork(nn.Module):
    """
    改进的Actor网络
    包含注意力机制、层归一化和残差连接
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ImprovedActorNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 注意力层
        self.attention = MultiHeadAttention(state_dim, hidden_dim,
                                            CONFIG.train.NUM_ATTENTION_HEADS)

        # 主网络
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # 输出层
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # 层归一化
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(CONFIG.train.DROPOUT_RATE)

    def forward(self, state):
        # 注意力处理
        x = self.attention(state)

        # 前向传播with残差连接
        h1 = F.relu(self.ln1(self.fc1(x)))
        h2 = F.relu(self.ln2(self.fc2(h1)))
        h2 = h2 + h1  # 残差连接
        h3 = F.relu(self.ln3(self.fc3(h2)))
        h3 = self.dropout(h3)

        # 输出均值和对数标准差
        mean = self.mean_layer(h3)
        log_std = self.log_std_layer(h3)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, state):
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化技巧
        action = torch.tanh(x_t)

        # 计算对数概率
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class ImprovedCriticNetwork(nn.Module):
    """
    改进的Critic网络
    双Q网络with注意力机制
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ImprovedCriticNetwork, self).__init__()

        input_dim = state_dim + action_dim

        # Q1网络
        self.q1_attention = MultiHeadAttention(input_dim, hidden_dim,
                                               CONFIG.train.NUM_ATTENTION_HEADS)
        self.q1_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)
        self.q1_ln1 = nn.LayerNorm(hidden_dim)
        self.q1_ln2 = nn.LayerNorm(hidden_dim)

        # Q2网络
        self.q2_attention = MultiHeadAttention(input_dim, hidden_dim,
                                               CONFIG.train.NUM_ATTENTION_HEADS)
        self.q2_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)
        self.q2_ln1 = nn.LayerNorm(hidden_dim)
        self.q2_ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, state, action):
        # 拼接状态和动作
        x = torch.cat([state, action], dim=1)

        # Q1
        q1 = self.q1_attention(x)
        q1 = F.relu(self.q1_ln1(self.q1_fc1(q1)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = self.q1_fc3(q1)

        # Q2
        q2 = self.q2_attention(x)
        q2 = F.relu(self.q2_ln1(self.q2_fc1(q2)))
        q2 = F.relu(self.q2_ln2(self.q2_fc2(q2)))
        q2 = self.q2_fc3(q2)

        return q1, q2


class BaseActorNetwork(nn.Module):
    """
    基础Actor网络
    用于传统SAC、DDPG、TD3
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(BaseActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # 对于SAC,输出均值和对数标准差
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # 对于DDPG/TD3,直接输出动作
        self.action_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, deterministic=False):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        if deterministic:
            # 用于DDPG/TD3
            action = torch.tanh(self.action_layer(x))
            return action
        else:
            # 用于SAC
            mean = self.mean_layer(x)
            log_std = self.log_std_layer(x)
            log_std = torch.clamp(log_std, min=-20, max=2)
            return mean, log_std

    def sample(self, state):
        """采样动作(用于SAC)"""
        mean, log_std = self.forward(state, deterministic=False)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class BaseCriticNetwork(nn.Module):
    """
    基础Critic网络
    双Q网络,用于SAC、TD3
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(BaseCriticNetwork, self).__init__()

        # Q1网络
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)

        # Q2网络
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        # Q1
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        # Q2
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return q1, q2

    def Q1(self, state, action):
        """只返回Q1的值"""
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        return q1


class PPOActorNetwork(nn.Module):
    """
    PPO Actor网络
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PPOActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean_layer(x))
        return mean

    def get_distribution(self, state):
        """获取动作分布"""
        mean = self.forward(state)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def evaluate_actions(self, state, action):
        """评估动作的对数概率和熵"""
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy


class PPOCriticNetwork(nn.Module):
    """
    PPO Critic网络(Value Network)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(PPOCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_layer(x)
        return value


def init_weights(m):
    """初始化网络权重"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # 测试网络
    state_dim = CONFIG.env.STATE_DIM
    action_dim = CONFIG.env.ACTION_DIM
    hidden_dim = CONFIG.train.HIDDEN_DIM

    print("=" * 70)
    print("Testing Neural Networks".center(70))
    print("=" * 70)

    # 测试改进Actor
    print("\n1. Testing Improved Actor Network...")
    improved_actor = ImprovedActorNetwork(state_dim, action_dim, hidden_dim)
    improved_actor.to(CONFIG.device)
    test_state = torch.randn(32, state_dim).to(CONFIG.device)
    mean, log_std = improved_actor(test_state)
    print(f"   Output shapes: mean={mean.shape}, log_std={log_std.shape}")
    action, log_prob = improved_actor.sample(test_state)
    print(f"   Sample shapes: action={action.shape}, log_prob={log_prob.shape}")
    print("   ✓ Improved Actor works!")

    # 测试改进Critic
    print("\n2. Testing Improved Critic Network...")
    improved_critic = ImprovedCriticNetwork(state_dim, action_dim, hidden_dim)
    improved_critic.to(CONFIG.device)
    test_action = torch.randn(32, action_dim).to(CONFIG.device)
    q1, q2 = improved_critic(test_state, test_action)
    print(f"   Output shapes: q1={q1.shape}, q2={q2.shape}")
    print("   ✓ Improved Critic works!")

    # 测试基础Actor
    print("\n3. Testing Base Actor Network...")
    base_actor = BaseActorNetwork(state_dim, action_dim, hidden_dim)
    base_actor.to(CONFIG.device)
    action = base_actor(test_state, deterministic=True)
    print(f"   Deterministic action shape: {action.shape}")
    mean, log_std = base_actor(test_state, deterministic=False)
    print(f"   Stochastic output shapes: mean={mean.shape}, log_std={log_std.shape}")
    print("   ✓ Base Actor works!")

    # 测试基础Critic
    print("\n4. Testing Base Critic Network...")
    base_critic = BaseCriticNetwork(state_dim, action_dim, hidden_dim)
    base_critic.to(CONFIG.device)
    q1, q2 = base_critic(test_state, test_action)
    print(f"   Output shapes: q1={q1.shape}, q2={q2.shape}")
    print("   ✓ Base Critic works!")

    # 测试PPO网络
    print("\n5. Testing PPO Networks...")
    ppo_actor = PPOActorNetwork(state_dim, action_dim, hidden_dim)
    ppo_critic = PPOCriticNetwork(state_dim, hidden_dim)
    ppo_actor.to(CONFIG.device)
    ppo_critic.to(CONFIG.device)
    action = ppo_actor(test_state)
    value = ppo_critic(test_state)
    print(f"   Actor output shape: {action.shape}")
    print(f"   Critic output shape: {value.shape}")
    log_prob, entropy = ppo_actor.evaluate_actions(test_state, action)
    print(f"   Log prob shape: {log_prob.shape}, Entropy shape: {entropy.shape}")
    print("   ✓ PPO networks work!")

    print("\n" + "=" * 70)
    print("All networks passed the tests!".center(70))
    print("=" * 70)
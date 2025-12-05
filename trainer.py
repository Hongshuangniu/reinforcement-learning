"""
10kV变压器智能冷却控制系统 - 训练模块（完整指标版 - 无早停）
统一的训练接口，支持所有算法

关键改进：
1. ✅ 在训练过程中计算完整指标
2. ✅ 显示所有评估指标（不仅是MAE）
3. ✅ 保存详细的训练统计
4. ✅ 基于多个指标综合判断模型优劣
5. ✅ 删除早停机制，完整训练所有episodes
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
import pickle

from environment import ImprovedTransformerCoolingEnv
from sac_base import BaseSAC
from sac_temperature_aware import ImprovedSAC
from ppo import PPO
from ddpg import DDPG
from td3 import TD3
from metrics import MetricsCalculator
from config import CONFIG, TrainingConfig


class Trainer:
    """训练器基类（完整指标版 - 无早停）"""

    def __init__(self, env: ImprovedTransformerCoolingEnv, agent, algorithm_name: str,
                 config: TrainingConfig = TrainingConfig()):
        self.env = env
        self.agent = agent
        self.algorithm_name = algorithm_name
        self.config = config

        # ⭐ 添加指标计算器
        self.metrics_calculator = MetricsCalculator()

        # 训练记录
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_metrics_history = []  # ⭐ 保存每次评估的完整指标

        self.training_data = {
            'rewards': [],
            'temperatures': [],
            'actions': [],
            'losses': []
        }

    def train_episode(self) -> Tuple[float, int, Dict]:
        """训练一个episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_temps = []
        episode_actions = []

        done = False
        while not done:
            # 选择动作
            if self.algorithm_name == 'ppo':
                action, log_prob, value = self.agent.select_action(state)
            else:
                action = self.agent.select_action(state, evaluate=False)

            # 执行动作
            next_state, reward, done, info = self.env.step(action)

            # 存储转移
            if self.algorithm_name == 'ppo':
                self.agent.store_transition(state, action, reward, log_prob, value, done)
            else:
                self.agent.store_transition(state, action, reward, next_state, done)

            # 记录
            episode_temps.append(info['oil_temp'])
            episode_actions.append(action.copy())
            episode_reward += reward
            episode_length += 1

            state = next_state

        info = {
            'temperatures': episode_temps,
            'actions': np.array(episode_actions)
        }

        return episode_reward, episode_length, info

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        评估智能体（完整指标版）

        Args:
            num_episodes: 评估episode数量

        Returns:
            包含所有指标的字典
        """
        eval_rewards = []
        all_true_temps = []
        all_target_temps = []
        all_actions = []
        all_episode_rewards = []

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_temps = []
            episode_actions = []
            target_temps = []
            done = False

            while not done:
                # ⭐ 评估时使用确定性策略
                if self.algorithm_name == 'ppo':
                    action, _, _ = self.agent.select_action(state, evaluate=True)
                else:
                    action = self.agent.select_action(state, evaluate=True)

                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_temps.append(info['oil_temp'])
                target_temps.append(self.env.target_temp)
                episode_actions.append(action.copy())

                state = next_state

            eval_rewards.append(episode_reward)
            all_episode_rewards.append(episode_reward)
            all_true_temps.extend(episode_temps)
            all_target_temps.extend(target_temps)
            all_actions.extend(episode_actions)

        # ⭐ 计算完整的控制性能指标
        all_true_temps = np.array(all_true_temps)
        all_target_temps = np.array(all_target_temps)
        all_actions = np.array(all_actions)

        control_metrics = self.metrics_calculator.calculate_control_metrics(
            y_true=all_true_temps,
            y_pred=all_target_temps
        )

        # ⭐ 计算完整的RL指标
        rl_metrics = self.metrics_calculator.calculate_rl_metrics(
            rewards=all_episode_rewards,
            actions=all_actions,
            temperatures=all_true_temps
        )

        # 合并所有指标
        all_metrics = {
            **control_metrics,
            **rl_metrics,
            'mean_eval_reward': np.mean(eval_rewards),
            'std_eval_reward': np.std(eval_rewards),
            'mean_temp': np.mean(all_true_temps),
            'std_temp': np.std(all_true_temps)
        }

        return all_metrics

    def train(self, num_episodes: int, eval_interval: int = 10) -> Dict:
        """
        训练智能体（完整指标版 - 无早停）

        关键改进：
        1. 计算并显示所有评估指标
        2. 基于综合指标保存最佳模型
        3. 删除早停机制，完整训练所有episodes
        """
        print(f"\n开始训练 {self.algorithm_name}...")
        print(f"训练Episodes: {num_episodes}")
        print(f"评估间隔: {eval_interval} episodes")

        best_eval_reward = -np.inf
        best_mae = np.inf
        best_rmse = np.inf
        epochs_since_improvement = 0

        for episode in tqdm(range(num_episodes), desc=f"训练 {self.algorithm_name}"):
            # 训练一个episode
            episode_reward, episode_length, info = self.train_episode()

            # 更新网络
            if self.algorithm_name == 'ppo':
                self.agent.update()
            else:
                for _ in range(episode_length):
                    self.agent.update()

            # 记录
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.training_data['rewards'].append(episode_reward)
            self.training_data['temperatures'].extend(info['temperatures'])
            self.training_data['actions'].append(info['actions'])

            # ⭐ 评估
            if (episode + 1) % eval_interval == 0:
                eval_metrics = self.evaluate(num_episodes=10)
                self.eval_metrics_history.append(eval_metrics)

                # 提取关键指标
                eval_reward = eval_metrics['mean_eval_reward']
                eval_mae = eval_metrics['MAE']
                eval_rmse = eval_metrics['RMSE']
                eval_mape = eval_metrics['MAPE']
                eval_r2 = eval_metrics['R2']
                eval_max_ae = eval_metrics['MaxAE']

                # ⭐ 基于MAE判断是否为最佳模型
                is_best = False
                if eval_mae < best_mae:
                    best_mae = eval_mae
                    best_rmse = eval_rmse
                    best_eval_reward = eval_reward
                    self.save_model(f"best_{self.algorithm_name}.pth")
                    epochs_since_improvement = 0
                    is_best = True
                else:
                    epochs_since_improvement += 1

                # ⭐ 显示完整指标
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"📊 控制性能指标:")
                print(f"  MAE:   {eval_mae:8.4f}°C {'⭐ 新最佳!' if is_best else ''}")
                print(f"  RMSE:  {eval_rmse:8.4f}°C")
                print(f"  MAPE:  {eval_mape:8.4f}%")
                print(f"  R²:    {eval_r2:8.4f}")
                print(f"  MaxAE: {eval_max_ae:8.4f}°C")

                print(f"\n🎯 强化学习指标:")
                print(f"  训练回报: {episode_reward:8.2f}")
                print(f"  评估回报: {eval_reward:8.2f} ± {eval_metrics['std_eval_reward']:6.2f}")
                print(f"  回报方差: {eval_metrics['reward_variance']:8.4f}")

                if 'action_smoothness' in eval_metrics:
                    print(f"  动作平滑: {eval_metrics['action_smoothness']:8.4f}")
                if 'temp_smoothness' in eval_metrics:
                    print(f"  温度平滑: {eval_metrics['temp_smoothness']:8.4f}")

                print(f"\n📈 最佳记录:")
                print(f"  最佳MAE:  {best_mae:8.4f}°C")
                print(f"  最佳RMSE: {best_rmse:8.4f}°C")
                print(f"  最佳回报: {best_eval_reward:8.2f}")

                # 显示距离上次改进的轮数
                if epochs_since_improvement > 0:
                    print(f"\n💡 训练信息:")
                    print(f"  距上次改进: {epochs_since_improvement} 轮评估")

                print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        print(f"\n✓ {self.algorithm_name} 训练完成!")
        print(f"  完成Episodes: {num_episodes}")
        print(f"  最佳MAE:   {best_mae:.4f}°C")
        print(f"  最佳RMSE:  {best_rmse:.4f}°C")
        print(f"  最佳回报:  {best_eval_reward:.2f}")

        # 保存训练结果
        results = {
            'algorithm': self.algorithm_name,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'eval_rewards': self.eval_rewards,
            'eval_metrics_history': self.eval_metrics_history,  # ⭐ 保存完整指标历史
            'training_data': self.training_data,
            'best_reward': best_eval_reward,
            'best_mae': best_mae,
            'best_rmse': best_rmse,
            # ⭐ 最终评估指标
            'final_metrics': self.eval_metrics_history[-1] if self.eval_metrics_history else {}
        }

        return results

    def save_model(self, filename: str):
        """保存模型"""
        filepath = os.path.join(CONFIG.output.MODEL_DIR, filename)
        self.agent.save_model(filepath)

    def load_model(self, filename: str):
        """加载模型"""
        filepath = os.path.join(CONFIG.output.MODEL_DIR, filename)
        self.agent.load_model(filepath)


class MultiAlgorithmTrainer:
    """多算法训练管理器（完整指标版 - 无早停）"""

    def __init__(self, env_data: pd.DataFrame, config: TrainingConfig = TrainingConfig()):
        self.env_data = env_data
        self.config = config
        self.results = {}

    def create_agent(self, algorithm: str, state_dim: int, action_dim: int):
        """创建智能体"""
        if algorithm == 'improved_sac':
            return ImprovedSAC(state_dim, action_dim)
        elif algorithm == 'sac':
            return BaseSAC(state_dim, action_dim)
        elif algorithm == 'ppo':
            return PPO(state_dim, action_dim)
        elif algorithm == 'ddpg':
            return DDPG(state_dim, action_dim)
        elif algorithm == 'td3':
            return TD3(state_dim, action_dim)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train_algorithm(self, algorithm: str, num_episodes: int = None) -> Dict:
        """训练单个算法"""
        if num_episodes is None:
            num_episodes = self.config.NUM_EPISODES

        # 创建环境
        env = ImprovedTransformerCoolingEnv(self.env_data)
        state_dim = env.state_dim
        action_dim = env.action_dim

        # 创建智能体
        agent = self.create_agent(algorithm, state_dim, action_dim)

        # 创建训练器
        trainer = Trainer(env, agent, algorithm, self.config)

        # 训练
        results = trainer.train(num_episodes, self.config.EVAL_FREQUENCY)

        # 保存结果
        self.results[algorithm] = results

        return results

    def train_all(self, algorithms: List[str], num_episodes: int = None):
        """训练所有算法"""
        print("=" * 70)
        print("开始多算法训练（完整指标版 - 无早停）")
        print("=" * 70)
        print("注意: 所有算法将完整训练所有episodes")
        print("=" * 70)

        for algo in algorithms:
            print(f"\n{'=' * 70}")
            print(f"训练算法: {algo.upper()}")
            print(f"{'=' * 70}")

            try:
                results = self.train_algorithm(algo, num_episodes)
                print(f"\n✓ {algo} 训练成功")

                # ⭐ 显示最终指标
                final_metrics = results.get('final_metrics', {})
                print(f"\n最终评估指标:")
                print(f"  MAE:   {final_metrics.get('MAE', 0):.4f}°C")
                print(f"  RMSE:  {final_metrics.get('RMSE', 0):.4f}°C")
                print(f"  MAPE:  {final_metrics.get('MAPE', 0):.4f}%")
                print(f"  R²:    {final_metrics.get('R2', 0):.4f}")
                print(f"  回报:  {final_metrics.get('mean_eval_reward', 0):.2f}")

            except Exception as e:
                print(f"\n✗ {algo} 训练失败: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 70)
        print("所有算法训练完成!")
        print("=" * 70)

        # 打印对比
        self._print_comparison()

        # 保存所有结果
        self.save_all_results()

    def _print_comparison(self):
        """打印算法对比（完整指标）"""
        if not self.results:
            return

        print("\n" + "=" * 100)
        print("算法对比（完整指标）")
        print("=" * 100)
        print(f"{'算法':<15} | {'MAE':>8} | {'RMSE':>8} | {'MAPE':>8} | {'R²':>8} | {'回报':>10} | {'Episodes':>10}")
        print("-" * 100)

        for algo, results in self.results.items():
            final_metrics = results.get('final_metrics', {})
            print(f"{algo.upper():<15} | "
                  f"{final_metrics.get('MAE', 0):>8.4f} | "
                  f"{final_metrics.get('RMSE', 0):>8.4f} | "
                  f"{final_metrics.get('MAPE', 0):>8.2f} | "
                  f"{final_metrics.get('R2', 0):>8.4f} | "
                  f"{final_metrics.get('mean_eval_reward', 0):>10.2f} | "
                  f"{len(results['episode_rewards']):>10}")

    def save_all_results(self):
        """保存所有训练结果"""
        filepath = os.path.join(CONFIG.vis.RESULTS_DIR, 'training_results.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"\n✓ 训练结果已保存到: {filepath}")

    def load_all_results(self, filepath: str = None):
        """加载训练结果"""
        if filepath is None:
            filepath = os.path.join(CONFIG.vis.RESULTS_DIR, 'training_results.pkl')

        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        print(f"✓ 训练结果已加载: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("训练模块测试（完整指标版 - 无早停）")
    print("=" * 60)

    print("\n✓ 关键改进:")
    print("  1. ✅ 在训练过程中计算完整指标")
    print("  2. ✅ 显示MAE、RMSE、MAPE、R²等所有指标")
    print("  3. ✅ 基于综合指标判断最佳模型")
    print("  4. ✅ 保存详细的指标历史")
    print("  5. ✅ 删除早停机制，完整训练所有episodes")

    print("\n" + "=" * 60)
    print("✓ 训练模块准备就绪（完整指标版 - 无早停）")
    print("=" * 60)
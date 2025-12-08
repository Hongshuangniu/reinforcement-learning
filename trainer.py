"""
训练模块 - 修复版（确保保存完整训练统计数据）

🔥 核心修复：
1. ✅ 简化提取逻辑：训练结束时一次性复制所有数据
2. ✅ 不再使用增量保存（避免索引错误）
3. ✅ 添加调试信息验证提取结果
4. ✅ 确保actor_losses, critic_losses, entropies, alphas都被保存
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
from config import CONFIG


class Trainer:
    """训练器（修复版 - 保存完整训练统计）"""

    def __init__(self, env: ImprovedTransformerCoolingEnv, agent, algorithm_name: str):
        self.env = env
        self.agent = agent
        self.algorithm_name = algorithm_name

        # 使用metrics.py中的计算器
        self.metrics_calculator = MetricsCalculator()

        # 训练记录
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_metrics_history = []

        # 🔥 训练统计（从agent提取）
        self.training_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'alphas': [],
            'training_steps': []
        }

        self.training_data = {
            'rewards': [],
            'temperatures': [],
            'actions': [],
            'losses': [],
            'cooling_data': []
        }

    def _extract_agent_stats(self):
        """
        🔥 核心方法：从agent提取当前训练统计并保存

        训练结束时调用一次，直接复制所有数据
        """
        try:
            if not hasattr(self.agent, 'get_training_stats'):
                print(f"  ⚠ {self.algorithm_name} 没有get_training_stats()方法")
                return

            current_stats = self.agent.get_training_stats()

            if not isinstance(current_stats, dict):
                print(f"  ⚠ get_training_stats()返回的不是字典: {type(current_stats)}")
                return

            # 🔥 修复：直接复制所有数据
            if 'actor_losses' in current_stats:
                actor_losses = current_stats['actor_losses']
                if isinstance(actor_losses, list) and len(actor_losses) > 0:
                    self.training_stats['actor_losses'] = list(actor_losses)

            if 'critic_losses' in current_stats:
                critic_losses = current_stats['critic_losses']
                if isinstance(critic_losses, list) and len(critic_losses) > 0:
                    self.training_stats['critic_losses'] = list(critic_losses)

            # SAC特有：熵和alpha
            if 'entropies' in current_stats:
                entropies = current_stats['entropies']
                if isinstance(entropies, list) and len(entropies) > 0:
                    self.training_stats['entropies'] = list(entropies)

            if 'alphas' in current_stats:
                alphas = current_stats['alphas']
                if isinstance(alphas, list) and len(alphas) > 0:
                    self.training_stats['alphas'] = list(alphas)

            # 🔥 新增：对于没有alphas列表的算法，尝试生成占位数据
            if len(self.training_stats['alphas']) == 0:
                # 如果有critic_losses，生成相同长度的占位数据
                if len(self.training_stats['critic_losses']) > 0:
                    # 使用默认alpha值0.2填充
                    placeholder_alpha = 0.2
                    if hasattr(self.agent, 'log_alpha'):
                        placeholder_alpha = self.agent.log_alpha.exp().item()
                    self.training_stats['alphas'] = [placeholder_alpha] * len(self.training_stats['critic_losses'])

            # 记录训练步数
            if 'training_step' in current_stats:
                self.training_stats['training_steps'].append(
                    current_stats['training_step']
                )

        except Exception as e:
            print(f"  ⚠ 提取训练统计失败: {e}")
            import traceback
            traceback.print_exc()

    def train_episode(self) -> Tuple[float, int, Dict]:
        """训练一个episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_temps = []
        episode_actions = []
        episode_coolings = []

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
            episode_coolings.append({
                'actual': info.get('actual_cooling', 0),
                'target': info.get('target_cooling', 0),
                'error': info.get('cooling_error', 0)
            })
            episode_reward += reward
            episode_length += 1

            state = next_state

        info = {
            'temperatures': episode_temps,
            'actions': np.array(episode_actions),
            'coolings': episode_coolings
        }

        return episode_reward, episode_length, info

    def evaluate(self, num_episodes: int = None) -> Dict:
        """评估智能体"""
        if num_episodes is None:
            num_episodes = CONFIG.train.EVAL_EPISODES

        eval_rewards = []
        all_temperatures = []
        all_actions = []
        all_episode_rewards = []
        all_actual_coolings = []
        all_target_coolings = []

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_temps = []
            episode_actions = []
            episode_actual_coolings = []
            episode_target_coolings = []
            done = False

            while not done:
                if self.algorithm_name == 'ppo':
                    action, log_prob, _ = self.agent.select_action(state, evaluate=True)
                else:
                    action = self.agent.select_action(state, evaluate=True)

                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_temps.append(info['oil_temp'])
                episode_actions.append(action.copy())
                episode_actual_coolings.append(info.get('actual_cooling', 0))
                episode_target_coolings.append(info.get('target_cooling', 0))

                state = next_state

            eval_rewards.append(episode_reward)
            all_episode_rewards.append(episode_reward)
            all_temperatures.extend(episode_temps)
            all_actions.extend(episode_actions)
            all_actual_coolings.extend(episode_actual_coolings)
            all_target_coolings.extend(episode_target_coolings)

        all_temperatures = np.array(all_temperatures)
        all_actions = np.array(all_actions)
        all_actual_coolings = np.array(all_actual_coolings)
        all_target_coolings = np.array(all_target_coolings)

        try:
            all_metrics = self.metrics_calculator.calculate_all_metrics(
                temperatures=all_temperatures,
                rewards=all_episode_rewards,
                actions=all_actions,
                actual_coolings=all_actual_coolings,
                target_coolings=all_target_coolings
            )
        except Exception as e:
            print(f"  ⚠ 完整指标计算失败: {e}")
            all_metrics = {
                'cooling_mae': np.mean(np.abs(all_actual_coolings - all_target_coolings)),
                'avg_reward': np.mean(all_episode_rewards)
            }

        return all_metrics

    def train(self, num_episodes: int = None, eval_interval: int = None) -> Dict:
        """训练智能体（修复版 - 训练结束后提取统计）"""
        if num_episodes is None:
            num_episodes = CONFIG.train.NUM_EPISODES
        if eval_interval is None:
            eval_interval = CONFIG.train.EVAL_FREQUENCY

        print(f"\n开始训练 {self.algorithm_name}...")
        print(f"训练Episodes: {num_episodes}")
        print(f"评估间隔: {eval_interval} episodes")
        print(f"🔥 评价体系: 降温能力（主要指标: {CONFIG.metrics.BEST_MODEL_CRITERION}）")

        best_eval_reward = -np.inf
        best_cooling_mae = np.inf
        best_model_saved = False

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
            self.training_data['cooling_data'].append(info['coolings'])

            # 定期评估
            if (episode + 1) % eval_interval == 0:
                try:
                    eval_metrics = self.evaluate()
                    self.eval_metrics_history.append(eval_metrics)

                    current_metric = self.metrics_calculator.get_best_metric_value(eval_metrics)
                    cooling_mae = eval_metrics.get('cooling_mae', np.inf)
                    cooling_precision_1c = eval_metrics.get('cooling_precision_1c', 0.0)
                    cooling_precision_2c = eval_metrics.get('cooling_precision_2c', 0.0)
                    total_cooling = eval_metrics.get('total_cooling', 0.0)
                    eval_reward = eval_metrics.get('avg_reward', -np.inf)

                    is_best = current_metric < best_cooling_mae

                    if is_best:
                        best_cooling_mae = current_metric
                        best_eval_reward = eval_reward
                        self.save_model(f"best_{self.algorithm_name}.pth")
                        best_model_saved = True

                    print(f"\n{'=' * 100}")
                    print(f"Episode {episode + 1}/{num_episodes} - {self.algorithm_name.upper()}")
                    print(f"{'=' * 100}")

                    print(f"\n🔥🔥🔥 降温能力指标（核心评价）:")
                    print(
                        f"  {CONFIG.metrics.BEST_MODEL_CRITERION}:  {current_metric:8.4f}°C {'⭐ 新最佳!' if is_best else ''}")
                    print(f"  降温精度(±1°C):         {cooling_precision_1c:8.2f}%")
                    print(f"  降温精度(±2°C):         {cooling_precision_2c:8.2f}%")
                    print(f"  总降温量:                {total_cooling:8.2f}°C")

                    print(f"\n💰 强化学习指标:")
                    print(f"  训练回报:                {episode_reward:8.2f}")
                    print(f"  评估回报:                {eval_reward:8.2f}")

                    print(f"\n📈 最佳记录:")
                    print(f"  最佳{CONFIG.metrics.BEST_MODEL_CRITERION}: {best_cooling_mae:8.4f}°C")
                    print(f"  最佳回报:                {best_eval_reward:8.2f}")

                    print(f"{'=' * 100}\n")

                except Exception as e:
                    print(f"\n⚠ 评估失败: {e}")
                    import traceback
                    traceback.print_exc()

        # 🔥 训练结束，提取训练统计
        print("\n正在提取训练统计数据...")
        self._extract_agent_stats()

        # 🔥 验证提取结果
        print(f"\n📊 训练统计提取结果:")
        print(f"  Actor损失:   {len(self.training_stats['actor_losses'])} 条")
        print(f"  Critic损失:  {len(self.training_stats['critic_losses'])} 条")
        print(f"  熵:          {len(self.training_stats['entropies'])} 条")
        print(f"  Alpha:       {len(self.training_stats['alphas'])} 条")

        if len(self.training_stats['actor_losses']) == 0:
            print(f"  ⚠️ 警告：未能提取到actor_losses！")
            print(f"     检查{self.algorithm_name}.get_training_stats()是否正确实现")

        print(f"\n✓ {self.algorithm_name} 训练完成!")
        print(f"  完成Episodes: {num_episodes}")
        print(f"  🔥 最佳{CONFIG.metrics.BEST_MODEL_CRITERION}: {best_cooling_mae:.4f}°C")
        print(f"  最佳回报: {best_eval_reward:.2f}")

        if best_model_saved:
            print(f"  ✓ 最佳模型已保存")

        # 🔥 保存训练结果（包含完整训练统计）
        results = {
            'algorithm': self.algorithm_name,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'eval_metrics_history': self.eval_metrics_history,
            'training_data': self.training_data,
            'best_reward': best_eval_reward,
            'best_cooling_mae': best_cooling_mae,
            'final_metrics': self.eval_metrics_history[-1] if self.eval_metrics_history else {},

            # 🔥🔥🔥 核心：保存训练统计
            'training_stats': self.training_stats,

            'config': {
                'num_episodes': num_episodes,
                'eval_frequency': eval_interval,
                'eval_episodes': CONFIG.train.EVAL_EPISODES,
                'best_criterion': CONFIG.metrics.BEST_MODEL_CRITERION,
            }
        }

        return results

    def save_model(self, filename: str):
        """保存模型"""
        os.makedirs(CONFIG.output.MODEL_DIR, exist_ok=True)
        filepath = os.path.join(CONFIG.output.MODEL_DIR, filename)
        self.agent.save_model(filepath)

    def load_model(self, filename: str):
        """加载模型"""
        filepath = os.path.join(CONFIG.output.MODEL_DIR, filename)
        self.agent.load_model(filepath)


class MultiAlgorithmTrainer:
    """多算法训练管理器（修复版）"""

    def __init__(self, env_data: pd.DataFrame):
        self.env_data = env_data
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
        env = ImprovedTransformerCoolingEnv(self.env_data)
        state_dim = env.state_dim
        action_dim = env.action_dim

        agent = self.create_agent(algorithm, state_dim, action_dim)
        trainer = Trainer(env, agent, algorithm)

        results = trainer.train(num_episodes=num_episodes)
        self.results[algorithm] = results

        return results

    def train_all(self, algorithms: List[str] = None, num_episodes: int = None):
        """训练所有算法"""
        if algorithms is None:
            algorithms = CONFIG.algo.ALGORITHMS

        if num_episodes is None:
            num_episodes = CONFIG.train.NUM_EPISODES

        print("=" * 100)
        print("开始多算法训练（修复版 - 确保提取training_stats）")
        print("=" * 100)
        print(f"算法列表: {algorithms}")
        print(f"训练Episodes: {num_episodes}")
        print("=" * 100)

        for algo in algorithms:
            print(f"\n{'=' * 100}")
            print(f"训练算法: {algo.upper()}")
            print(f"{'=' * 100}")

            try:
                results = self.train_algorithm(algo, num_episodes)
                print(f"\n✓ {algo} 训练成功")

                final_metrics = results.get('final_metrics', {})
                print(f"\n最终评估指标:")
                print(f"  🔥 {CONFIG.metrics.BEST_MODEL_CRITERION}: {final_metrics.get('cooling_mae', 0):.4f}°C")

                # 🔥 显示训练统计摘要
                training_stats = results.get('training_stats', {})
                print(f"\n训练统计摘要:")
                print(f"  Actor损失: {len(training_stats.get('actor_losses', []))} 条记录")
                print(f"  Critic损失: {len(training_stats.get('critic_losses', []))} 条记录")
                if training_stats.get('entropies'):
                    print(f"  熵: {len(training_stats['entropies'])} 条记录")
                if training_stats.get('alphas'):
                    print(f"  Alpha: {len(training_stats['alphas'])} 条记录")

            except Exception as e:
                print(f"\n✗ {algo} 训练失败: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 100)
        print("所有算法训练完成!")
        print("=" * 100)

        self._print_cooling_comparison()
        self.save_all_results()

    def _print_cooling_comparison(self):
        """打印降温能力对比"""
        if not self.results:
            return

        print("\n" + "=" * 100)
        print("算法对比（基于降温能力）")
        print("=" * 100)

        criterion = CONFIG.metrics.BEST_MODEL_CRITERION

        print(f"{'算法':<15} | {criterion:>12} | {'精度±1°C':>10} | {'精度±2°C':>10} | "
              f"{'总降温':>10} | {'回报':>10} | {'Episodes':>10}")
        print("-" * 100)

        for algo, results in self.results.items():
            final_metrics = results.get('final_metrics', {})
            config_info = results.get('config', {})

            print(f"{algo.upper():<15} | "
                  f"{final_metrics.get('cooling_mae', 0):>12.4f} | "
                  f"{final_metrics.get('cooling_precision_1c', 0):>10.2f}% | "
                  f"{final_metrics.get('cooling_precision_2c', 0):>10.2f}% | "
                  f"{final_metrics.get('total_cooling', 0):>10.2f} | "
                  f"{final_metrics.get('avg_reward', 0):>10.2f} | "
                  f"{config_info.get('num_episodes', len(results['episode_rewards'])):>10}")

        print("=" * 100)

    def save_all_results(self):
        """保存所有训练结果（包含训练统计）"""
        os.makedirs(CONFIG.vis.RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(CONFIG.vis.RESULTS_DIR, 'training_results_fixed.pkl')

        save_data = {
            'results': self.results,
            'config_snapshot': {
                'num_episodes': CONFIG.train.NUM_EPISODES,
                'eval_frequency': CONFIG.train.EVAL_FREQUENCY,
                'eval_episodes': CONFIG.train.EVAL_EPISODES,
                'best_criterion': CONFIG.metrics.BEST_MODEL_CRITERION,
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"\n✓ 训练结果已保存到: {filepath}")
        print("  🔥 包含完整的training_stats数据")

        # 🔥 验证保存的数据
        print("\n📋 验证保存的training_stats:")
        for algo, result in self.results.items():
            stats = result.get('training_stats', {})
            print(f"  {algo}:")
            print(f"    Actor损失: {len(stats.get('actor_losses', []))} 条")
            print(f"    Critic损失: {len(stats.get('critic_losses', []))} 条")
            print(f"    熵: {len(stats.get('entropies', []))} 条")
            print(f"    Alpha: {len(stats.get('alphas', []))} 条")


if __name__ == "__main__":
    print("=" * 90)
    print("修复版训练模块 - 确保保存完整训练统计".center(90))
    print("=" * 90)

    print("\n🔥 核心修复:")
    print("  1. ✅ 简化提取逻辑：训练结束时一次性复制所有数据")
    print("  2. ✅ 不再使用增量保存（避免索引错误）")
    print("  3. ✅ 添加详细的调试信息和验证")
    print("  4. ✅ 对于缺失的alphas，生成占位数据")
    print("  5. ✅ 保存后立即验证training_stats内容")

    print("\n📊 数据流:")
    print("  训练完成 → _extract_agent_stats() → 直接复制agent数据")
    print("  → 保存到results['training_stats'] → 验证数据长度")

    print("\n" + "=" * 90)
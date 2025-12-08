"""
评估模块 - 完全使用CONFIG参数（降温能力评价）

核心改进：
1. ✅ 所有参数从CONFIG读取
2. ✅ 完全基于降温能力评价
3. ✅ 移除固定温度依赖
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import os
import pickle

from environment import ImprovedTransformerCoolingEnv
from config import CONFIG
from metrics import MetricsCalculator


class Evaluator:
    """评估器（完全使用CONFIG）"""

    def __init__(self, env: ImprovedTransformerCoolingEnv, agent, algorithm_name: str):
        self.env = env
        self.agent = agent
        self.algorithm_name = algorithm_name

        # ⭐ 使用metrics.py中的计算器（不需要target_temp）
        self.metrics_calc = MetricsCalculator()

    def evaluate_episode(self, deterministic: bool = True) -> Dict:
        """
        评估一个episode（使用完整的降温能力评价体系）

        Args:
            deterministic: 是否使用确定性策略

        Returns:
            包含完整指标的字典
        """
        state = self.env.reset()

        temperatures = []  # 实际温度序列
        rewards = []
        actions = []
        # 🔥 降温数据
        actual_coolings = []
        target_coolings = []

        done = False
        step = 0

        while not done:
            # 选择动作
            if self.algorithm_name == 'ppo':
                action, _, _ = self.agent.select_action(state, evaluate=deterministic)
            else:
                action = self.agent.select_action(state, evaluate=deterministic)

            # 执行动作
            next_state, reward, done, info = self.env.step(action)

            # 收集数据
            temperatures.append(info['oil_temp'])
            rewards.append(reward)
            actions.append(action.copy())
            # 🔥 收集降温数据
            actual_coolings.append(info.get('actual_cooling', 0))
            target_coolings.append(info.get('target_cooling', 0))

            state = next_state
            step += 1

        # 转换为numpy数组
        temperatures = np.array(temperatures)
        actions = np.array(actions)
        actual_coolings = np.array(actual_coolings)
        target_coolings = np.array(target_coolings)

        # ⭐⭐⭐ 核心：使用metrics.py计算所有指标 ⭐⭐⭐
        all_metrics = self.metrics_calc.calculate_all_metrics(
            temperatures=temperatures,
            rewards=rewards,
            actions=actions,
            actual_coolings=actual_coolings,  # 🔥 传入降温数据
            target_coolings=target_coolings  # 🔥 传入降温数据
        )

        return {
            'temperatures': temperatures,
            'rewards': rewards,
            'actions': actions,
            'actual_coolings': actual_coolings,
            'target_coolings': target_coolings,
            'metrics': all_metrics,
            'total_reward': sum(rewards),
            'avg_temp': np.mean(temperatures),
            'max_temp': np.max(temperatures),
            'min_temp': np.min(temperatures),
            'steps': step,
        }

    def evaluate_multiple_episodes(self, num_episodes: int = None, verbose: bool = True) -> Dict:
        """
        评估多个episodes（使用CONFIG参数）

        Args:
            num_episodes: 评估episode数量（None则从CONFIG读取）
            verbose: 是否打印进度

        Returns:
            汇总结果
        """
        # 🔥 从CONFIG读取
        if num_episodes is None:
            num_episodes = CONFIG.train.EVAL_EPISODES

        all_episodes = []
        all_metrics = []

        for i in range(num_episodes):
            episode_data = self.evaluate_episode()
            all_episodes.append(episode_data)
            all_metrics.append(episode_data['metrics'])

            if verbose and (i + 1) % 5 == 0:
                print(f"  评估进度: {i + 1}/{num_episodes} episodes")

        # 计算所有指标的平均值和标准差
        metrics_summary = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                metrics_summary[key] = np.mean(values)
                metrics_summary[f'{key}_std'] = np.std(values)

        # 汇总统计
        summary = {
            'avg_reward': np.mean([ep['total_reward'] for ep in all_episodes]),
            'std_reward': np.std([ep['total_reward'] for ep in all_episodes]),
            'avg_temp': np.mean([ep['avg_temp'] for ep in all_episodes]),
            'max_temp': np.max([ep['max_temp'] for ep in all_episodes]),
            'min_temp': np.min([ep['min_temp'] for ep in all_episodes]),
            'episodes': all_episodes,
            'metrics': metrics_summary,
            'num_eval_episodes': num_episodes  # 🔥 记录评估episode数
        }

        return summary


class MultiAlgorithmEvaluator:
    """多算法评估器（完全使用CONFIG）"""

    def __init__(self):
        self.results = {}
        self.metrics_calc = MetricsCalculator()

    def evaluate_algorithm(
            self,
            env: ImprovedTransformerCoolingEnv,
            agent,
            algorithm_name: str,
            num_episodes: int = None
    ) -> Dict:
        """
        评估单个算法（使用CONFIG参数）

        Args:
            env: 环境
            agent: 智能体
            algorithm_name: 算法名称
            num_episodes: 评估episode数量（None则从CONFIG读取）

        Returns:
            评估结果
        """
        # 🔥 从CONFIG读取
        if num_episodes is None:
            num_episodes = CONFIG.train.EVAL_EPISODES

        print(f"\n🔍 评估算法: {algorithm_name.upper()}")
        print(f"  评估Episodes: {num_episodes} (来自CONFIG)")

        evaluator = Evaluator(env, agent, algorithm_name)

        # 评估多个episodes
        summary = evaluator.evaluate_multiple_episodes(num_episodes)

        # 保存结果
        result = {
            'algorithm': algorithm_name,
            'summary': summary,
            'metrics': summary['metrics'],
            'all_episodes': summary['episodes'],
            'config_info': {  # 🔥 保存CONFIG信息
                'eval_episodes': num_episodes,
                'best_criterion': CONFIG.metrics.BEST_MODEL_CRITERION,
            }
        }

        self.results[algorithm_name] = result

        # 打印关键指标（降温能力优先）
        m = summary['metrics']
        print(f"\n  🔥🔥🔥 降温能力指标（核心）:")
        print(f"    {CONFIG.metrics.BEST_MODEL_CRITERION}:  {m.get('cooling_mae', 0):8.4f}°C  👈 主要评价")

        # 显示所有配置的精度阈值
        for threshold in CONFIG.env.COOLING_PRECISION_THRESHOLDS:
            key = f'cooling_precision_{int(threshold)}c'
            print(f"    精度±{int(threshold)}°C:        {m.get(key, 0):8.2f}%")

        print(f"    总降温量:           {m.get('total_cooling', 0):8.2f}°C")
        print(f"    降温效率:           {m.get('cooling_efficiency', 0):8.4f}")

        print(f"\n  📊 温度相关指标（参考）:")
        print(f"    温度波动范围:       {m.get('temperature_range', 0):8.2f}°C")
        print(f"    温度标准差:         {m.get('temperature_std', 0):8.4f}°C")

        print(f"\n  💰 强化学习指标:")
        print(f"    平均回报:           {m.get('avg_reward', 0):8.2f}")
        print(f"    回报标准差:         {m.get('reward_std', 0):8.4f}")

        return result

    def compare_algorithms(self, save_table: bool = True) -> pd.DataFrame:
        """
        对比所有算法（使用CONFIG参数）

        Args:
            save_table: 是否保存表格

        Returns:
            对比表格
        """
        if not self.results:
            raise ValueError("No evaluation results available.")

        comparison_data = []

        # 🔥 使用CONFIG中定义的指标
        for algo_name, result in self.results.items():
            metrics = result['metrics']

            row = {
                'Algorithm': algo_name.upper().replace('_', ' '),
                # 🔥 降温能力指标（核心）
                f'{CONFIG.metrics.BEST_MODEL_CRITERION} (°C)': metrics.get('cooling_mae', 0),
            }

            # 添加所有配置的精度阈值
            for threshold in CONFIG.env.COOLING_PRECISION_THRESHOLDS:
                key = f'cooling_precision_{int(threshold)}c'
                row[f'Precision ±{int(threshold)}°C (%)'] = metrics.get(key, 0)

            # 其他降温指标
            row.update({
                'Total Cooling (°C)': metrics.get('total_cooling', 0),
                'Cooling Efficiency': metrics.get('cooling_efficiency', 0),
                'Cooling Stability': metrics.get('cooling_stability', 0),
                # 温度指标（参考）
                'Temp Range (°C)': metrics.get('temperature_range', 0),
                'Temp Std (°C)': metrics.get('temperature_std', 0),
                # RL指标
                'Avg Reward': metrics.get('avg_reward', 0),
                'Reward Std': metrics.get('reward_std', 0),
            })

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        if save_table:
            self.save_comparison_table(df)

        return df

    def save_comparison_table(self, df: pd.DataFrame, filename: str = 'algorithm_comparison_cooling_based.csv'):
        """保存完整对比表格"""
        os.makedirs(CONFIG.vis.TABLE_DIR, exist_ok=True)
        filepath = os.path.join(CONFIG.vis.TABLE_DIR, filename)
        df.to_csv(filepath, index=False, float_format='%.4f')
        print(f"\n✓ 对比表格已保存到: {filepath}")

    def print_detailed_results(self):
        """打印详细结果（使用CONFIG格式）"""
        print("\n" + "=" * 100)
        print("详细评估结果（降温能力评价体系 - 来自CONFIG）".center(100))
        print("=" * 100)
        print(f"评估标准: {CONFIG.metrics.BEST_MODEL_CRITERION} (主要)")
        print(f"           {CONFIG.metrics.SECONDARY_CRITERION} (次要)")
        print(f"           {CONFIG.metrics.TERTIARY_CRITERION} (第三)")
        print("=" * 100)

        for algo_name, result in self.results.items():
            print(f"\n算法: {algo_name.upper()}")
            print("-" * 100)

            metrics = result['metrics']
            config_info = result.get('config_info', {})

            print(f"配置: 评估{config_info.get('eval_episodes', 'N/A')}个episodes")

            # 使用metrics.py的打印格式
            self.metrics_calc.print_metrics_summary(metrics)

    def save_all_results(self, filename: str = 'evaluation_results_cooling_based.pkl'):
        """保存所有评估结果"""
        os.makedirs(CONFIG.vis.RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(CONFIG.vis.RESULTS_DIR, filename)

        # 包含CONFIG信息
        save_data = {
            'results': self.results,
            'config_snapshot': {
                'eval_episodes': CONFIG.train.EVAL_EPISODES,
                'best_criterion': CONFIG.metrics.BEST_MODEL_CRITERION,
                'secondary_criterion': CONFIG.metrics.SECONDARY_CRITERION,
                'tertiary_criterion': CONFIG.metrics.TERTIARY_CRITERION,
                'cooling_precision_thresholds': CONFIG.env.COOLING_PRECISION_THRESHOLDS,
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"✓ 评估结果已保存到: {filepath}")
        print("  （包含完整CONFIG快照）")

    def load_all_results(self, filename: str = 'evaluation_results_cooling_based.pkl'):
        """加载评估结果"""
        filepath = os.path.join(CONFIG.vis.RESULTS_DIR, filename)
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        self.results = save_data.get('results', save_data)  # 兼容旧格式

        if 'config_snapshot' in save_data:
            print(f"✓ 评估结果已加载: {filepath}")
            print("  CONFIG快照:")
            for key, value in save_data['config_snapshot'].items():
                print(f"    {key}: {value}")
        else:
            print(f"✓ 评估结果已加载: {filepath} (旧格式，无CONFIG快照)")


def generate_evaluation_csv_files(results: Dict, save_dir: str = None):
    """生成评估相关的CSV文件（使用CONFIG）"""
    if save_dir is None:
        save_dir = CONFIG.vis.RESULTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    for algo_name, algo_results in results.items():
        episodes = algo_results['all_episodes']

        # 只保存前3个episode的详细数据
        for ep_idx, episode in enumerate(episodes[:3]):
            # 温度和降温数据CSV
            data_df = pd.DataFrame({
                'step': range(len(episode['temperatures'])),
                'temperature': episode['temperatures'],
                'actual_cooling': episode['actual_coolings'],
                'target_cooling': episode['target_coolings'],
            })
            data_df.to_csv(
                os.path.join(save_dir, f'{algo_name}_temp_cooling_ep{ep_idx}.csv'),
                index=False
            )

            # 控制动作CSV
            action_df = pd.DataFrame(
                episode['actions'],
                columns=['pressure', 'peltier', 'valve_opening']
            )
            action_df['step'] = range(len(action_df))
            action_df.to_csv(
                os.path.join(save_dir, f'{algo_name}_control_action_ep{ep_idx}.csv'),
                index=False
            )

    print(f"✓ 评估CSV文件已生成到: {save_dir}")


if __name__ == "__main__":
    print("=" * 90)
    print("评估模块测试（完全使用CONFIG参数）".center(90))
    print("=" * 90)

    print("\n✅ 核心改进:")
    print("  1. ✅ 所有参数从CONFIG读取")
    print("  2. ✅ EVAL_EPISODES: CONFIG.train.EVAL_EPISODES")
    print("  3. ✅ 最佳模型判定: CONFIG.metrics.BEST_MODEL_CRITERION")
    print("  4. ✅ 降温精度阈值: CONFIG.env.COOLING_PRECISION_THRESHOLDS")
    print("  5. ✅ 完全移除固定温度依赖")
    print("  6. ✅ 使用MetricsCalculator（不需要target_temp）")

    print("\n📊 CONFIG参数展示:")
    print(f"  EVAL_EPISODES = {CONFIG.train.EVAL_EPISODES}")
    print(f"  BEST_MODEL_CRITERION = '{CONFIG.metrics.BEST_MODEL_CRITERION}'")
    print(f"  SECONDARY_CRITERION = '{CONFIG.metrics.SECONDARY_CRITERION}'")
    print(f"  TERTIARY_CRITERION = '{CONFIG.metrics.TERTIARY_CRITERION}'")
    print(f"  COOLING_PRECISION_THRESHOLDS = {CONFIG.env.COOLING_PRECISION_THRESHOLDS}")

    print("\n" + "=" * 90)
    print("✓ 评估模块修复完成（完全使用CONFIG）".center(90))
    print("=" * 90)
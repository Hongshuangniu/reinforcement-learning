"""
10kV变压器智能冷却控制系统 - 评估模块（完整修复版）
修复指标计算逻辑，确保准确评估
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os
import pickle

from environment import ImprovedTransformerCoolingEnv
from config import CONFIG


class ControlMetricsCalculator:
    """控制性能指标计算器（独立版本）"""

    @staticmethod
    def calculate_temperature_control_metrics(
            temperatures: np.ndarray,
            target_temp: float = 50.0,
            tolerance: float = 2.0
    ) -> Dict[str, float]:
        """
        计算温度控制性能指标

        Args:
            temperatures: 控制后的温度序列
            target_temp: 目标温度
            tolerance: 允许偏差范围

        Returns:
            指标字典
        """
        # 计算温度偏差
        temp_errors = temperatures - target_temp
        abs_errors = np.abs(temp_errors)

        # 基础统计指标
        mae = np.mean(abs_errors)  # 平均绝对误差
        rmse = np.sqrt(np.mean(temp_errors ** 2))  # 均方根误差
        max_ae = np.max(abs_errors)  # 最大绝对误差

        # 相对误差（MAPE）- 对于温度控制，用偏差占目标温度的比例
        mape = np.mean(abs_errors / target_temp) * 100

        # 温度达标率（在允许范围内的比例）
        in_range_ratio = np.mean(abs_errors <= tolerance) * 100

        # 温度稳定性指标
        temp_std = np.std(temperatures)  # 标准差
        temp_range = np.ptp(temperatures)  # 极差

        # 超调指标
        overshoot_ratio = np.mean(temperatures > (target_temp + tolerance)) * 100
        undershoot_ratio = np.mean(temperatures < (target_temp - tolerance)) * 100

        # 温度变化平滑度（连续时刻的温度变化）
        if len(temperatures) > 1:
            temp_changes = np.abs(np.diff(temperatures))
            temp_smoothness = np.mean(temp_changes)  # 平均温度变化率
        else:
            temp_smoothness = 0.0

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MaxAE': max_ae,
            'temp_in_range_ratio': in_range_ratio,
            'temp_std': temp_std,
            'temp_range': temp_range,
            'overshoot_ratio': overshoot_ratio,
            'undershoot_ratio': undershoot_ratio,
            'temp_smoothness': temp_smoothness,
            'avg_temp': np.mean(temperatures),
            'max_temp': np.max(temperatures),
            'min_temp': np.min(temperatures)
        }

    @staticmethod
    def calculate_reward_metrics(rewards: List[float]) -> Dict[str, float]:
        """
        计算强化学习回报指标

        Args:
            rewards: 回报序列

        Returns:
            指标字典
        """
        rewards_arr = np.array(rewards)

        # 基础统计
        total_reward = np.sum(rewards_arr)
        avg_reward = np.mean(rewards_arr)
        reward_std = np.std(rewards_arr)
        reward_variance = np.var(rewards_arr)

        # 收敛性分析（后50%的平均回报）
        mid_point = len(rewards_arr) // 2
        if mid_point > 0:
            late_avg_reward = np.mean(rewards_arr[mid_point:])
        else:
            late_avg_reward = avg_reward

        # 稳定性分析（后50%的标准差）
        if mid_point > 0:
            late_reward_std = np.std(rewards_arr[mid_point:])
        else:
            late_reward_std = reward_std

        return {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'reward_std': reward_std,
            'reward_variance': reward_variance,
            'late_avg_reward': late_avg_reward,
            'late_reward_std': late_reward_std,
            'max_reward': np.max(rewards_arr),
            'min_reward': np.min(rewards_arr)
        }

    @staticmethod
    def calculate_action_metrics(actions: np.ndarray) -> Dict[str, float]:
        """
        计算动作性能指标

        Args:
            actions: 动作序列 (N, action_dim)

        Returns:
            指标字典
        """
        if len(actions) <= 1:
            return {
                'action_smoothness': 0.0,
                'action_std': 0.0
            }

        # 动作平滑度（连续动作的变化）
        action_changes = np.abs(np.diff(actions, axis=0))
        action_smoothness = np.mean(action_changes)

        # 动作标准差（每个维度）
        action_std = np.mean(np.std(actions, axis=0))

        return {
            'action_smoothness': action_smoothness,
            'action_std': action_std
        }


class Evaluator:
    """评估器（完整修复版）"""

    def __init__(self, env: ImprovedTransformerCoolingEnv, agent, algorithm_name: str):
        self.env = env
        self.agent = agent
        self.algorithm_name = algorithm_name
        self.metrics_calc = ControlMetricsCalculator()

    def evaluate_episode(self, deterministic: bool = True) -> Dict:
        """
        评估一个episode

        Args:
            deterministic: 是否使用确定性策略

        Returns:
            episode数据字典（包含所有指标）
        """
        state = self.env.reset()

        temperatures = []  # 实际温度序列
        rewards = []
        actions = []
        log_probs = []

        done = False
        step = 0

        while not done:
            # 选择动作
            if self.algorithm_name == 'ppo':
                action, log_prob, _ = self.agent.select_action(state, evaluate=deterministic)
                log_probs.append(log_prob)
            else:
                action = self.agent.select_action(state, evaluate=deterministic)
                log_probs.append(0.0)

            # 执行动作
            next_state, reward, done, info = self.env.step(action)

            # 收集数据
            temperatures.append(info['oil_temp'])  # 当前油温
            rewards.append(reward)
            actions.append(action.copy())

            state = next_state
            step += 1

        # 转换为numpy数组
        temperatures = np.array(temperatures)
        actions = np.array(actions)
        log_probs = np.array(log_probs)

        # ⭐ 计算温度控制指标
        temp_metrics = self.metrics_calc.calculate_temperature_control_metrics(
            temperatures=temperatures,
            target_temp=CONFIG.env.TARGET_TEMP,
            tolerance=CONFIG.env.TEMP_TOLERANCE
        )

        # ⭐ 计算回报指标
        reward_metrics = self.metrics_calc.calculate_reward_metrics(rewards)

        # ⭐ 计算动作指标
        action_metrics = self.metrics_calc.calculate_action_metrics(actions)

        # 合并所有指标
        all_metrics = {**temp_metrics, **reward_metrics, **action_metrics}

        return {
            'temperatures': temperatures,
            'rewards': rewards,
            'actions': actions,
            'log_probs': log_probs,
            'metrics': all_metrics,
            'total_reward': sum(rewards),
            'avg_temp': np.mean(temperatures),
            'max_temp': np.max(temperatures),
            'min_temp': np.min(temperatures),
            'steps': step
        }

    def evaluate_multiple_episodes(self, num_episodes: int = 10, verbose: bool = True) -> Dict:
        """
        评估多个episodes

        Args:
            num_episodes: 评估episode数量
            verbose: 是否打印进度

        Returns:
            汇总结果
        """
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
            values = [m[key] for m in all_metrics]
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
            'metrics': metrics_summary
        }

        return summary


class MultiAlgorithmEvaluator:
    """多算法评估器"""

    def __init__(self):
        self.results = {}

    def evaluate_algorithm(
            self,
            env: ImprovedTransformerCoolingEnv,
            agent,
            algorithm_name: str,
            num_episodes: int = 10
    ) -> Dict:
        """
        评估单个算法

        Args:
            env: 环境
            agent: 智能体
            algorithm_name: 算法名称
            num_episodes: 评估episode数量

        Returns:
            评估结果
        """
        print(f"\n🔍 评估算法: {algorithm_name.upper()}")
        evaluator = Evaluator(env, agent, algorithm_name)

        # 评估多个episodes
        summary = evaluator.evaluate_multiple_episodes(num_episodes)

        # 保存结果
        result = {
            'algorithm': algorithm_name,
            'summary': summary,
            'metrics': summary['metrics'],
            'all_episodes': summary['episodes']
        }

        self.results[algorithm_name] = result

        # 打印关键指标
        m = summary['metrics']
        print(f"  ✓ MAE: {m['MAE']:.2f}°C | RMSE: {m['RMSE']:.2f}°C | "
              f"达标率: {m['temp_in_range_ratio']:.1f}% | "
              f"平均回报: {m['avg_reward']:.1f}")

        return result

    def compare_algorithms(self, save_table: bool = True) -> pd.DataFrame:
        """
        对比所有算法

        Args:
            save_table: 是否保存表格

        Returns:
            对比表格
        """
        if not self.results:
            raise ValueError("No evaluation results available.")

        comparison_data = []
        for algo_name, result in self.results.items():
            metrics = result['metrics']

            row = {
                'Algorithm': algo_name.upper().replace('_', ' '),
                'MAE (°C)': metrics['MAE'],
                'RMSE (°C)': metrics['RMSE'],
                'MAPE (%)': metrics['MAPE'],
                'MaxAE (°C)': metrics['MaxAE'],
                'Temp In Range (%)': metrics['temp_in_range_ratio'],
                'Temp Std (°C)': metrics['temp_std'],
                'Overshoot (%)': metrics['overshoot_ratio'],
                'Avg Reward': metrics['avg_reward'],
                'Reward Std': metrics['reward_std'],
                'Action Smoothness': metrics['action_smoothness']
            }

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        if save_table:
            self.save_comparison_table(df)

        return df

    def save_comparison_table(self, df: pd.DataFrame, filename: str = 'algorithm_comparison.csv'):
        """保存对比表格"""
        os.makedirs(CONFIG.vis.TABLE_DIR, exist_ok=True)
        filepath = os.path.join(CONFIG.vis.TABLE_DIR, filename)
        df.to_csv(filepath, index=False, float_format='%.4f')
        print(f"\n✓ 对比表格已保存到: {filepath}")

    def print_detailed_results(self):
        """打印详细结果"""
        print("\n" + "=" * 90)
        print("详细评估结果".center(90))
        print("=" * 90)

        for algo_name, result in self.results.items():
            print(f"\n算法: {algo_name.upper()}")
            print("-" * 90)

            metrics = result['metrics']

            # 温度控制性能
            print("\n📊 温度控制性能:")
            print(f"  平均温度偏差 (MAE):        {metrics['MAE']:.4f} °C")
            print(f"  均方根偏差 (RMSE):         {metrics['RMSE']:.4f} °C")
            print(f"  相对误差 (MAPE):          {metrics['MAPE']:.2f} %")
            print(f"  最大偏差 (MaxAE):         {metrics['MaxAE']:.4f} °C")
            print(f"  温度达标率:                {metrics['temp_in_range_ratio']:.2f} %")
            print(f"  温度标准差:                {metrics['temp_std']:.4f} °C")
            print(f"  超调比例:                  {metrics['overshoot_ratio']:.2f} %")
            print(f"  欠调比例:                  {metrics['undershoot_ratio']:.2f} %")
            print(f"  温度平滑度:                {metrics['temp_smoothness']:.4f} °C/step")

            # 控制效果统计
            print("\n📈 控制效果统计:")
            print(f"  平均温度:                  {metrics['avg_temp']:.2f} °C")
            print(f"  温度范围:                  [{metrics['min_temp']:.2f}, {metrics['max_temp']:.2f}] °C")
            print(f"  温度极差:                  {metrics['temp_range']:.2f} °C")

            # 强化学习性能
            print("\n🎯 强化学习性能:")
            print(f"  平均回报:                  {metrics['avg_reward']:.2f}")
            print(f"  回报标准差:                {metrics['reward_std']:.4f}")
            print(f"  后期平均回报:              {metrics['late_avg_reward']:.2f}")
            print(f"  后期回报标准差:            {metrics['late_reward_std']:.4f}")

            # 动作性能
            print("\n🎮 动作性能:")
            print(f"  动作平滑度:                {metrics['action_smoothness']:.4f}")
            print(f"  动作标准差:                {metrics['action_std']:.4f}")

    def save_all_results(self, filename: str = 'evaluation_results.pkl'):
        """保存所有评估结果"""
        os.makedirs(CONFIG.vis.RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(CONFIG.vis.RESULTS_DIR, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"✓ 评估结果已保存到: {filepath}")

    def load_all_results(self, filename: str = 'evaluation_results.pkl'):
        """加载评估结果"""
        filepath = os.path.join(CONFIG.vis.RESULTS_DIR, filename)
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        print(f"✓ 评估结果已加载: {filepath}")


def generate_evaluation_csv_files(results: Dict, save_dir: str = 'results'):
    """生成评估相关的CSV文件"""
    os.makedirs(save_dir, exist_ok=True)

    for algo_name, algo_results in results.items():
        episodes = algo_results['all_episodes']

        for ep_idx, episode in enumerate(episodes[:3]):
            # 温度控制CSV
            temp_df = pd.DataFrame({
                'step': range(len(episode['temperatures'])),
                'temperature': episode['temperatures'],
                'target_temp': CONFIG.env.TARGET_TEMP,
                'upper_bound': CONFIG.env.TARGET_TEMP + CONFIG.env.TEMP_TOLERANCE,
                'lower_bound': CONFIG.env.TARGET_TEMP - CONFIG.env.TEMP_TOLERANCE
            })
            temp_df.to_csv(
                os.path.join(save_dir, f'{algo_name}_temperature_control_ep{ep_idx}.csv'),
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

        print(f"✓ {algo_name} 评估CSV文件已生成")


def generate_metrics_table(results: Dict, save_dir: str = 'tables'):
    """生成指标对比表格"""
    os.makedirs(save_dir, exist_ok=True)

    # 控制性能指标表
    control_data = []
    for algo_name, algo_results in results.items():
        m = algo_results['metrics']
        control_data.append({
            'Algorithm': algo_name.upper().replace('_', ' '),
            'MAE (°C)': m['MAE'],
            'RMSE (°C)': m['RMSE'],
            'MAPE (%)': m['MAPE'],
            'MaxAE (°C)': m['MaxAE'],
            'Temp In Range (%)': m['temp_in_range_ratio'],
            'Temp Std (°C)': m['temp_std'],
            'Overshoot (%)': m['overshoot_ratio']
        })

    control_df = pd.DataFrame(control_data)
    control_df.to_csv(
        os.path.join(save_dir, 'control_performance_metrics.csv'),
        index=False,
        float_format='%.4f'
    )

    # RL性能指标表
    rl_data = []
    for algo_name, algo_results in results.items():
        m = algo_results['metrics']
        rl_data.append({
            'Algorithm': algo_name.upper().replace('_', ' '),
            'Avg Reward': m['avg_reward'],
            'Reward Std': m['reward_std'],
            'Late Avg Reward': m['late_avg_reward'],
            'Action Smoothness': m['action_smoothness']
        })

    rl_df = pd.DataFrame(rl_data)
    rl_df.to_csv(
        os.path.join(save_dir, 'rl_performance_metrics.csv'),
        index=False,
        float_format='%.4f'
    )

    print(f"✓ 指标表格已保存到: {save_dir}")


if __name__ == "__main__":
    print("=" * 70)
    print("评估模块测试（完整修复版）".center(70))
    print("=" * 70)

    print("\n✅ 关键改进:")
    print("  1. ✅ 移除对不存在的 'predicted_temp' 的依赖")
    print("  2. ✅ 直接计算温度与目标温度(50°C)的偏差")
    print("  3. ✅ 独立实现指标计算器，避免依赖外部模块")
    print("  4. ✅ 新增温度达标率、超调率、平滑度等实用指标")
    print("  5. ✅ 详细的指标解释和可视化准备")

    print("\n📊 指标解释:")
    print("  • MAE/RMSE: 越小越好（理想值 <5°C）")
    print("  • 温度达标率: 越高越好（目标 >90%）")
    print("  • 温度标准差: 越小越好（表示控制稳定）")
    print("  • 超调率: 越低越好（避免温度过高）")
    print("  • 动作平滑度: 越小越好（避免频繁调整）")

    print("\n" + "=" * 70)
    print("✓ 评估模块准备就绪".center(70))
    print("=" * 70)

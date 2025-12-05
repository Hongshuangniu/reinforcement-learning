"""
10kV变压器智能冷却控制系统 - 指标计算模块
实现所有评估指标的计算
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class MetricsCalculator:
    """指标计算器"""

    def __init__(self):
        self.metrics_history = []

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算平均绝对误差 (Mean Absolute Error)

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            MAE值
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算均方根误差 (Root Mean Square Error)

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            RMSE值
        """
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)

    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        计算平均绝对百分比误差 (Mean Absolute Percentage Error)

        Args:
            y_true: 真实值
            y_pred: 预测值
            epsilon: 防止除零的小常数

        Returns:
            MAPE值 (百分比)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # 避免除零
        mask = np.abs(y_true) > epsilon
        if not np.any(mask):
            return 100.0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape

    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算决定系数 (R² Score)

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            R²值
        """
        return r2_score(y_true, y_pred)

    @staticmethod
    def calculate_max_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算最大绝对误差 (Maximum Absolute Error)

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            MaxAE值
        """
        return np.max(np.abs(y_true - y_pred))

    def calculate_control_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算所有控制性能指标

        Args:
            y_true: 真实油温序列
            y_pred: 预测/控制油温序列

        Returns:
            包含所有控制指标的字典
        """
        metrics = {
            'MAE': self.calculate_mae(y_true, y_pred),
            'RMSE': self.calculate_rmse(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'R2': self.calculate_r2(y_true, y_pred),
            'MaxAE': self.calculate_max_ae(y_true, y_pred)
        }
        return metrics

    @staticmethod
    def calculate_avg_reward(rewards: List[float]) -> float:
        """
        计算平均回报

        Args:
            rewards: 回报序列

        Returns:
            平均回报
        """
        return np.mean(rewards)

    @staticmethod
    def calculate_convergence_step(rewards: List[float],
                                   window: int = 50,
                                   threshold: float = 0.05) -> int:
        """
        计算收敛步数
        判断标准：窗口内回报的变化率小于阈值

        Args:
            rewards: 回报序列
            window: 判断窗口大小
            threshold: 收敛阈值

        Returns:
            收敛步数（未收敛返回总步数）
        """
        if len(rewards) < window:
            return len(rewards)

        for i in range(window, len(rewards)):
            window_rewards = rewards[i - window:i]
            mean_reward = np.mean(window_rewards)
            std_reward = np.std(window_rewards)

            # 判断收敛：标准差相对于均值的比例小于阈值
            if mean_reward != 0:
                cv = std_reward / abs(mean_reward)
                if cv < threshold:
                    return i

        return len(rewards)

    @staticmethod
    def calculate_reward_variance(rewards: List[float], window: int = 50) -> float:
        """
        计算回报方差（稳定性指标）

        Args:
            rewards: 回报序列
            window: 计算窗口（使用后半部分数据）

        Returns:
            回报方差
        """
        if len(rewards) < window:
            return np.var(rewards)

        # 使用后半部分数据计算方差
        stable_rewards = rewards[-window:]
        return np.var(stable_rewards)

    @staticmethod
    def calculate_policy_entropy(log_probs: np.ndarray) -> float:
        """
        计算策略熵

        Args:
            log_probs: 对数概率序列

        Returns:
            平均策略熵
        """
        # 熵 = -sum(p * log(p))
        # 由于输入是log_probs，熵 = -mean(log_probs)
        entropy = -np.mean(log_probs)
        return entropy

    @staticmethod
    def calculate_action_smoothness(actions: np.ndarray) -> float:
        """
        计算动作平滑度
        使用动作变化的标准差来衡量

        Args:
            actions: 动作序列 (T, action_dim)

        Returns:
            动作平滑度分数（越小越平滑）
        """
        if len(actions) < 2:
            return 0.0

        # 计算相邻动作的差异
        action_diff = np.diff(actions, axis=0)

        # 计算差异的平均标准差
        smoothness = np.mean(np.std(action_diff, axis=0))

        return smoothness

    @staticmethod
    def calculate_temperature_smoothness(temperatures: np.ndarray) -> float:
        """
        计算温度变化平滑度

        Args:
            temperatures: 温度序列

        Returns:
            温度平滑度分数（越小越平滑）
        """
        if len(temperatures) < 2:
            return 0.0

        # 计算温度变化率
        temp_diff = np.diff(temperatures)

        # 计算变化率的标准差
        smoothness = np.std(temp_diff)

        return smoothness

    def calculate_rl_metrics(self,
                             rewards: List[float],
                             log_probs: np.ndarray = None,
                             actions: np.ndarray = None,
                             temperatures: np.ndarray = None) -> Dict[str, float]:
        """
        计算所有强化学习指标

        Args:
            rewards: 回报序列
            log_probs: 对数概率序列（可选）
            actions: 动作序列（可选）
            temperatures: 温度序列（可选）

        Returns:
            包含所有RL指标的字典
        """
        metrics = {
            'avg_reward': self.calculate_avg_reward(rewards),
            'convergence_step': self.calculate_convergence_step(rewards),
            'reward_variance': self.calculate_reward_variance(rewards),
        }

        # 可选指标
        if log_probs is not None:
            metrics['policy_entropy'] = self.calculate_policy_entropy(log_probs)

        if actions is not None:
            metrics['action_smoothness'] = self.calculate_action_smoothness(actions)

        if temperatures is not None:
            metrics['temp_smoothness'] = self.calculate_temperature_smoothness(temperatures)

        return metrics

    def calculate_all_metrics(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              rewards: List[float],
                              log_probs: np.ndarray = None,
                              actions: np.ndarray = None) -> Dict[str, float]:
        """
        计算所有指标（控制性能 + RL指标）

        Args:
            y_true: 真实温度
            y_pred: 预测/控制温度
            rewards: 回报序列
            log_probs: 对数概率序列
            actions: 动作序列

        Returns:
            包含所有指标的字典
        """
        # 控制性能指标
        control_metrics = self.calculate_control_metrics(y_true, y_pred)

        # RL指标
        rl_metrics = self.calculate_rl_metrics(
            rewards=rewards,
            log_probs=log_probs,
            actions=actions,
            temperatures=y_pred
        )

        # 合并所有指标
        all_metrics = {**control_metrics, **rl_metrics}

        # 记录到历史
        self.metrics_history.append(all_metrics)

        return all_metrics

    def get_metrics_summary(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        计算多次运行的指标统计

        Args:
            metrics_list: 多次运行的指标列表

        Returns:
            包含均值和标准差的统计字典
        """
        if not metrics_list:
            return {}

        # 获取所有指标名称
        metric_names = metrics_list[0].keys()

        summary = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list]
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return summary

    def format_metrics_table(self, metrics_dict: Dict[str, float]) -> pd.DataFrame:
        """
        将指标格式化为表格

        Args:
            metrics_dict: 指标字典

        Returns:
            pandas DataFrame
        """
        df = pd.DataFrame({
            'Metric': list(metrics_dict.keys()),
            'Value': list(metrics_dict.values())
        })
        return df

    def save_metrics(self, metrics_dict: Dict[str, float], filepath: str):
        """
        保存指标到CSV文件

        Args:
            metrics_dict: 指标字典
            filepath: 保存路径
        """
        df = self.format_metrics_table(metrics_dict)
        df.to_csv(filepath, index=False)
        print(f"✓ 指标已保存到: {filepath}")


class ComparisonMetrics:
    """算法对比指标计算"""

    def __init__(self):
        self.calculator = MetricsCalculator()

    def compare_algorithms(self,
                           results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        对比多个算法的性能

        Args:
            results_dict: 算法结果字典
                {
                    'algorithm_name': {
                        'y_true': [...],
                        'y_pred': [...],
                        'rewards': [...],
                        ...
                    }
                }

        Returns:
            对比结果DataFrame
        """
        comparison_data = []

        for algo_name, results in results_dict.items():
            # 计算该算法的所有指标
            metrics = self.calculator.calculate_all_metrics(
                y_true=results['y_true'],
                y_pred=results['y_pred'],
                rewards=results['rewards'],
                log_probs=results.get('log_probs'),
                actions=results.get('actions')
            )

            # 添加算法名称
            metrics['Algorithm'] = algo_name
            comparison_data.append(metrics)

        # 创建DataFrame
        df = pd.DataFrame(comparison_data)

        # 重新排列列顺序，将算法名称放在第一列
        cols = ['Algorithm'] + [col for col in df.columns if col != 'Algorithm']
        df = df[cols]

        return df

    def calculate_improvement(self,
                              baseline_metrics: Dict[str, float],
                              improved_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        计算改进百分比

        Args:
            baseline_metrics: 基线算法指标
            improved_metrics: 改进算法指标

        Returns:
            改进百分比字典
        """
        improvements = {}

        for metric_name in baseline_metrics.keys():
            if metric_name in improved_metrics:
                baseline_val = baseline_metrics[metric_name]
                improved_val = improved_metrics[metric_name]

                # 对于误差类指标，减少是改进；对于R²和回报，增加是改进
                if metric_name in ['MAE', 'RMSE', 'MAPE', 'MaxAE', 'reward_variance',
                                   'action_smoothness', 'temp_smoothness']:
                    # 误差减少为正改进
                    improvement = (baseline_val - improved_val) / baseline_val * 100
                else:
                    # 指标增加为正改进
                    improvement = (improved_val - baseline_val) / abs(baseline_val) * 100 if baseline_val != 0 else 0

                improvements[metric_name] = improvement

        return improvements


def calculate_episode_metrics(episode_data: Dict) -> Dict[str, float]:
    """
    计算单个episode的指标

    Args:
        episode_data: episode数据字典

    Returns:
        指标字典
    """
    calculator = MetricsCalculator()

    metrics = calculator.calculate_all_metrics(
        y_true=episode_data['true_temps'],
        y_pred=episode_data['pred_temps'],
        rewards=episode_data['rewards'],
        log_probs=episode_data.get('log_probs'),
        actions=episode_data.get('actions')
    )

    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("指标计算模块测试")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n_samples = 100

    y_true = 60 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_samples)) + np.random.normal(0, 2, n_samples)
    y_pred = y_true + np.random.normal(0, 1, n_samples)
    rewards = np.random.normal(10, 2, n_samples).tolist()
    log_probs = np.random.normal(-1, 0.5, n_samples)
    actions = np.random.randn(n_samples, 3)

    # 创建计算器
    calculator = MetricsCalculator()

    # 计算所有指标
    metrics = calculator.calculate_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        rewards=rewards,
        log_probs=log_probs,
        actions=actions
    )

    # 打印结果
    print("\n控制性能指标:")
    print(f"  MAE:   {metrics['MAE']:.4f}")
    print(f"  RMSE:  {metrics['RMSE']:.4f}")
    print(f"  MAPE:  {metrics['MAPE']:.4f}%")
    print(f"  R²:    {metrics['R2']:.4f}")
    print(f"  MaxAE: {metrics['MaxAE']:.4f}")

    print("\n强化学习指标:")
    print(f"  平均回报:     {metrics['avg_reward']:.4f}")
    print(f"  收敛步数:     {metrics['convergence_step']}")
    print(f"  回报方差:     {metrics['reward_variance']:.4f}")
    print(f"  策略熵:       {metrics['policy_entropy']:.4f}")
    print(f"  动作平滑度:   {metrics['action_smoothness']:.4f}")
    print(f"  温度平滑度:   {metrics['temp_smoothness']:.4f}")

    print("\n" + "=" * 60)
    print("✓ 指标计算模块测试完成")
    print("=" * 60)
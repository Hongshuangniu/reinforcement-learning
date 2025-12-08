"""
评估指标模块 - 完全基于降温能力评价（保留所有工业指标）

核心改进：
1. ✅ 移除固定温度依赖
2. ✅ 所有指标基于"实际降温 vs 目标降温"计算
3. ✅ 保留完整的工业控制指标体系
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

from config import CONFIG


class MetricsCalculator:
    """
    评估指标计算器 - 纯降温能力评价版本
    所有指标基于实际降温量 vs 目标降温量
    """

    def __init__(self):
        """初始化（不再需要target_temp参数）"""
        self.metrics_history = []

    # ============= 🔥 基础降温误差指标（核心）=============

    @staticmethod
    def calculate_cooling_mae(actual_coolings: np.ndarray,
                              target_coolings: np.ndarray) -> float:
        """降温平均绝对误差 (Cooling MAE)"""
        return np.mean(np.abs(actual_coolings - target_coolings))

    @staticmethod
    def calculate_cooling_rmse(actual_coolings: np.ndarray,
                               target_coolings: np.ndarray) -> float:
        """降温均方根误差 (Cooling RMSE)"""
        errors = actual_coolings - target_coolings
        return np.sqrt(np.mean(errors ** 2))

    @staticmethod
    def calculate_cooling_max_error(actual_coolings: np.ndarray,
                                    target_coolings: np.ndarray) -> float:
        """最大降温误差 (Maximum Cooling Error)"""
        return np.max(np.abs(actual_coolings - target_coolings))

    # ============= 🔥 工业控制经典指标（基于降温）=============

    @staticmethod
    def calculate_cooling_ise(actual_coolings: np.ndarray,
                              target_coolings: np.ndarray) -> float:
        """
        降温积分平方误差 (Cooling ISE)
        原理：∫(actual_cooling - target_cooling)² dt
        """
        errors = actual_coolings - target_coolings
        return np.sum(errors ** 2)

    @staticmethod
    def calculate_cooling_iae(actual_coolings: np.ndarray,
                              target_coolings: np.ndarray) -> float:
        """
        降温积分绝对误差 (Cooling IAE)
        原理：∫|actual_cooling - target_cooling| dt
        """
        errors = np.abs(actual_coolings - target_coolings)
        return np.sum(errors)

    @staticmethod
    def calculate_cooling_itae(actual_coolings: np.ndarray,
                               target_coolings: np.ndarray) -> float:
        """
        降温时间加权积分绝对误差 (Cooling ITAE)
        原理：∫t·|actual_cooling - target_cooling| dt
        越晚的误差权重越大
        """
        errors = np.abs(actual_coolings - target_coolings)
        time_weights = np.arange(1, len(errors) + 1)
        return np.sum(time_weights * errors)

    # ============= 🔥 动态性能指标（基于降温）=============

    def calculate_cooling_settling_time(self,
                                        actual_coolings: np.ndarray,
                                        target_coolings: np.ndarray,
                                        tolerance: float = 1.0) -> int:
        """
        降温调节时间 (Cooling Settling Time)
        定义：降温误差最后一次超过tolerance的时刻

        Args:
            tolerance: 允许误差（默认±1°C）

        Returns:
            调节时间（步数）
        """
        errors = np.abs(actual_coolings - target_coolings)

        # 从后往前找，最后一次超过tolerance的位置
        for i in range(len(errors) - 1, -1, -1):
            if errors[i] > tolerance:
                return i + 1
        return 0

    def calculate_cooling_overshoot(self,
                                    actual_coolings: np.ndarray,
                                    target_coolings: np.ndarray) -> float:
        """
        降温超调量 (Cooling Overshoot)
        定义：实际降温超过目标降温的最大百分比

        Returns:
            超调百分比（%）
        """
        # 找到实际降温超过目标的最大值
        overshoot_amounts = actual_coolings - target_coolings
        max_overshoot = np.max(overshoot_amounts)

        if max_overshoot > 0:
            # 相对于平均目标降温的百分比
            avg_target = np.mean(target_coolings)
            if avg_target > 0:
                overshoot_pct = (max_overshoot / avg_target) * 100
            else:
                overshoot_pct = 0.0
        else:
            overshoot_pct = 0.0

        return overshoot_pct

    def calculate_cooling_steady_state_error(self,
                                             actual_coolings: np.ndarray,
                                             target_coolings: np.ndarray,
                                             steady_ratio: float = 0.2) -> float:
        """
        降温稳态误差 (Cooling Steady-State Error)
        定义：最后20%数据的平均降温误差

        Args:
            steady_ratio: 稳态区间比例（默认0.2，即最后20%）

        Returns:
            稳态误差（°C）
        """
        steady_start = int(len(actual_coolings) * (1 - steady_ratio))
        steady_actual = actual_coolings[steady_start:]
        steady_target = target_coolings[steady_start:]

        steady_error = np.mean(np.abs(steady_actual - steady_target))
        return steady_error

    # ============= 🔥 控制精度指标（基于降温）=============

    def calculate_cooling_precision(self,
                                    actual_coolings: np.ndarray,
                                    target_coolings: np.ndarray,
                                    precision_band: float = 2.0) -> float:
        """
        降温控制精度 (Cooling Control Precision)
        定义：降温误差在±precision_band内的比例

        Args:
            precision_band: 精度带宽（默认±2°C）

        Returns:
            精度百分比（%）
        """
        errors = np.abs(actual_coolings - target_coolings)
        in_band = errors <= precision_band
        precision = np.mean(in_band) * 100
        return precision

    def calculate_cooling_stability(self, actual_coolings: np.ndarray) -> float:
        """
        降温稳定性 (Cooling Stability)
        定义：降温量的标准差（越小越稳定）

        Returns:
            稳定性指标（°C）
        """
        return np.std(actual_coolings)

    def calculate_cooling_smoothness(self, actual_coolings: np.ndarray) -> float:
        """
        降温平滑度 (Cooling Smoothness)
        定义：相邻降温量变化的标准差（越小越平滑）

        Returns:
            平滑度指标
        """
        if len(actual_coolings) < 2:
            return 0.0
        cooling_changes = np.abs(np.diff(actual_coolings))
        smoothness = np.std(cooling_changes)
        return smoothness

    # ============= 🔥 降温效果指标 =============

    @staticmethod
    def calculate_total_cooling(actual_coolings: np.ndarray) -> float:
        """
        总降温量 (Total Cooling)
        只统计正的降温量
        """
        positive_coolings = actual_coolings[actual_coolings > 0]
        return np.sum(positive_coolings)

    @staticmethod
    def calculate_cooling_efficiency(actual_coolings: np.ndarray,
                                     actions: np.ndarray) -> float:
        """
        降温效率 (Cooling Efficiency)
        定义：单位能耗的降温量
        """
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        # 计算总能耗（使用CONFIG参数）
        pump_energy = np.sum((actions[:, 0] - CONFIG.env.PUMP_PRESSURE_MIN) /
                             (CONFIG.env.PUMP_PRESSURE_MAX - CONFIG.env.PUMP_PRESSURE_MIN) * 100)
        peltier_energy = np.sum(actions[:, 1] * CONFIG.env.PELTIER_POWER)
        valve_energy = np.sum(actions[:, 2] / 100 * 50)
        total_energy = pump_energy + peltier_energy + valve_energy

        # 计算总降温量
        total_cooling = np.sum(actual_coolings[actual_coolings > 0])

        # 降温效率
        if total_energy > 0:
            efficiency = total_cooling / total_energy * 100
        else:
            efficiency = 0

        return efficiency

    @staticmethod
    def calculate_cooling_achievement_rate(actual_coolings: np.ndarray,
                                           target_coolings: np.ndarray,
                                           threshold: float = 0.8) -> float:
        """
        降温达标率 (Cooling Achievement Rate)
        定义：实际降温达到目标降温X%以上的比例

        Args:
            threshold: 达标阈值（默认0.8，即80%）

        Returns:
            达标率百分比（%）
        """
        achievement = actual_coolings >= (target_coolings * threshold)
        return np.mean(achievement) * 100

    # ============= 📊 温度相关指标（参考）=============

    @staticmethod
    def calculate_temperature_range(temperatures: np.ndarray) -> float:
        """温度波动范围"""
        return np.max(temperatures) - np.min(temperatures)

    @staticmethod
    def calculate_temperature_std(temperatures: np.ndarray) -> float:
        """温度标准差"""
        return np.std(temperatures)

    @staticmethod
    def calculate_temperature_smoothness(temperatures: np.ndarray) -> float:
        """温度变化平滑度"""
        if len(temperatures) < 2:
            return 0.0
        temp_diff = np.diff(temperatures)
        return np.std(temp_diff)

    # ============= ⚙️ 控制性能指标 =============

    @staticmethod
    def calculate_action_smoothness(actions: np.ndarray) -> float:
        """动作平滑度"""
        if len(actions) < 2:
            return 0.0
        action_diff = np.diff(actions, axis=0)
        smoothness = np.mean(np.std(action_diff, axis=0))
        return smoothness

    @staticmethod
    def calculate_control_effort(actions: np.ndarray) -> float:
        """控制努力（动作幅度）"""
        normalized_actions = np.abs(actions)
        return np.mean(normalized_actions)

    # ============= 💰 强化学习指标 =============

    @staticmethod
    def calculate_avg_reward(rewards: List[float]) -> float:
        """平均回报"""
        return np.mean(rewards)

    @staticmethod
    def calculate_reward_std(rewards: List[float]) -> float:
        """回报标准差"""
        return np.std(rewards)

    # ============= 🏆 综合性能评分（基于降温）=============

    def calculate_cooling_performance_index(self,
                                            actual_coolings: np.ndarray,
                                            target_coolings: np.ndarray,
                                            actions: np.ndarray = None,
                                            weights: Optional[Dict] = None) -> Dict[str, float]:
        """
        降温综合性能指标 (Cooling Performance Index) - 0-100分制

        Args:
            actual_coolings: 实际降温量
            target_coolings: 目标降温量
            actions: 动作序列
            weights: 权重字典

        Returns:
            各项评分和总分
        """
        if weights is None:
            weights = {
                'precision': 0.40,  # 降温精度
                'efficiency': 0.25,  # 降温效率
                'stability': 0.20,  # 降温稳定性
                'achievement': 0.15  # 降温达标率
            }

        # 1. 降温精度分（基于MAE）
        mae = self.calculate_cooling_mae(actual_coolings, target_coolings)
        precision_score = max(0, 100 - mae * 8)  # MAE每增加1°C，扣8分

        # 2. 降温效率分
        if actions is not None:
            efficiency = self.calculate_cooling_efficiency(actual_coolings, actions)
            efficiency_score = min(100, efficiency * 10)
        else:
            efficiency_score = 50  # 默认分

        # 3. 降温稳定性分
        stability = self.calculate_cooling_stability(actual_coolings)
        stability_score = max(0, 100 - stability * 10)

        # 4. 降温达标率分
        achievement = self.calculate_cooling_achievement_rate(actual_coolings, target_coolings)
        achievement_score = achievement  # 直接使用百分比

        # 总分
        total_score = (
                weights['precision'] * precision_score +
                weights['efficiency'] * efficiency_score +
                weights['stability'] * stability_score +
                weights['achievement'] * achievement_score
        )

        return {
            'precision_score': precision_score,
            'efficiency_score': efficiency_score,
            'stability_score': stability_score,
            'achievement_score': achievement_score,
            'total_cooling_performance_index': total_score
        }

    # ============= 📦 综合计算方法 =============

    def calculate_all_metrics(self,
                              temperatures: np.ndarray,
                              rewards: List[float],
                              actions: np.ndarray = None,
                              actual_coolings: np.ndarray = None,
                              target_coolings: np.ndarray = None) -> Dict[str, float]:
        """
        🔥 计算所有指标（完全基于降温能力）

        Args:
            temperatures: 温度序列（用于参考）
            rewards: 回报序列
            actions: 动作序列（可选）
            actual_coolings: 实际降温量序列（必需）
            target_coolings: 目标降温量序列（必需）

        Returns:
            完整指标字典
        """
        all_metrics = {}

        # 🔥🔥🔥 核心：降温能力指标 🔥🔥🔥
        if actual_coolings is not None and target_coolings is not None:
            # 1. 基础降温误差
            all_metrics['cooling_mae'] = self.calculate_cooling_mae(
                actual_coolings, target_coolings)
            all_metrics['cooling_rmse'] = self.calculate_cooling_rmse(
                actual_coolings, target_coolings)
            all_metrics['cooling_max_error'] = self.calculate_cooling_max_error(
                actual_coolings, target_coolings)

            # 2. 工业控制指标（基于降温）
            all_metrics['cooling_ise'] = self.calculate_cooling_ise(
                actual_coolings, target_coolings)
            all_metrics['cooling_iae'] = self.calculate_cooling_iae(
                actual_coolings, target_coolings)
            all_metrics['cooling_itae'] = self.calculate_cooling_itae(
                actual_coolings, target_coolings)

            # 3. 动态性能指标（基于降温）
            all_metrics['cooling_settling_time'] = self.calculate_cooling_settling_time(
                actual_coolings, target_coolings, tolerance=1.0)
            all_metrics['cooling_overshoot'] = self.calculate_cooling_overshoot(
                actual_coolings, target_coolings)
            all_metrics['cooling_steady_state_error'] = self.calculate_cooling_steady_state_error(
                actual_coolings, target_coolings)

            # 4. 控制精度（基于降温）
            all_metrics['cooling_precision_1c'] = self.calculate_cooling_precision(
                actual_coolings, target_coolings, 1.0)
            all_metrics['cooling_precision_2c'] = self.calculate_cooling_precision(
                actual_coolings, target_coolings, 2.0)
            all_metrics['cooling_precision_3c'] = self.calculate_cooling_precision(
                actual_coolings, target_coolings, 3.0)

            # 5. 稳定性和平滑度
            all_metrics['cooling_stability'] = self.calculate_cooling_stability(actual_coolings)
            all_metrics['cooling_smoothness'] = self.calculate_cooling_smoothness(actual_coolings)

            # 6. 降温效果
            all_metrics['total_cooling'] = self.calculate_total_cooling(actual_coolings)
            all_metrics['avg_cooling'] = np.mean(actual_coolings)
            all_metrics['max_cooling'] = np.max(actual_coolings)
            all_metrics['min_cooling'] = np.min(actual_coolings)
            all_metrics['avg_target_cooling'] = np.mean(target_coolings)
            all_metrics['cooling_achievement_rate'] = self.calculate_cooling_achievement_rate(
                actual_coolings, target_coolings)

            # 7. 降温效率（需要动作数据）
            if actions is not None:
                all_metrics['cooling_efficiency'] = self.calculate_cooling_efficiency(
                    actual_coolings, actions)

            # 8. 综合性能评分（基于降温）
            if actions is not None:
                pi_scores = self.calculate_cooling_performance_index(
                    actual_coolings, target_coolings, actions)
                all_metrics.update(pi_scores)

        else:
            print("⚠️ 警告: 缺少降温数据，无法计算降温能力指标")

        # 📊 温度相关指标（仅作参考）
        all_metrics['temperature_range'] = self.calculate_temperature_range(temperatures)
        all_metrics['temperature_std'] = self.calculate_temperature_std(temperatures)
        all_metrics['temperature_smoothness'] = self.calculate_temperature_smoothness(temperatures)
        all_metrics['avg_temp'] = np.mean(temperatures)
        all_metrics['max_temp'] = np.max(temperatures)
        all_metrics['min_temp'] = np.min(temperatures)

        # ⚙️ 控制性能指标
        if actions is not None:
            all_metrics['action_smoothness'] = self.calculate_action_smoothness(actions)
            all_metrics['control_effort'] = self.calculate_control_effort(actions)

        # 💰 强化学习指标
        all_metrics['avg_reward'] = self.calculate_avg_reward(rewards)
        all_metrics['reward_std'] = self.calculate_reward_std(rewards)
        all_metrics['total_reward'] = np.sum(rewards)
        all_metrics['max_reward'] = np.max(rewards)
        all_metrics['min_reward'] = np.min(rewards)
        all_metrics['episode_length'] = len(temperatures)

        # 保存历史
        self.metrics_history.append(all_metrics)

        return all_metrics

    # ============= 📝 打印方法 =============

    def print_metrics_summary(self, metrics: Dict[str, float]):
        """打印指标摘要（降温能力优先）"""
        print("\n" + "=" * 100)
        print("评估指标总结（完全基于降温能力）".center(100))
        print("=" * 100)

        # 🔥🔥🔥 降温能力指标（核心，优先显示）
        if 'cooling_mae' in metrics:
            print("\n🔥🔥🔥 降温能力指标（核心评价）:")
            print(f"  【基础误差】")
            print(f"    降温MAE (平均误差):        {metrics.get('cooling_mae', 0):8.4f}°C  ⭐ 主要评价")
            print(f"    降温RMSE:                  {metrics.get('cooling_rmse', 0):8.4f}°C")
            print(f"    最大降温误差:              {metrics.get('cooling_max_error', 0):8.4f}°C")

            print(f"\n  【工业控制指标】（基于降温）")
            print(f"    ISE (积分平方误差):        {metrics.get('cooling_ise', 0):8.2f}")
            print(f"    IAE (积分绝对误差):        {metrics.get('cooling_iae', 0):8.2f}")
            print(f"    ITAE (时间加权误差):       {metrics.get('cooling_itae', 0):8.2f}")

            print(f"\n  【动态性能】（基于降温）")
            print(f"    调节时间:                  {metrics.get('cooling_settling_time', 0):8.0f} 步")
            print(f"    超调量:                    {metrics.get('cooling_overshoot', 0):8.2f}%")
            print(f"    稳态误差:                  {metrics.get('cooling_steady_state_error', 0):8.4f}°C")

            print(f"\n  【控制精度】（基于降温）")
            print(f"    ±1°C精度:                 {metrics.get('cooling_precision_1c', 0):8.2f}%")
            print(f"    ±2°C精度:                 {metrics.get('cooling_precision_2c', 0):8.2f}%")
            print(f"    ±3°C精度:                 {metrics.get('cooling_precision_3c', 0):8.2f}%")

            print(f"\n  【稳定性与平滑度】")
            print(f"    降温稳定性:                {metrics.get('cooling_stability', 0):8.4f}")
            print(f"    降温平滑度:                {metrics.get('cooling_smoothness', 0):8.4f}")

            print(f"\n  【降温效果】")
            print(f"    总降温量:                  {metrics.get('total_cooling', 0):8.2f}°C")
            print(f"    平均降温量:                {metrics.get('avg_cooling', 0):8.2f}°C")
            print(f"    最大单次降温:              {metrics.get('max_cooling', 0):8.2f}°C")
            print(f"    降温达标率:                {metrics.get('cooling_achievement_rate', 0):8.2f}%")

            if 'cooling_efficiency' in metrics:
                print(f"    降温效率:                  {metrics.get('cooling_efficiency', 0):8.4f}")

        # 📊 温度相关指标（仅作参考）
        print("\n📊 温度相关指标（参考）:")
        print(f"  温度波动范围:                {metrics.get('temperature_range', 0):8.2f}°C")
        print(f"  温度标准差:                  {metrics.get('temperature_std', 0):8.4f}°C")
        print(f"  温度平滑度:                  {metrics.get('temperature_smoothness', 0):8.4f}")
        print(f"  平均温度:                    {metrics.get('avg_temp', 0):8.2f}°C")

        # ⚙️ 控制性能指标
        if 'action_smoothness' in metrics:
            print("\n⚙️ 控制性能指标:")
            print(f"  动作平滑度:                  {metrics.get('action_smoothness', 0):8.4f}")
            print(f"  控制努力:                    {metrics.get('control_effort', 0):8.4f}")

        # 💰 强化学习指标
        if 'avg_reward' in metrics:
            print("\n💰 强化学习指标:")
            print(f"  平均回报:                    {metrics.get('avg_reward', 0):8.2f}")
            print(f"  回报标准差:                  {metrics.get('reward_std', 0):8.4f}")
            print(f"  总回报:                      {metrics.get('total_reward', 0):8.2f}")
            print(f"  Episode长度:                 {metrics.get('episode_length', 0):8.0f} 步")

        # 🏆 综合性能评分
        if 'total_cooling_performance_index' in metrics:
            print("\n🏆 综合性能评分 (基于降温, 0-100):")
            print(f"  降温精度分:                  {metrics.get('precision_score', 0):8.2f}")
            print(f"  降温效率分:                  {metrics.get('efficiency_score', 0):8.2f}")
            print(f"  降温稳定性分:                {metrics.get('stability_score', 0):8.2f}")
            print(f"  降温达标率分:                {metrics.get('achievement_score', 0):8.2f}")
            print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"  综合性能指标(CPI):           {metrics.get('total_cooling_performance_index', 0):8.2f}")

        print("\n" + "=" * 100)

        # 📌 评价说明
        print("\n📌 评价说明:")
        print("  🔥 所有指标完全基于'实际降温 vs 目标降温'计算")
        print("  ⭐ 核心指标: 降温MAE（越小越好，目标 <1°C）")
        print("  📊 辅助指标: 降温精度±2°C（越高越好，目标 >90%）")
        print("  🏆 综合评分: 考虑精度、效率、稳定性和达标率")

    def get_best_metric_value(self, metrics: Dict[str, float]) -> float:
        """获取最佳模型判定的主要指标值"""
        return metrics.get(CONFIG.metrics.BEST_MODEL_CRITERION, np.inf)

    def compare_models(self, metrics1: Dict[str, float],
                       metrics2: Dict[str, float]) -> int:
        """
        比较两个模型的性能

        Returns:
            1: metrics1更好
            -1: metrics2更好
            0: 相当
        """
        # 主要标准
        val1 = metrics1.get(CONFIG.metrics.BEST_MODEL_CRITERION, np.inf)
        val2 = metrics2.get(CONFIG.metrics.BEST_MODEL_CRITERION, np.inf)

        threshold = 0.01  # 1%的差异认为相当

        if abs(val1 - val2) / max(val1, val2, 1e-6) < threshold:
            # 使用次要标准
            sec1 = metrics1.get(CONFIG.metrics.SECONDARY_CRITERION, 0)
            sec2 = metrics2.get(CONFIG.metrics.SECONDARY_CRITERION, 0)

            if abs(sec1 - sec2) < 1.0:  # 精度差异<1%
                # 使用第三标准
                ter1 = metrics1.get(CONFIG.metrics.TERTIARY_CRITERION, -np.inf)
                ter2 = metrics2.get(CONFIG.metrics.TERTIARY_CRITERION, -np.inf)
                return 1 if ter1 > ter2 else (-1 if ter1 < ter2 else 0)
            else:
                return 1 if sec1 > sec2 else -1
        else:
            return 1 if val1 < val2 else -1


# ============= 🔥🔥🔥 MSA-SAC专用评价指标（新增）=============

class MSASACMetricsCalculator(MetricsCalculator):
    """
    MSA-SAC专用评估指标计算器

    继承基础MetricsCalculator，添加MSA-SAC特有的评价指标：
    1. 注意力机制有效性指标
    2. 异构Critic网络指标
    3. 维度解耦效果指标
    4. 多尺度特征融合指标
    5. 温度感知自适应指标
    """

    def __init__(self):
        super().__init__()

    # ============= 1. 注意力机制有效性指标 =============

    @staticmethod
    def calculate_attention_entropy(attention_weights: np.ndarray) -> float:
        """
        注意力熵（分布随机性）
        熵越低，注意力越聚焦；熵越高，注意力越分散

        Args:
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]

        Returns:
            平均注意力熵
        """
        # 平均所有头和批次
        if len(attention_weights.shape) == 4:
            attention_weights = attention_weights.mean(axis=(0, 1))
        elif len(attention_weights.shape) == 3:
            attention_weights = attention_weights.mean(axis=0)

        # 计算熵
        entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=-1)
        return float(np.mean(entropy))

    @staticmethod
    def calculate_attention_focus_ratio(attention_weights: np.ndarray) -> float:
        """
        注意力聚焦度（最大权重/平均权重）
        比率越高，注意力越集中

        Returns:
            聚焦度比率
        """
        if len(attention_weights.shape) == 4:
            attention_weights = attention_weights.mean(axis=(0, 1))
        elif len(attention_weights.shape) == 3:
            attention_weights = attention_weights.mean(axis=0)

        max_attention = np.max(attention_weights, axis=-1)
        mean_attention = np.mean(attention_weights, axis=-1)

        focus_ratio = max_attention / (mean_attention + 1e-8)
        return float(np.mean(focus_ratio))

    @staticmethod
    def calculate_attention_stability(attention_weights: np.ndarray) -> float:
        """
        注意力稳定性（随时间的变化）
        值越小，注意力越稳定

        Returns:
            稳定性指标
        """
        if len(attention_weights.shape) == 4:
            attention_weights = attention_weights.mean(axis=1)  # 平均头

        # 计算时间维度上的方差
        attention_var = np.var(attention_weights, axis=0)
        return float(np.mean(attention_var))

    # ============= 2. 异构Critic网络指标 =============

    @staticmethod
    def calculate_critic_discrepancy(q1_values: np.ndarray,
                                     q2_values: np.ndarray) -> Dict[str, float]:
        """
        评估异构双Critic的性能

        Args:
            q1_values: Q1网络输出值
            q2_values: Q2网络输出值

        Returns:
            Critic分歧度指标字典
        """
        metrics = {}

        # 1. Critic分歧度（Q1和Q2的绝对差异）
        critic_diff = np.abs(q1_values - q2_values)
        metrics['critic_discrepancy_mean'] = float(np.mean(critic_diff))
        metrics['critic_discrepancy_std'] = float(np.std(critic_diff))

        # 2. Critic一致性（Q1和Q2同向预测的比例）
        same_direction = np.sign(q1_values) == np.sign(q2_values)
        metrics['critic_agreement_ratio'] = float(np.mean(same_direction) * 100)

        # 3. 过估计偏差
        min_q = np.minimum(q1_values, q2_values)
        avg_q = (q1_values + q2_values) / 2
        overestimation_bias = avg_q - min_q
        metrics['overestimation_bias'] = float(np.mean(overestimation_bias))

        # 4. Critic置信度（方差倒数）
        critic_var = np.var(np.stack([q1_values, q2_values], axis=0), axis=0)
        metrics['critic_confidence'] = float(np.mean(1 / (critic_var + 1e-8)))

        return metrics

    # ============= 3. 维度解耦效果指标 =============

    @staticmethod
    def calculate_action_dimension_independence(actions: np.ndarray) -> Dict[str, float]:
        """
        评估三个动作维度（压力、帕尔贴、阀门）的独立性

        Args:
            actions: 动作序列 [num_steps, 3]

        Returns:
            维度独立性指标字典
        """
        metrics = {}

        if actions.shape[1] != 3:
            print(f"⚠️ 警告: 动作维度应为3，实际为{actions.shape[1]}")
            return metrics

        # 1. 维度间相关系数
        corr_matrix = np.corrcoef(actions.T)  # [3×3] 相关系数矩阵
        metrics['pressure_peltier_correlation'] = float(corr_matrix[0, 1])
        metrics['pressure_valve_correlation'] = float(corr_matrix[0, 2])
        metrics['peltier_valve_correlation'] = float(corr_matrix[1, 2])

        # 2. 平均互相关（越小越独立）
        off_diagonal = np.abs(corr_matrix - np.eye(3))
        metrics['avg_dimension_correlation'] = float(np.mean(off_diagonal))

        # 3. 控制维度方差比
        total_var = np.var(actions)
        if total_var > 0:
            var_ratio = np.var(actions, axis=0) / total_var
            metrics['pressure_variance_ratio'] = float(var_ratio[0])
            metrics['peltier_variance_ratio'] = float(var_ratio[1])
            metrics['valve_variance_ratio'] = float(var_ratio[2])

        # 4. 维度解耦效率（通过PCA评估）
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca.fit(actions)
            explained_variance = pca.explained_variance_ratio_
            metrics['decoupling_efficiency'] = float(explained_variance[0])
            metrics['explained_variance_ratio_1'] = float(explained_variance[0])
            metrics['explained_variance_ratio_2'] = float(explained_variance[1])
            metrics['explained_variance_ratio_3'] = float(explained_variance[2])
        except ImportError:
            print("⚠️ 警告: sklearn未安装，跳过PCA分析")

        return metrics

    # ============= 4. 多尺度特征融合指标 =============

    @staticmethod
    def calculate_multi_scale_effectiveness(
            single_scale_performance: float,
            multi_scale_performance: float
    ) -> Dict[str, float]:
        """
        评估多尺度融合的增益

        Args:
            single_scale_performance: 单尺度性能（如cooling_mae）
            multi_scale_performance: 多尺度性能

        Returns:
            融合效果指标
        """
        metrics = {}

        # 多尺度融合增益（相对改进百分比）
        if single_scale_performance > 0:
            fusion_gain = ((single_scale_performance - multi_scale_performance) /
                           single_scale_performance * 100)
            metrics['fusion_performance_gain'] = float(fusion_gain)

        # 融合效率（绝对改进）
        metrics['fusion_absolute_improvement'] = float(
            single_scale_performance - multi_scale_performance
        )

        return metrics

    # ============= 5. 温度感知自适应指标 =============

    @staticmethod
    def calculate_temperature_adaptation(
            actions: np.ndarray,
            temperatures: np.ndarray,
            actual_coolings: np.ndarray,
            target_coolings: np.ndarray
    ) -> Dict[str, float]:
        """
        评估算法对不同温度条件的适应能力

        Args:
            actions: 动作序列 [num_steps, 3]
            temperatures: 温度序列
            actual_coolings: 实际降温量
            target_coolings: 目标降温量

        Returns:
            温度适应性指标
        """
        metrics = {}

        # 1. 温度-压力相关性（理想情况应呈负相关或正相关，取决于控制策略）
        if len(temperatures) > 1 and len(actions) > 0:
            temp_pressure_corr = np.corrcoef(temperatures[:-1], actions[:-1, 0])[0, 1]
            metrics['temp_pressure_correlation'] = float(temp_pressure_corr)

        # 2. 适应性指数（动作调整与温度变化的响应度）
        if len(temperatures) > 1:
            temp_change = np.diff(temperatures)
            pressure_change = np.diff(actions[:, 0])

            if len(temp_change) == len(pressure_change) and len(temp_change) > 0:
                # 计算响应相关性
                adaptation_corr = np.corrcoef(temp_change, pressure_change)[0, 1]
                metrics['temperature_adaptation_index'] = float(np.abs(adaptation_corr))

        # 3. 温度区间适应性（不同温度区间的控制精度）
        temp_bins = [50, 60, 70, 80, 90]  # 温度区间
        for i in range(len(temp_bins) - 1):
            mask = (temperatures >= temp_bins[i]) & (temperatures < temp_bins[i + 1])
            if np.sum(mask) > 0:
                # 该温度区间的降温误差
                cooling_error = np.abs(actual_coolings[mask] - target_coolings[mask])
                metrics[f'adaptation_{temp_bins[i]}_{temp_bins[i + 1]}'] = float(
                    np.mean(cooling_error)
                )

        # 4. 热惯性补偿度（提前调整能力）
        if len(temperatures) > 2:
            # 计算温度趋势与动作变化的时间滞后相关性
            temp_trend = temperatures[2:] - temperatures[:-2]
            action_response = actions[1:-1, 0] - actions[:-2, 0]

            if len(temp_trend) == len(action_response) and len(temp_trend) > 0:
                lead_corr = np.corrcoef(temp_trend, action_response)[0, 1]
                metrics['thermal_inertia_compensation'] = float(np.abs(lead_corr))

        return metrics

    # ============= 6. MSA-SAC综合评分系统 =============

    def calculate_msa_sac_comprehensive_score(
            self,
            msa_metrics: Dict[str, float],
            baseline_metrics: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        计算MSA-SAC综合改进指数 (Composite Improvement Index, CII)

        Args:
            msa_metrics: MSA-SAC的评估指标
            baseline_metrics: 基线SAC的指标（可选，用于计算改进）

        Returns:
            综合评分字典
        """
        scores = {}

        # === 1. 性能改进评分（与基线对比）===
        if baseline_metrics:
            # 降温精度改进
            if 'cooling_mae' in msa_metrics and 'cooling_mae' in baseline_metrics:
                mae_improvement = ((baseline_metrics['cooling_mae'] -
                                    msa_metrics['cooling_mae']) /
                                   baseline_metrics['cooling_mae'] * 100)
                scores['precision_improvement_score'] = min(100, max(0, 50 + mae_improvement * 2))

            # 降温效率改进
            if 'cooling_efficiency' in msa_metrics and 'cooling_efficiency' in baseline_metrics:
                eff_improvement = ((msa_metrics['cooling_efficiency'] -
                                    baseline_metrics['cooling_efficiency']) /
                                   baseline_metrics['cooling_efficiency'] * 100)
                scores['efficiency_improvement_score'] = min(100, max(0, 50 + eff_improvement))

        # === 2. 算法特性评分（MSA-SAC特有）===

        # 注意力有效性评分
        if 'attention_focus_ratio' in msa_metrics:
            focus_ratio = msa_metrics['attention_focus_ratio']
            if 2 <= focus_ratio <= 5:  # 理想范围
                scores['attention_effectiveness'] = 100.0
            elif 1.5 <= focus_ratio <= 8:
                scores['attention_effectiveness'] = 80.0
            else:
                scores['attention_effectiveness'] = 60.0
        else:
            scores['attention_effectiveness'] = 50.0

        # Critic异构效果评分
        if 'critic_discrepancy_mean' in msa_metrics:
            discrepancy = msa_metrics['critic_discrepancy_mean']
            if 0.05 <= discrepancy <= 0.2:
                scores['critic_heterogeneity'] = 100.0
            elif 0.02 <= discrepancy <= 0.5:
                scores['critic_heterogeneity'] = 80.0
            else:
                scores['critic_heterogeneity'] = 60.0
        else:
            scores['critic_heterogeneity'] = 50.0

        # 维度解耦评分
        if 'pressure_peltier_correlation' in msa_metrics:
            correlation = abs(msa_metrics['pressure_peltier_correlation'])
            if correlation < 0.1:
                scores['dimension_decoupling'] = 100.0
            elif correlation < 0.3:
                scores['dimension_decoupling'] = 80.0
            elif correlation < 0.5:
                scores['dimension_decoupling'] = 60.0
            else:
                scores['dimension_decoupling'] = 40.0
        else:
            scores['dimension_decoupling'] = 50.0

        # 多尺度融合评分
        if 'fusion_performance_gain' in msa_metrics:
            gain = msa_metrics['fusion_performance_gain']
            if gain > 10:
                scores['multi_scale_fusion'] = 100.0
            elif gain > 5:
                scores['multi_scale_fusion'] = 80.0
            elif gain > 0:
                scores['multi_scale_fusion'] = 60.0
            else:
                scores['multi_scale_fusion'] = 40.0
        else:
            scores['multi_scale_fusion'] = 50.0

        # 温度适应评分
        if 'temperature_adaptation_index' in msa_metrics:
            adaptation = msa_metrics['temperature_adaptation_index']
            if adaptation > 0.7:
                scores['temperature_adaptation'] = 100.0
            elif adaptation > 0.5:
                scores['temperature_adaptation'] = 80.0
            elif adaptation > 0.3:
                scores['temperature_adaptation'] = 60.0
            else:
                scores['temperature_adaptation'] = 40.0
        else:
            scores['temperature_adaptation'] = 50.0

        # === 3. 综合改进指数（CII）===
        weights = {
            'precision': 0.30,  # 精度改进
            'efficiency': 0.20,  # 效率改进
            'attention': 0.15,  # 注意力机制
            'critic': 0.10,  # Critic网络
            'decoupling': 0.10,  # 维度解耦
            'fusion': 0.10,  # 多尺度融合
            'adaptation': 0.05  # 温度适应
        }

        cii = (
                weights['precision'] * scores.get('precision_improvement_score', 50) +
                weights['efficiency'] * scores.get('efficiency_improvement_score', 50) +
                weights['attention'] * scores['attention_effectiveness'] +
                weights['critic'] * scores['critic_heterogeneity'] +
                weights['decoupling'] * scores['dimension_decoupling'] +
                weights['fusion'] * scores['multi_scale_fusion'] +
                weights['adaptation'] * scores['temperature_adaptation']
        )

        scores['composite_improvement_index'] = float(cii)

        # === 4. 评级 ===
        if cii >= 90:
            scores['rating'] = "卓越 (Excellent)"
        elif cii >= 80:
            scores['rating'] = "优秀 (Very Good)"
        elif cii >= 70:
            scores['rating'] = "良好 (Good)"
        elif cii >= 60:
            scores['rating'] = "中等 (Average)"
        else:
            scores['rating'] = "需要改进 (Needs Improvement)"

        return scores

    def print_msa_sac_metrics_summary(self, msa_metrics: Dict[str, float]):
        """打印MSA-SAC专用指标摘要"""

        print("\n" + "=" * 100)
        print("MSA-SAC专用评价指标总结".center(100))
        print("=" * 100)

        # 1. 降温能力指标（基础）
        self.print_metrics_summary(msa_metrics)

        # 2. MSA-SAC特有指标
        print("\n" + "=" * 100)
        print("🔥🔥🔥 MSA-SAC算法特性指标".center(100))
        print("=" * 100)

        # 注意力机制
        if 'attention_focus_ratio' in msa_metrics:
            print("\n  【注意力机制有效性】")
            print(f"    注意力聚焦度:           {msa_metrics.get('attention_focus_ratio', 0):8.2f}")
            print(f"    注意力熵:               {msa_metrics.get('attention_entropy', 0):8.4f}")
            print(f"    注意力稳定性:           {msa_metrics.get('attention_stability', 0):8.4f}")

        # Critic网络
        if 'critic_discrepancy_mean' in msa_metrics:
            print(f"\n  【异构Critic网络】")
            print(f"    Critic分歧度:           {msa_metrics.get('critic_discrepancy_mean', 0):8.4f}")
            print(f"    Critic一致性:           {msa_metrics.get('critic_agreement_ratio', 0):8.2f}%")
            print(f"    过估计偏差:             {msa_metrics.get('overestimation_bias', 0):8.4f}")
            print(f"    Critic置信度:           {msa_metrics.get('critic_confidence', 0):8.4f}")

        # 维度解耦
        if 'pressure_peltier_correlation' in msa_metrics:
            print(f"\n  【动作维度解耦】")
            print(f"    压力-帕尔贴相关性:      {msa_metrics.get('pressure_peltier_correlation', 0):8.4f}")
            print(f"    压力-阀门相关性:        {msa_metrics.get('pressure_valve_correlation', 0):8.4f}")
            print(f"    帕尔贴-阀门相关性:      {msa_metrics.get('peltier_valve_correlation', 0):8.4f}")
            print(f"    平均维度相关:           {msa_metrics.get('avg_dimension_correlation', 0):8.4f}")
            print(f"    解耦效率:               {msa_metrics.get('decoupling_efficiency', 0):8.4f}")

        # 温度适应
        if 'temperature_adaptation_index' in msa_metrics:
            print(f"\n  【温度感知自适应】")
            print(f"    温度适应指数:           {msa_metrics.get('temperature_adaptation_index', 0):8.4f}")
            print(f"    温度-压力相关性:        {msa_metrics.get('temp_pressure_correlation', 0):8.4f}")
            print(f"    热惯性补偿:             {msa_metrics.get('thermal_inertia_compensation', 0):8.4f}")

        # 综合评分
        if 'composite_improvement_index' in msa_metrics:
            print(f"\n  【🏆 综合改进指数 (CII)】")
            print(f"    注意力有效性分:         {msa_metrics.get('attention_effectiveness', 0):8.2f}")
            print(f"    Critic异构分:           {msa_metrics.get('critic_heterogeneity', 0):8.2f}")
            print(f"    维度解耦分:             {msa_metrics.get('dimension_decoupling', 0):8.2f}")
            print(f"    多尺度融合分:           {msa_metrics.get('multi_scale_fusion', 0):8.2f}")
            print(f"    温度适应分:             {msa_metrics.get('temperature_adaptation', 0):8.2f}")
            print(f"    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"    综合改进指数(CII):      {msa_metrics.get('composite_improvement_index', 0):8.2f}/100")
            print(f"    评级:                   {msa_metrics.get('rating', 'N/A')}")

        print("\n" + "=" * 100)


# ============= 辅助函数 =============

def calculate_msa_sac_all_metrics(
        temperatures: np.ndarray,
        rewards: List[float],
        actions: np.ndarray,
        actual_coolings: np.ndarray,
        target_coolings: np.ndarray,
        attention_weights: np.ndarray = None,
        q1_values: np.ndarray = None,
        q2_values: np.ndarray = None,
        baseline_metrics: Dict[str, float] = None
) -> Dict[str, float]:
    """
    计算MSA-SAC的所有评价指标（便捷函数）

    Args:
        temperatures: 温度序列
        rewards: 回报序列
        actions: 动作序列
        actual_coolings: 实际降温量
        target_coolings: 目标降温量
        attention_weights: 注意力权重（可选）
        q1_values: Q1网络输出（可选）
        q2_values: Q2网络输出（可选）
        baseline_metrics: 基线指标（可选）

    Returns:
        完整的MSA-SAC指标字典
    """
    calculator = MSASACMetricsCalculator()

    # 1. 计算基础降温能力指标
    all_metrics = calculator.calculate_all_metrics(
        temperatures=temperatures,
        rewards=rewards,
        actions=actions,
        actual_coolings=actual_coolings,
        target_coolings=target_coolings
    )

    # 2. 添加MSA-SAC特有指标

    # 注意力机制指标
    if attention_weights is not None:
        all_metrics['attention_entropy'] = calculator.calculate_attention_entropy(attention_weights)
        all_metrics['attention_focus_ratio'] = calculator.calculate_attention_focus_ratio(attention_weights)
        all_metrics['attention_stability'] = calculator.calculate_attention_stability(attention_weights)

    # Critic网络指标
    if q1_values is not None and q2_values is not None:
        critic_metrics = calculator.calculate_critic_discrepancy(q1_values, q2_values)
        all_metrics.update(critic_metrics)

    # 维度解耦指标
    if actions is not None and actions.shape[1] == 3:
        dimension_metrics = calculator.calculate_action_dimension_independence(actions)
        all_metrics.update(dimension_metrics)

    # 温度适应指标
    if temperatures is not None and actions is not None:
        temp_metrics = calculator.calculate_temperature_adaptation(
            actions, temperatures, actual_coolings, target_coolings
        )
        all_metrics.update(temp_metrics)

    # 3. 计算综合评分
    comprehensive_scores = calculator.calculate_msa_sac_comprehensive_score(
        all_metrics, baseline_metrics
    )
    all_metrics.update(comprehensive_scores)

    return all_metrics


if __name__ == "__main__":
    print("=" * 100)
    print("评估指标模块测试（完全基于降温能力）".center(100))
    print("=" * 100)

    # 生成测试数据
    np.random.seed(42)
    n_samples = 100

    temperatures = 60 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_samples)) + np.random.normal(0, 2, n_samples)
    rewards = np.random.normal(10, 2, n_samples).tolist()
    actions = np.random.rand(n_samples, 3)
    actions[:, 0] = actions[:, 0] * 3 + 2  # 压力
    actions[:, 2] = actions[:, 2] * 100  # 阀门

    # 🔥 生成降温数据
    actual_coolings = np.random.normal(8, 2, n_samples)  # 实际降温量
    target_coolings = np.full(n_samples, 8.0)  # 目标降温量

    # 创建计算器（不需要target_temp参数）
    calculator = MetricsCalculator()

    # 计算所有指标
    metrics = calculator.calculate_all_metrics(
        temperatures=temperatures,
        rewards=rewards,
        actions=actions,
        actual_coolings=actual_coolings,
        target_coolings=target_coolings
    )

    # 打印结果
    calculator.print_metrics_summary(metrics)

    print("\n" + "=" * 100)
    print("✓ 评估指标模块测试完成".center(100))
    print("=" * 100)

    print("\n✅ 核心特点:")
    print("  1. ✅ 完全移除固定温度依赖")
    print("  2. ✅ 所有指标基于'实际降温 vs 目标降温'")
    print("  3. ✅ 保留完整的工业控制指标体系（ISE/IAE/ITAE等）")
    print("  4. ✅ 所有指标都是基于降温能力的原生计算")
    print("  5. ✅ 指标含义更符合冷却系统的实际控制目标")

    print("\n📊 指标对照:")
    print("  原指标                 → 新指标（基于降温）")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  温度MAE                → 降温MAE")
    print("  温度ISE                → 降温ISE")
    print("  温度调节时间           → 降温调节时间")
    print("  温度超调量             → 降温超调量")
    print("  温度控制精度           → 降温控制精度")
    print("  温度稳态误差           → 降温稳态误差")

    # 测试模型比较
    print("\n📊 模型比较功能测试:")
    metrics2 = metrics.copy()
    metrics2['cooling_mae'] = metrics['cooling_mae'] * 1.1

    result = calculator.compare_models(metrics, metrics2)
    print(f"  Model 1 vs Model 2: {'Model 1 更好' if result > 0 else ('Model 2 更好' if result < 0 else '相当')}")
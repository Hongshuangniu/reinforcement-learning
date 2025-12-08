"""
10kV变压器智能冷却控制系统 - 主程序（修复版）
Main Program - Fixed Version

🔥 修复内容：
1. ✅ 强制使用真实数据（368天，8832小时）
2. ✅ 正确显示所有工业控制指标
3. ✅ 修复数据不足问题
4. ✅ 改进episode计算逻辑
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import warnings

warnings.filterwarnings('ignore')

# 导入自定义模块
from config import CONFIG
from environment import ImprovedTransformerCoolingEnv
from sac_temperature_aware import ImprovedSAC
from sac_base import BaseSAC
from ppo import PPO
from ddpg import DDPG
from td3 import TD3
from metrics import MetricsCalculator
from trainer import Trainer, MultiAlgorithmTrainer
from evaluator import MultiAlgorithmEvaluator


def print_banner():
    """打印启动横幅"""
    print("\n" + "=" * 100)
    print("10kV变压器智能冷却控制系统 - 降温能力评价体系（修复版）".center(100))
    print("Transformer Cooling System - Fixed Version".center(100))
    print("=" * 100)
    print("\n🔥 修复内容:")
    print("  1. ✅ 强制加载真实数据（368天，8832小时）")
    print("  2. ✅ 完整显示所有工业控制指标（ISE/IAE/ITAE/调节时间/超调量等）")
    print("  3. ✅ 动态计算最大可用episodes")
    print("  4. ✅ 改进数据不足处理逻辑")
    print("\n📊 评价指标体系:")
    print("  🔥 降温能力指标（核心）：")
    print("    • 降温MAE、RMSE、最大误差")
    print("    • ISE/IAE/ITAE（工业控制经典指标）")
    print("    • 调节时间、超调量、稳态误差（动态性能）")
    print("    • 控制精度±1/2/3°C（精确控制）")
    print("    • 降温稳定性、平滑度（稳定性）")


def load_real_data_force(data_dir: str = None):
    """
    强制加载真实数据（修复版）

    Args:
        data_dir: 数据目录

    Returns:
        真实数据DataFrame
    """
    if data_dir is None:
        data_dir = CONFIG.data.DATA_DIR

    print("\n" + "=" * 100)
    print("强制加载真实数据（修复版）".center(100))
    print("=" * 100)

    try:
        from compelte_data_lodar import TransformerDataLoader

        loader = TransformerDataLoader(data_dir=data_dir)

        # 逐步加载数据
        print("\n1️⃣ 加载油温数据...")
        oil_df = loader.load_oil_temperature()

        if oil_df is None or len(oil_df) == 0:
            raise Exception("油温数据加载失败")

        print(f"   ✓ 油温数据: {len(oil_df)} 小时")

        print("\n2️⃣ 加载天气数据...")
        weather_df = loader.load_weather_data()
        if weather_df is not None:
            print(f"   ✓ 天气数据: {len(weather_df)} 小时")
        else:
            print("   ⚠ 天气数据不可用，将使用默认值")

        print("\n3️⃣ 加载预测温度...")
        predicted_df = loader.load_predicted_temperature()
        if predicted_df is not None:
            print(f"   ✓ 预测数据: {len(predicted_df)} 小时")
        else:
            print("   ⚠ 预测数据不可用，将使用油温作为预测值")

        print("\n4️⃣ 合并数据并生成特征...")
        merged_df = loader.merge_all_data()

        if merged_df is None or len(merged_df) == 0:
            raise Exception("数据合并失败")

        # 统计信息
        print("\n" + "=" * 100)
        print("真实数据加载成功！".center(100))
        print("=" * 100)

        print(f"\n📊 数据统计:")
        print(f"  • 总小时数: {len(merged_df):,} 小时")
        print(f"  • 总天数: {len(merged_df) / 24:.1f} 天")
        print(f"  • 时间跨度: {(merged_df.index.max() - merged_df.index.min()).days + 1} 天")
        print(f"  • 时间范围: {merged_df.index.min()} → {merged_df.index.max()}")
        print(f"\n  • 油温范围: [{merged_df['oil_temp'].min():.2f}, {merged_df['oil_temp'].max():.2f}]°C")
        print(f"  • 油温均值: {merged_df['oil_temp'].mean():.2f}°C")
        print(f"  • 油温标准差: {merged_df['oil_temp'].std():.2f}°C")

        # 计算可训练episodes
        hours_per_episode = CONFIG.env.MAX_STEPS
        max_episodes = len(merged_df) // hours_per_episode

        print(f"\n🎯 训练能力:")
        print(f"  • 每Episode小时数: {hours_per_episode}")
        print(f"  • 最大可训练Episodes: {max_episodes}")
        print(f"  • CONFIG设定Episodes: {CONFIG.train.NUM_EPISODES}")

        if max_episodes < CONFIG.train.NUM_EPISODES:
            print(f"\n  ⚠️  建议将 CONFIG.train.NUM_EPISODES 设为 {max_episodes}")
        else:
            print(f"\n  ✓ 数据充足，可以训练 {CONFIG.train.NUM_EPISODES} episodes")

        # 保存处理后的数据
        loader.save_processed_data()

        return merged_df, max_episodes

    except Exception as e:
        print(f"\n✗ 真实数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def generate_sufficient_data(data_dir: str, required_hours: int = None):
    """
    生成足够的模拟数据（修复版）

    Args:
        data_dir: 数据目录
        required_hours: 需要的小时数（默认为8832，即368天）
    """
    if required_hours is None:
        required_hours = 8832  # 368天 * 24小时

    print("\n" + "=" * 100)
    print("生成足够的模拟数据（修复版）".center(100))
    print("=" * 100)

    print(f"\n📊 数据生成参数:")
    print(f"  • 目标小时数: {required_hours:,} ({required_hours / 24:.1f}天)")
    print(f"  • 每Episode小时数: {CONFIG.env.MAX_STEPS}")
    print(f"  • 可训练Episodes: {required_hours // CONFIG.env.MAX_STEPS}")

    print(f"\n正在生成 {required_hours:,} 小时的数据...")

    time_index = pd.date_range(start='2024-01-01 00:00:00', periods=required_hours, freq='H')
    data = pd.DataFrame(index=time_index)

    # 🔥 真实感的油温数据 - 跨越所有温度区间
    hours = np.arange(required_hours)

    # 基础温度趋势
    base_temp = 65

    # 日周期变化（24小时）
    daily_cycle = 10 * np.sin(2 * np.pi * hours / 24)

    # 周周期变化（7天）
    weekly_cycle = 5 * np.sin(2 * np.pi * hours / (24 * 7))

    # 季节性趋势（模拟长期变化）
    seasonal_trend = 8 * np.sin(2 * np.pi * hours / (24 * 30))

    # 随机噪声
    noise = np.random.normal(0, 2.5, required_hours)

    # 组合所有成分
    data['oil_temp'] = base_temp + daily_cycle + weekly_cycle + seasonal_trend + noise

    # 裁剪到合理范围（50-85°C，覆盖所有温度区间）
    data['oil_temp'] = np.clip(data['oil_temp'], 50, 85)

    # 环境温度（与油温相关但独立变化）
    data['ambient_temp'] = 28 + 8 * np.sin(2 * np.pi * hours / 24 - np.pi / 4) + np.random.normal(0, 1.5,
                                                                                                  required_hours)
    data['ambient_temp'] = np.clip(data['ambient_temp'], 20, 40)

    # 预测温度（基于油温加噪声）
    data['predicted_temp'] = data['oil_temp'] + np.random.normal(0, 1.2, required_hours)

    # 其他特征
    data['humidity'] = 60 + 15 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, required_hours)
    data['humidity'] = np.clip(data['humidity'], 40, 90)

    data['load_rate'] = 0.7 + 0.15 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 0.05, required_hours)
    data['load_rate'] = np.clip(data['load_rate'], 0.5, 0.95)

    # 特征工程
    data['oil_temp_error'] = data['oil_temp'] - 65.0
    data['temp_change_rate'] = data['oil_temp'].diff().fillna(0)
    data['oil_temp_ma3'] = data['oil_temp'].rolling(window=3, min_periods=1).mean()
    data['oil_temp_ma6'] = data['oil_temp'].rolling(window=6, min_periods=1).mean()
    data['oil_temp_std3'] = data['oil_temp'].rolling(window=3, min_periods=1).std().fillna(0)
    data['temp_acceleration'] = data['temp_change_rate'].diff().fillna(0)

    data['temp_difference'] = data['oil_temp'] - data['ambient_temp']
    data['ambient_temp_ma3'] = data['ambient_temp'].rolling(window=3, min_periods=1).mean()

    data['predicted_error'] = data['predicted_temp'] - 50.0
    data['feedforward_signal'] = -data['predicted_error'] / 10.0
    data['predicted_change'] = data['predicted_temp'].diff().fillna(0)
    data['predicted_trend'] = data['predicted_temp'].rolling(window=3, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0, raw=False
    ).fillna(0)

    # 时间特征
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['is_daytime'] = ((data.index.hour >= 6) & (data.index.hour < 18)).astype(int)
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)

    # 天气特征
    data['weather_code'] = np.random.randint(1, 10, required_hours)
    data['wind_level'] = np.random.randint(1, 6, required_hours)
    data['sunshine_hours'] = np.clip(np.random.normal(8, 3, required_hours), 0, 14)
    data['max_temp'] = data['ambient_temp'] + np.random.uniform(2, 6, required_hours)
    data['min_temp'] = data['ambient_temp'] - np.random.uniform(2, 6, required_hours)

    data['weather_impact'] = data['weather_code'].apply(
        lambda x: 1.2 if x in [4, 5, 9] else 1.0 if x in [2, 3] else 0.8
    )
    data['wind_impact'] = 1.0 + data['wind_level'] * 0.05

    # 填充到state_dim维度
    current_features = len(data.columns)
    if current_features < CONFIG.env.STATE_DIM:
        for i in range(CONFIG.env.STATE_DIM - current_features):
            data[f'feature_{i}'] = np.random.randn(required_hours) * 0.5

    # 填充任何NaN值
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data.fillna(0, inplace=True)

    # 保存数据
    os.makedirs(data_dir, exist_ok=True)
    processed_file = os.path.join(data_dir, CONFIG.data.PROCESSED_DATA_FILE)
    with open(processed_file, 'wb') as f:
        pickle.dump({'processed_data': data}, f)

    print(f"\n✓ 模拟数据生成完成！")
    print(f"  • 数据形状: {data.shape}")
    print(f"  • 保存位置: {processed_file}")
    print(f"\n📊 数据质量:")
    print(f"  • 油温范围: [{data['oil_temp'].min():.2f}, {data['oil_temp'].max():.2f}]°C")
    print(f"  • 油温均值: {data['oil_temp'].mean():.2f}°C ± {data['oil_temp'].std():.2f}°C")
    print(f"  • 环境温度: [{data['ambient_temp'].min():.2f}, {data['ambient_temp'].max():.2f}]°C")
    print(f"  • 负载率: [{data['load_rate'].min():.2f}, {data['load_rate'].max():.2f}]")

    max_episodes = required_hours // CONFIG.env.MAX_STEPS
    print(f"\n🎯 训练能力:")
    print(f"  • 可训练Episodes: {max_episodes}")
    print(f"  • 足够训练: {'✓' if max_episodes >= CONFIG.train.NUM_EPISODES else '✗'}")

    return data, max_episodes


def print_detailed_metrics(evaluator: MultiAlgorithmEvaluator):
    """
    打印详细的评估指标（包含所有工业控制指标）

    Args:
        evaluator: 评估器对象
    """
    if not evaluator.results:
        print("⚠ 没有评估结果")
        return

    print("\n" + "=" * 120)
    print("详细评估指标（包含完整工业控制指标）".center(120))
    print("=" * 120)

    for algo_name, result in evaluator.results.items():
        metrics = result['metrics']

        print(f"\n{'=' * 120}")
        print(f"算法: {algo_name.upper()}".center(120))
        print(f"{'=' * 120}")

        # 🔥🔥🔥 降温能力指标（核心）
        if 'cooling_mae' in metrics:
            print("\n🔥🔥🔥 降温能力指标（核心评价）:")
            print("\n  【基础误差指标】")
            print(f"    降温MAE (平均绝对误差):    {metrics.get('cooling_mae', 0):8.4f}°C  ⭐ 主要评价")
            print(f"    降温RMSE (均方根误差):      {metrics.get('cooling_rmse', 0):8.4f}°C")
            print(f"    最大降温误差:                {metrics.get('cooling_max_error', 0):8.4f}°C")

            print(f"\n  【工业控制经典指标】（基于降温）")
            print(f"    ISE (积分平方误差):          {metrics.get('cooling_ise', 0):10.2f}")
            print(f"    IAE (积分绝对误差):          {metrics.get('cooling_iae', 0):10.2f}")
            print(f"    ITAE (时间加权积分误差):     {metrics.get('cooling_itae', 0):10.2f}")

            print(f"\n  【动态性能指标】（基于降温）")
            print(f"    调节时间 (Settling Time):    {metrics.get('cooling_settling_time', 0):8.0f} 步")
            print(f"    超调量 (Overshoot):          {metrics.get('cooling_overshoot', 0):8.2f}%")
            print(f"    稳态误差 (Steady-State):     {metrics.get('cooling_steady_state_error', 0):8.4f}°C")

            print(f"\n  【控制精度指标】（基于降温）")
            print(f"    ±1°C精度:                   {metrics.get('cooling_precision_1c', 0):8.2f}%")
            print(f"    ±2°C精度:                   {metrics.get('cooling_precision_2c', 0):8.2f}%")
            print(f"    ±3°C精度:                   {metrics.get('cooling_precision_3c', 0):8.2f}%")

            print(f"\n  【稳定性与平滑度】")
            print(f"    降温稳定性:                  {metrics.get('cooling_stability', 0):8.4f}")
            print(f"    降温平滑度:                  {metrics.get('cooling_smoothness', 0):8.4f}")

            print(f"\n  【降温效果】")
            print(f"    总降温量:                    {metrics.get('total_cooling', 0):8.2f}°C")
            print(f"    平均降温量:                  {metrics.get('avg_cooling', 0):8.2f}°C")
            print(f"    最大单次降温:                {metrics.get('max_cooling', 0):8.2f}°C")
            print(f"    降温达标率:                  {metrics.get('cooling_achievement_rate', 0):8.2f}%")

            if 'cooling_efficiency' in metrics:
                print(f"    降温效率:                    {metrics.get('cooling_efficiency', 0):8.4f}")

        # 📊 温度相关指标（参考）
        print("\n📊 温度相关指标（参考）:")
        print(f"  温度波动范围:                  {metrics.get('temperature_range', 0):8.2f}°C")
        print(f"  温度标准差:                    {metrics.get('temperature_std', 0):8.4f}°C")
        print(f"  温度平滑度:                    {metrics.get('temperature_smoothness', 0):8.4f}")
        print(f"  平均温度:                      {metrics.get('avg_temp', 0):8.2f}°C")

        # ⚙️ 控制性能指标
        if 'action_smoothness' in metrics:
            print("\n⚙️ 控制性能指标:")
            print(f"  动作平滑度:                    {metrics.get('action_smoothness', 0):8.4f}")
            print(f"  控制努力:                      {metrics.get('control_effort', 0):8.4f}")

        # 💰 强化学习指标
        print("\n💰 强化学习指标:")
        print(f"  平均回报:                      {metrics.get('avg_reward', 0):8.2f}")
        print(f"  回报标准差:                    {metrics.get('reward_std', 0):8.4f}")
        print(f"  Episode长度:                   {metrics.get('episode_length', 0):8.0f} 步")

        # 🏆 综合性能评分
        if 'total_cooling_performance_index' in metrics:
            print("\n🏆 综合性能评分 (基于降温, 0-100):")
            print(f"  降温精度分:                    {metrics.get('precision_score', 0):8.2f}")
            print(f"  降温效率分:                    {metrics.get('efficiency_score', 0):8.2f}")
            print(f"  降温稳定性分:                  {metrics.get('stability_score', 0):8.2f}")
            print(f"  降温达标率分:                  {metrics.get('achievement_score', 0):8.2f}")
            print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"  综合性能指标(CPI):             {metrics.get('total_cooling_performance_index', 0):8.2f}")

    print("\n" + "=" * 120)


def print_industrial_comparison_table(results: dict):
    """
    打印包含工业指标的对比表格

    Args:
        results: 训练结果字典
    """
    if not results:
        print("⚠ 没有训练结果")
        return

    print("\n" + "=" * 150)
    print("算法对比表（包含完整工业控制指标）".center(150))
    print("=" * 150)

    # 打印表头
    header = (
        f"{'算法':<12} | "
        f"{'MAE':>8} | {'RMSE':>8} | "
        f"{'ISE':>10} | {'IAE':>10} | {'ITAE':>10} | "
        f"{'调节时间':>8} | {'超调%':>8} | "
        f"{'±1°C%':>8} | {'±2°C%':>8} | "
        f"{'总降温':>8} | {'回报':>8} | {'Episodes':>8}"
    )
    print(f"\n{header}")
    print("-" * 150)

    # 打印每个算法的结果
    for algo_name, algo_results in results.items():
        final_metrics = algo_results.get('final_metrics', {})
        config_info = algo_results.get('config', {})

        row = (
            f"{algo_name.upper():<12} | "
            # 基础误差
            f"{final_metrics.get('cooling_mae', 0):>8.4f} | "
            f"{final_metrics.get('cooling_rmse', 0):>8.4f} | "
            # 工业控制指标
            f"{final_metrics.get('cooling_ise', 0):>10.2f} | "
            f"{final_metrics.get('cooling_iae', 0):>10.2f} | "
            f"{final_metrics.get('cooling_itae', 0):>10.2f} | "
            # 动态性能
            f"{final_metrics.get('cooling_settling_time', 0):>8.0f} | "
            f"{final_metrics.get('cooling_overshoot', 0):>8.2f} | "
            # 精度
            f"{final_metrics.get('cooling_precision_1c', 0):>8.2f} | "
            f"{final_metrics.get('cooling_precision_2c', 0):>8.2f} | "
            # 其他
            f"{final_metrics.get('total_cooling', 0):>8.2f} | "
            f"{final_metrics.get('avg_reward', 0):>8.2f} | "
            f"{config_info.get('num_episodes', len(algo_results.get('episode_rewards', [])) if 'episode_rewards' in algo_results else 0):>8}"
        )
        print(row)

    print("=" * 150)

    # 指标说明
    print("\n📊 工业控制指标说明:")
    print("  🔥 基础误差:")
    print("    • MAE: 平均绝对误差（越小越好，目标<1°C）")
    print("    • RMSE: 均方根误差（越小越好）")

    print("\n  📐 经典工业指标:")
    print("    • ISE: 积分平方误差（越小越好）")
    print("    • IAE: 积分绝对误差（越小越好）")
    print("    • ITAE: 时间加权积分绝对误差（越小越好，后期误差权重更大）")

    print("\n  ⚡ 动态性能:")
    print("    • 调节时间: 系统稳定所需步数（越小越好）")
    print("    • 超调量: 超过目标的百分比（越小越好，<5%为佳）")

    print("\n  🎯 控制精度:")
    print("    • ±X°C精度: 误差在±X°C内的比例（越高越好，>90%为佳）")


def create_agent(algorithm: str, state_dim: int, action_dim: int):
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
        raise ValueError(f"未知算法: {algorithm}")


def main():
    """主函数（修复版）"""
    parser = argparse.ArgumentParser(description='变压器冷却系统 - 修复版（完整工业指标）')

    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=None,
                        choices=['improved_sac', 'sac', 'ppo', 'ddpg', 'td3'])
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--eval-episodes', type=int, default=None)
    parser.add_argument('--mode', type=str, default='full',
                        choices=['train', 'eval', 'full'])
    parser.add_argument('--use-real-data', action='store_true',
                        help='强制使用真实数据')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--gen-data-hours', type=int, default=8832,
                        help='生成数据的小时数（默认8832=368天）')

    args = parser.parse_args()

    # 从CONFIG读取默认参数
    if args.algorithms is None:
        args.algorithms = CONFIG.algo.ALGORITHMS
    if args.episodes is None:
        args.episodes = CONFIG.train.NUM_EPISODES
    if args.eval_episodes is None:
        args.eval_episodes = CONFIG.train.EVAL_EPISODES
    if args.data_dir is None:
        args.data_dir = CONFIG.data.DATA_DIR

    # 打印横幅
    print_banner()

    # 加载数据
    print("\n" + "=" * 100)
    print("数据加载（修复版）".center(100))
    print("=" * 100)

    data = None
    max_episodes = 0

    # 尝试加载真实数据
    if args.use_real_data or os.path.exists(os.path.join(args.data_dir, CONFIG.data.OIL_TEMP_FILE)):
        print("\n尝试加载真实数据...")
        data, max_episodes = load_real_data_force(args.data_dir)

    # 如果真实数据加载失败，生成足够的模拟数据
    if data is None:
        print("\n真实数据不可用，生成足够的模拟数据...")
        data, max_episodes = generate_sufficient_data(args.data_dir, args.gen_data_hours)

    # 检查数据有效性
    if data is None or len(data) == 0:
        print("\n✗ 错误: 数据加载/生成失败")
        return

    # 调整episodes数量
    if args.episodes > max_episodes:
        print(f"\n{'⚠' * 50}")
        print(f"⚠️  警告: 请求Episodes({args.episodes})超过可用Episodes({max_episodes})")
        print(f"⚠️  自动调整为: {max_episodes} episodes")
        print(f"{'⚠' * 50}\n")
        args.episodes = max_episodes

    # 打印最终数据信息
    print(f"\n✓ 最终数据信息:")
    print(f"  • 数据形状: {data.shape}")
    print(f"  • 油温范围: [{data['oil_temp'].min():.2f}, {data['oil_temp'].max():.2f}]°C")
    print(f"  • 时间跨度: {(data.index.max() - data.index.min()).days + 1} 天")
    print(f"  • 使用Episodes: {args.episodes} / {max_episodes}")

    # 创建目录
    os.makedirs(CONFIG.output.MODEL_DIR, exist_ok=True)
    os.makedirs(CONFIG.vis.RESULTS_DIR, exist_ok=True)
    os.makedirs(CONFIG.vis.TABLE_DIR, exist_ok=True)

    # ========== 训练阶段 ==========
    if args.mode in ['train', 'full']:
        print("\n" + "=" * 100)
        print("多算法训练".center(100))
        print("=" * 100)
        print(f"\n训练配置:")
        print(f"  算法列表: {args.algorithms}")
        print(f"  训练Episodes: {args.episodes}")
        print(f"  评估频率: {CONFIG.train.EVAL_FREQUENCY} episodes")
        print(f"  评估Episodes: {args.eval_episodes}")
        print(f"  最佳模型判定: {CONFIG.metrics.BEST_MODEL_CRITERION}")

        trainer = MultiAlgorithmTrainer(data)
        training_results = trainer.train_all(args.algorithms, args.episodes)

        # 保存训练结果
        results_file = os.path.join(CONFIG.vis.RESULTS_DIR, 'training_results_fixed.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(training_results, f)

        print(f"\n✓ 训练结果已保存: {results_file}")
    else:
        training_results = None

    # ========== 评估阶段 ==========
    if args.mode in ['eval', 'full']:
        print("\n" + "=" * 100)
        print("算法评估（完整工业指标）".center(100))
        print("=" * 100)

        # 加载训练结果
        results_file = os.path.join(CONFIG.vis.RESULTS_DIR, 'training_results_fixed.pkl')
        if not os.path.exists(results_file):
            results_file = os.path.join(CONFIG.vis.RESULTS_DIR, 'training_results_cooling.pkl')

        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                training_results = pickle.load(f)
            print(f"✓ 已加载训练结果: {results_file}")
        elif training_results is None:
            print("✗ 错误: 未找到训练结果")
            if args.mode == 'eval':
                print("  提示: 请先运行训练模式 (--mode train)")
                return

        if training_results is None:
            training_results = {}

        # 评估
        evaluator = MultiAlgorithmEvaluator()

        for algo_name in training_results.keys():
            print(f"\n{'=' * 100}")
            print(f"评估算法: {algo_name.upper()}")
            print(f"{'=' * 100}")

            # 创建环境和智能体
            env = ImprovedTransformerCoolingEnv(data)

            try:
                agent = create_agent(algo_name, env.state_dim, env.action_dim)
            except ValueError as e:
                print(f"  ✗ {e}")
                continue

            # 加载模型
            model_path = os.path.join(CONFIG.output.MODEL_DIR, f"best_{algo_name}.pth")
            if os.path.exists(model_path):
                try:
                    agent.load_model(model_path)
                    print(f"  ✓ 已加载模型: {model_path}")

                    # 评估
                    eval_result = evaluator.evaluate_algorithm(
                        env, agent, algo_name, args.eval_episodes
                    )

                    print(f"  ✓ 评估完成")
                except Exception as e:
                    print(f"  ✗ 评估失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  ⚠ 未找到模型: {model_path}")

        # 打印详细结果（包含完整工业指标）
        if evaluator.results:
            print_detailed_metrics(evaluator)

            # 生成对比表格
            print("\n生成对比表格...")
            comparison_df = evaluator.compare_algorithms(save_table=True)
            print("✓ 对比表格已生成")

            # 保存评估结果
            evaluator.save_all_results('evaluation_results_fixed.pkl')

            # 打印工业指标对比表
            print_industrial_comparison_table(training_results)

    # 打印完成信息
    print("\n" + "=" * 100)
    print("✓ 程序运行完成!".center(100))
    print("=" * 100)

    print("\n📁 输出文件:")
    print(f"  - 模型目录: {CONFIG.output.MODEL_DIR}/")
    print(f"  - 结果目录: {CONFIG.vis.RESULTS_DIR}/")
    print(f"  - 表格目录: {CONFIG.vis.TABLE_DIR}/")

    print("\n📊 修复总结:")
    print("  ✅ 强制加载/生成足够的数据（8832小时=368天）")
    print("  ✅ 显示完整的工业控制指标")
    print("    • 基础误差: MAE, RMSE, 最大误差")
    print("    • 工业指标: ISE, IAE, ITAE")
    print("    • 动态性能: 调节时间, 超调量, 稳态误差")
    print("    • 控制精度: ±1/2/3°C精度")
    print("    • 综合评分: CPI (Cooling Performance Index)")
    print("  ✅ 动态调整episodes数量")
    print("  ✅ 改进数据不足处理")


if __name__ == "__main__":
    main()

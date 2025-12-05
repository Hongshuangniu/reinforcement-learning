"""
10kV变压器智能冷却控制系统 - 主程序（改进版）
Main Program with Clear Multi-Algorithm Training

改进要点：
1. ✅ 清晰显示将训练哪些算法
2. ✅ 明确显示使用的数据源（真实/模拟）
3. ✅ 添加数据检查和确认提示
4. ✅ 优化训练流程和输出信息
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import warnings
from scipy.io import savemat

warnings.filterwarnings('ignore')

# 导入自定义模块
from config import CONFIG
from environment import ImprovedTransformerCoolingEnv, MultiEpisodeEnv
from sac_temperature_aware import ImprovedSAC
from sac_base import BaseSAC
from ppo import PPO
from ddpg import DDPG
from td3 import TD3
from trainer import Trainer, MultiAlgorithmTrainer
from evaluator import MultiAlgorithmEvaluator, generate_evaluation_csv_files, generate_metrics_table
from metrics import MetricsCalculator


def print_banner():
    """打印启动横幅"""
    print("\n" + "=" * 80)
    print("10kV变压器智能冷却控制系统 - 多算法强化学习训练".center(80))
    print("Transformer Cooling System - Multi-Algorithm RL Training".center(80))
    print("=" * 80)


def check_data_files(data_dir: str = 'data') -> dict:
    """
    检查数据文件是否存在

    Returns:
        dict: 包含文件检查结果
    """
    print("\n" + "=" * 80)
    print("数据文件检查".center(80))
    print("=" * 80)

    required_files = {
        'oil_temp': 'Oil_temperature_data_for_July_2024.xlsx',
        'weather': 'Weather_data_for_24_hours_on_July_2024.xlsx',
        'predicted': 'Predicted_temperature_data_for_July_2024.xlsx'
    }

    file_status = {}
    all_exist = True

    for key, filename in required_files.items():
        filepath = os.path.join(data_dir, filename)
        exists = os.path.exists(filepath)
        file_status[key] = exists

        status_icon = "✓" if exists else "✗"
        status_text = "存在" if exists else "缺失"
        color = "\033[92m" if exists else "\033[91m"
        reset = "\033[0m"

        print(f"{color}{status_icon}{reset} {filename:<50} [{status_text}]")

        if not exists:
            all_exist = False

    print("=" * 80)

    if all_exist:
        print("\n✓ 所有真实数据文件都存在，将使用真实Excel数据训练")
        return {'status': 'real', 'files': file_status}
    else:
        print("\n⚠ 部分数据文件缺失，将使用模拟数据训练")
        print("提示: 如需使用真实数据，请将Excel文件放置在 'data/' 目录下")
        return {'status': 'simulated', 'files': file_status}


def load_and_process_data(data_dir: str = None, force_reload: bool = False):
    """
    加载和预处理数据（真实数据版）

    Args:
        data_dir: 数据目录路径
        force_reload: 是否强制重新加载（忽略缓存）

    Returns:
        处理后的数据DataFrame
    """
    if data_dir is None:
        data_dir = CONFIG.data.DATA_DIR

    processed_file = os.path.join(data_dir, CONFIG.data.PROCESSED_DATA_FILE)

    # 如果不强制重新加载，尝试使用缓存
    if not force_reload and os.path.exists(processed_file):
        print(f"\n发现已处理的数据: {processed_file}")
        print("正在加载缓存数据...")
        try:
            with open(processed_file, 'rb') as f:
                data_dict = pickle.load(f)
            print("✓ 成功加载缓存数据")
            return data_dict['processed_data']
        except Exception as e:
            print(f"⚠ 缓存加载失败: {e}")
            print("将重新处理数据...")

    # 尝试加载真实数据
    print("\n" + "=" * 70)
    print("尝试加载真实Excel数据".center(70))
    print("=" * 70)

    try:
        # 导入数据加载器（假设已创建）
        from complete_data_loader import TransformerDataLoader

        loader = TransformerDataLoader(data_dir=data_dir)
        data = loader.load_all_and_process()

        if data is not None and len(data) > 0:
            print("\n✓ 成功加载真实数据！")
            return data
        else:
            raise Exception("数据加载返回空值")

    except ImportError:
        print("\n⚠ 未找到数据加载器模块 (complete_data_loader.py)")
        print("请先运行数据加载脚本")
    except FileNotFoundError as e:
        print(f"\n⚠ Excel文件未找到: {e}")
        print("请确保以下文件存在:")
        print(f"  - {data_dir}/Oil_temperature_data_for_July_2024.xlsx")
        print(f"  - {data_dir}/Weather_data_for_24_hours_on_July_2024.xlsx")
        print(f"  - {data_dir}/Predicted_temperature_data_for_July_2024.xlsx")
    except Exception as e:
        print(f"\n⚠ 真实数据加载失败: {e}")
        import traceback
        traceback.print_exc()

    # 如果真实数据加载失败，生成模拟数据（仅用于测试）
    print("\n" + "=" * 70)
    print("⚠ 使用模拟数据（仅供测试）".center(70))
    print("=" * 70)
    print("警告: 模拟数据无法代表真实场景！")
    print("建议: 请加载真实Excel数据以获得准确结果")
    print("=" * 70)

    return generate_simulation_data(data_dir)


def generate_simulation_data(data_dir: str):
    """生成模拟数据（备用方案）"""
    print("\n生成模拟数据...")

    # 生成48小时的模拟数据
    n_hours = 48 * 30  # 30天数据

    # 时间序列
    time_index = pd.date_range(
        start='2024-07-01 00:00:00',
        periods=n_hours,
        freq='H'
    )

    # 生成模拟数据
    data = pd.DataFrame(index=time_index)

    # 油温（正弦变化 + 噪声 + 趋势）
    base_temp = 60
    daily_variation = 10 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    weekly_trend = 5 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 7))
    noise = np.random.normal(0, 2, n_hours)
    data['oil_temp'] = base_temp + daily_variation + weekly_trend + noise

    # 环境温度
    ambient_base = 28
    ambient_variation = 8 * np.sin(2 * np.pi * np.arange(n_hours) / 24 - np.pi / 4)
    data['ambient_temp'] = ambient_base + ambient_variation + np.random.normal(0, 1, n_hours)

    # 预测温度（油温 + 小偏移）
    data['predicted_temp'] = data['oil_temp'] + np.random.normal(0, 1, n_hours)

    # 湿度
    data['humidity'] = 60 + np.random.normal(0, 5, n_hours)

    # 负载率
    data['load_rate'] = 0.7 + 0.2 * np.sin(2 * np.pi * np.arange(n_hours) / 24) + \
                        np.random.normal(0, 0.05, n_hours)

    # 时间特征
    data['hour'] = data.index.hour
    data['day'] = data.index.day
    data['month'] = data.index.month

    # 添加更多特征以满足state_dim要求
    data['oil_temp_error'] = data['oil_temp'] - 50.0
    data['temp_change_rate'] = data['oil_temp'].diff().fillna(0)
    data['oil_temp_ma3'] = data['oil_temp'].rolling(window=3, min_periods=1).mean()
    data['feedforward_signal'] = -(data['predicted_temp'] - 50.0) / 10.0

    for i in range(CONFIG.env.STATE_DIM - len(data.columns)):
        data[f'feature_{i}'] = np.random.randn(n_hours)

    # 保存处理后的数据
    processed_file = os.path.join(data_dir, CONFIG.data.PROCESSED_DATA_FILE)
    with open(processed_file, 'wb') as f:
        pickle.dump({'processed_data': data}, f)

    print(f"✓ 模拟数据已生成并保存: {processed_file}")
    print(f"  数据形状: {data.shape}")
    print(f"  时间范围: {data.index[0]} 到 {data.index[-1]}")

    return data


def confirm_training_config(algorithms: list, num_episodes: int, data_status: str):
    """
    确认训练配置

    Args:
        algorithms: 要训练的算法列表
        num_episodes: 训练episodes
        data_status: 数据状态（'real' 或 'simulated'）

    Returns:
        bool: 是否继续训练
    """
    print("\n" + "=" * 80)
    print("训练配置确认".center(80))
    print("=" * 80)

    print(f"\n📊 数据源: {'✓ 真实Excel数据' if data_status == 'real' else '⚠ 模拟测试数据'}")
    print(f"🎯 训练Episodes: {num_episodes}")
    print(f"🤖 训练算法数量: {len(algorithms)}")
    print(f"\n将训练以下算法:")

    algorithm_names = {
        'improved_sac': 'Improved SAC (TD3-SAC混合)',
        'sac': 'SAC (软Actor-Critic)',
        'ppo': 'PPO (近端策略优化)',
        'ddpg': 'DDPG (深度确定性策略梯度)',
        'td3': 'TD3 (双延迟DDPG)'
    }

    for i, algo in enumerate(algorithms, 1):
        name = algorithm_names.get(algo, algo.upper())
        print(f"  {i}. {name}")

    print("\n" + "=" * 80)

    if data_status == 'simulated':
        print("\n⚠ 警告: 当前使用模拟数据，训练结果仅供参考！")
        print("建议: 使用真实Excel数据以获得准确的模型性能")

    # 自动继续（非交互模式）
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='10kV变压器智能冷却系统 - 多算法训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练所有算法（默认）
  python main.py --mode full --episodes 100

  # 只训练特定算法
  python main.py --algorithms improved_sac sac --episodes 100

  # 只评估已训练的模型
  python main.py --mode eval

  # 强制重新加载数据
  python main.py --force-reload
        """
    )

    # ⭐ 默认训练所有算法
    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=['improved_sac', 'sac', 'ppo', 'ddpg', 'td3'],
                        choices=['improved_sac', 'sac', 'ppo', 'ddpg', 'td3'],
                        help='要训练的算法列表（默认：所有算法）')

    parser.add_argument('--episodes', type=int, default=None,
                        help='训练episodes (None=自动根据数据量调整)')

    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='评估episodes')

    parser.add_argument('--mode', type=str, default='full',
                        choices=['train', 'eval', 'full'],
                        help='运行模式: train(仅训练), eval(仅评估), full(训练+评估)')

    parser.add_argument('--force-reload', action='store_true',
                        help='强制重新加载数据（忽略缓存）')

    parser.add_argument('--skip-confirmation', action='store_true',
                        help='跳过配置确认提示')

    args = parser.parse_args()

    # 打印横幅
    print_banner()

    # 打印配置
    CONFIG.print_config()

    # ⭐ 检查数据文件
    data_check = check_data_files()
    data_status = data_check['status']

    # 加载数据
    print("\n加载数据...")
    data = load_and_process_data(force_reload=args.force_reload)
    print(f"✓ 数据形状: {data.shape}")
    print(f"✓ 时间范围: {data.index[0]} 到 {data.index[-1]}")

    # ⭐ 计算可用episodes
    available_episodes = len(data) // CONFIG.env.MAX_STEPS
    print(f"✓ 可用训练episodes: {available_episodes}")

    # ⭐ 自动调整训练episodes
    if args.episodes is None:
        # 使用数据增强可以增加3-5倍的有效episodes
        if CONFIG.aug.USE_AUGMENTATION:
            args.episodes = min(available_episodes * 3, CONFIG.train.NUM_EPISODES)
            print(f"✓ 数据增强已启用，调整训练episodes到: {args.episodes}")
        else:
            args.episodes = min(available_episodes, CONFIG.train.NUM_EPISODES)
            print(f"⚠ 数据增强未启用，训练episodes: {args.episodes}")

    # ⭐ 显示将训练的算法
    print("\n" + "=" * 80)
    print("训练配置".center(80))
    print("=" * 80)
    print(f"数据源: {'真实Excel数据 ✓' if data_status == 'real' else '模拟数据 ⚠'}")
    print(f"算法数量: {len(args.algorithms)}")
    print(f"训练Episodes: {args.episodes}")
    print(f"评估Episodes: {args.eval_episodes}")
    print(f"运行模式: {args.mode}")
    print(f"\n将训练的算法:")
    for i, algo in enumerate(args.algorithms, 1):
        print(f"  {i}. {algo.upper()}")
    print("=" * 80)

    # 确认配置
    if not args.skip_confirmation:
        if not confirm_training_config(args.algorithms, args.episodes, data_status):
            print("\n训练已取消")
            return

    # 创建目录
    os.makedirs(CONFIG.output.MODEL_DIR, exist_ok=True)
    os.makedirs(CONFIG.vis.RESULTS_DIR, exist_ok=True)

    # ========== 训练阶段 ==========
    if args.mode in ['train', 'full']:
        print("\n" + "=" * 80)
        print("开始多算法训练".center(80))
        print("=" * 80)

        trainer = MultiAlgorithmTrainer(data, CONFIG.train)
        training_results = trainer.train_all(args.algorithms, args.episodes)

        # 保存训练结果
        with open(os.path.join(CONFIG.vis.RESULTS_DIR, 'training_results.pkl'), 'wb') as f:
            pickle.dump(training_results, f)

        print("\n✓ 训练结果已保存")
    else:
        training_results = None

    # ========== 评估阶段 ==========
    if args.mode in ['eval', 'full']:
        print("\n" + "=" * 80)
        print("开始算法评估".center(80))
        print("=" * 80)

        # 加载训练结果
        results_file = os.path.join(CONFIG.vis.RESULTS_DIR, 'training_results.pkl')
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                training_results = pickle.load(f)
        elif training_results is None:
            print("错误: 未找到训练结果")
            if args.mode == 'eval':
                print("提示: 请先运行训练模式 (--mode train)")
                return
            else:
                print("⚠ 跳过评估阶段")
                training_results = {}

        # 确保training_results不是None
        if training_results is None:
            training_results = {}

        # 评估
        evaluator = MultiAlgorithmEvaluator()

        for algo_name in training_results.keys():
            print(f"\n评估 {algo_name.upper()}...")

            # 创建环境和智能体
            env = ImprovedTransformerCoolingEnv(data)

            if algo_name == 'improved_sac':
                agent = ImprovedSAC(env.state_dim, env.action_dim)
            elif algo_name == 'sac':
                agent = BaseSAC(env.state_dim, env.action_dim)
            elif algo_name == 'ppo':
                agent = PPO(env.state_dim, env.action_dim)
            elif algo_name == 'ddpg':
                agent = DDPG(env.state_dim, env.action_dim)
            elif algo_name == 'td3':
                agent = TD3(env.state_dim, env.action_dim)
            else:
                print(f"  ⚠ 跳过未知算法: {algo_name}")
                continue

            # 加载最佳模型
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
                    metrics = eval_result['metrics']
                    print(f"    平均回报:  {eval_result['summary']['avg_reward']:.2f}")
                    print(f"    MAE:       {metrics.get('MAE', 0):.4f}°C")
                    print(f"    RMSE:      {metrics.get('RMSE', 0):.4f}°C")
                    print(f"    R²:        {metrics.get('R2', 0):.4f}")
                except Exception as e:
                    print(f"  ✗ 评估失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  ⚠ 未找到模型: {model_path}")

        # 打印详细结果
        if evaluator.results:
            evaluator.print_detailed_results()

            # 生成对比表格
            comparison_df = evaluator.compare_algorithms(save_table=True)
            print("\n算法对比:")
            print(comparison_df.to_string(index=False))

            # 保存评估结果
            evaluator.save_all_results()

            # 生成CSV和表格
            generate_evaluation_csv_files(evaluator.results, 'results')
            generate_metrics_table(evaluator.results, 'tables')

    # 打印完成信息
    print("\n" + "=" * 80)
    print("✓ 程序运行完成!".center(80))
    print("=" * 80)

    print("\n📁 输出文件:")
    print(f"  - 模型: {CONFIG.output.MODEL_DIR}/")
    print(f"  - 结果: {CONFIG.vis.RESULTS_DIR}/")
    print(f"  - 表格: {CONFIG.vis.TABLE_DIR}/")

    print("\n💡 后续步骤:")
    print("  1. 查看训练曲线: 检查 results/ 目录")
    print("  2. 分析对比表格: 查看 tables/ 目录")
    print("  3. 测试最佳模型: 使用 --mode eval")


if __name__ == "__main__":
    main()
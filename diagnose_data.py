"""
数据诊断工具 - 检查训练结果文件的内容和结构
"""

import pickle
import os
from pprint import pprint


def diagnose_pickle_file(filepath):
    """诊断pickle文件"""
    print("=" * 80)
    print(f"诊断文件: {filepath}")
    print("=" * 80)

    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"✗ 文件不存在: {filepath}")
        return None

    # 检查文件大小
    file_size = os.path.getsize(filepath)
    print(f"\n文件大小: {file_size:,} bytes ({file_size / 1024:.2f} KB)")

    if file_size == 0:
        print("✗ 文件为空!")
        return None

    # 尝试加载
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print(f"\n✓ 文件加载成功")
        print(f"数据类型: {type(data)}")

        if data is None:
            print("✗ 数据为None!")
            return None

        # 如果是字典，显示结构
        if isinstance(data, dict):
            print(f"\n顶层键: {list(data.keys())}")

            # 显示每个键的详细信息
            for key, value in data.items():
                print(f"\n{'=' * 60}")
                print(f"键: {key}")
                print(f"值类型: {type(value)}")

                if isinstance(value, dict):
                    print(f"子键: {list(value.keys())[:10]}...")  # 只显示前10个

                    # 特别检查results键
                    if key == 'results':
                        print(f"\n算法列表: {list(value.keys())}")

                        # 检查每个算法
                        for algo, algo_data in value.items():
                            print(f"\n  算法: {algo}")
                            if isinstance(algo_data, dict):
                                print(f"    键: {list(algo_data.keys())}")

                                # 检查training_stats
                                if 'training_stats' in algo_data:
                                    stats = algo_data['training_stats']
                                    print(f"    ✓ 找到 training_stats")
                                    if isinstance(stats, dict):
                                        print(f"      子键: {list(stats.keys())}")
                                        for stat_key, stat_val in stats.items():
                                            if isinstance(stat_val, (list, tuple)):
                                                print(f"        {stat_key}: {len(stat_val)} 条记录")
                                            else:
                                                print(f"        {stat_key}: {type(stat_val)}")
                                else:
                                    print(f"    ✗ 没有 training_stats")

                                # 检查episode_rewards
                                if 'episode_rewards' in algo_data:
                                    rewards = algo_data['episode_rewards']
                                    print(f"    ✓ episode_rewards: {len(rewards)} episodes")

                elif isinstance(value, (list, tuple)):
                    print(f"长度: {len(value)}")
                    if len(value) > 0:
                        print(f"第一个元素类型: {type(value[0])}")
                else:
                    print(f"值: {value}")

        return data

    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_all_training_files():
    """检查所有可能的训练结果文件"""
    results_dir = 'results'

    possible_files = [
        'training_results_fixed.pkl',
        'training_results_cooling_based.pkl',
        'training_results.pkl'
    ]

    print("\n" + "=" * 80)
    print("检查所有训练结果文件")
    print("=" * 80)

    found_files = []
    for filename in possible_files:
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            found_files.append(filepath)
            print(f"\n✓ 找到: {filename}")

    if not found_files:
        print("\n✗ 没有找到任何训练结果文件!")
        print(f"\n请检查目录: {os.path.abspath(results_dir)}")
        return

    # 诊断每个文件
    for filepath in found_files:
        data = diagnose_pickle_file(filepath)

        if data is not None:
            print(f"\n✓ {filepath} 可以正常使用")
        else:
            print(f"\n✗ {filepath} 有问题")

        print("\n" + "=" * 80)


def suggest_fix():
    """建议修复方法"""
    print("\n" + "=" * 80)
    print("可能的问题和解决方案")
    print("=" * 80)

    print("\n1️⃣ 如果文件为空或不存在:")
    print("   → 需要重新训练模型")
    print("   → 运行: python main.py --mode train --algorithms improved_sac")

    print("\n2️⃣ 如果文件损坏:")
    print("   → 检查训练过程是否正常完成")
    print("   → 重新运行训练")

    print("\n3️⃣ 如果没有training_stats键:")
    print("   → 使用的是旧版本trainer.py")
    print("   → 需要使用修复版的trainer.py重新训练")

    print("\n4️⃣ 如果有其他问题:")
    print("   → 请查看上面的诊断信息")
    print("   → 确认数据结构是否符合预期")


if __name__ == "__main__":
    check_all_training_files()
    suggest_fix()
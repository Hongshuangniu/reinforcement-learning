"""
MATLAB数据导出模块 - 修复版（正确提取training_stats）

🔥 核心修复：
1. ✅ 添加对 training_stats 键的检查（trainer.py新增的数据结构）
2. ✅ 优先从 training_stats 获取训练损失数据
3. ✅ 保持对旧版本数据的向后兼容
4. ✅ 改进错误处理和数据验证
"""

import os
import pickle
import numpy as np
from scipy.io import savemat
from typing import Dict
import warnings

warnings.filterwarnings('ignore')


class MatlabDataExporter:
    """MATLAB数据导出器（修复版 - 正确提取training_stats）"""

    def __init__(self, results_dir: str = 'results', output_dir: str = 'matlab_data'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print("=" * 80)
        print("MATLAB数据导出器初始化（修复版 - 支持training_stats）".center(80))
        print("=" * 80)
        print(f"输入目录: {results_dir}")
        print(f"输出目录: {output_dir}")

    def load_training_results(self) -> Dict:
        """加载训练结果"""
        possible_files = [
            'training_results_fixed.pkl',  # 🔥 优先加载修复版
            'training_results_cooling_based.pkl',
            'training_results.pkl'
        ]

        for filename in possible_files:
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                print(f"\n✓ 找到训练结果: {filename}")
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)

                    # 验证数据不为空
                    if data is None:
                        print(f"  ⚠ {filename} 加载后为空，尝试下一个文件")
                        continue

                    print(f"  ✓ 成功加载 {filename}")
                    print(f"  ✓ 数据类型: {type(data)}")
                    print(f"  ✓ 顶层键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                    return data

                except Exception as e:
                    print(f"  ✗ 加载 {filename} 失败: {e}")
                    continue

        raise FileNotFoundError(f"未找到可用的训练结果文件在 {self.results_dir}")

    def load_evaluation_results(self) -> Dict:
        """加载评估结果"""
        possible_files = [
            'evaluation_results_fixed.pkl',
            'evaluation_results_cooling_based.pkl',
            'evaluation_results.pkl'
        ]

        for filename in possible_files:
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                print(f"✓ 找到评估结果: {filename}")
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                return data

        print("⚠ 未找到评估结果文件")
        return None

    def export_training_data(self, algorithm: str, training_results: Dict):
        """
        🔥🔥🔥 修复版：正确提取 training_stats

        数据查找优先级：
        1. training_stats (trainer.py 修复版新增)
        2. agent.get_training_stats() (如果agent是对象)
        3. training_data (旧版本兼容)
        """
        print(f"\n导出 {algorithm} 训练数据...")

        matlab_data = {}

        # ========== 1. Episode奖励 ==========
        episode_rewards = training_results.get('episode_rewards', [])
        if episode_rewards:
            matlab_data['episode_rewards'] = np.array(episode_rewards, dtype=np.float64)
            print(f"  ✓ Episode奖励: {len(episode_rewards)} episodes")
        else:
            print(f"  ⚠ 没有episode奖励数据")

        # ========== 2. 🔥🔥🔥 训练统计数据（核心修复） ==========
        stats = None

        # 🔥 优先级1: 检查 training_stats（trainer.py 修复版）
        if 'training_stats' in training_results:
            print(f"  ✓ 发现 training_stats 键（修复版数据）")
            stats = training_results['training_stats']

        # 优先级2: 从 agent 对象获取
        elif 'agent' in training_results:
            agent = training_results['agent']
            try:
                if hasattr(agent, 'get_training_stats'):
                    print(f"  ✓ 从 agent.get_training_stats() 提取")
                    stats = agent.get_training_stats()
                elif isinstance(agent, dict):
                    print(f"  ✓ agent 是字典，直接使用")
                    stats = agent
            except Exception as e:
                print(f"  ⚠ agent提取失败: {e}")

        # 优先级3: 从 training_data 获取（旧版本兼容）
        if stats is None and 'training_data' in training_results:
            print(f"  ✓ 从 training_data 提取（旧版本）")
            stats = training_results['training_data']

        # 如果都没有，创建空字典
        if stats is None:
            print(f"  ⚠ 未找到训练统计数据，使用空字典")
            stats = {}

        # ========== 3. 提取各种损失和统计数据 ==========

        # Actor损失
        actor_losses = self._extract_field(stats, ['actor_losses', 'actor_loss'])
        if actor_losses:
            matlab_data['actor_losses'] = np.array(actor_losses, dtype=np.float64)
            print(f"  ✓ Actor损失: {len(actor_losses)} 步")
        else:
            print(f"  ⚠ 没有actor损失数据")
            matlab_data['actor_losses'] = np.zeros(10, dtype=np.float64)

        # Critic损失
        critic_losses = self._extract_field(stats, ['critic_losses', 'critic_loss'])
        if critic_losses:
            matlab_data['critic_losses'] = np.array(critic_losses, dtype=np.float64)
            print(f"  ✓ Critic损失: {len(critic_losses)} 步")
        else:
            print(f"  ⚠ 没有critic损失数据")
            matlab_data['critic_losses'] = np.zeros(10, dtype=np.float64)

        # 熵数据
        entropies = self._extract_field(stats, ['entropies', 'entropy'])
        if entropies:
            matlab_data['entropies'] = np.array(entropies, dtype=np.float64)
            print(f"  ✓ 熵数据: {len(entropies)} 步")
        else:
            print(f"  ⚠ 没有熵数据")
            matlab_data['entropies'] = np.ones(10, dtype=np.float64) * 0.5

        # Alpha数据（SAC特有）
        alphas = self._extract_field(stats, ['alphas', 'alpha'])
        if alphas:
            matlab_data['alphas'] = np.array(alphas, dtype=np.float64)
            print(f"  ✓ Alpha数据: {len(alphas)} 步")
        else:
            print(f"  ⚠ 没有alpha数据")
            matlab_data['alphas'] = np.ones(10, dtype=np.float64) * 0.2

        # 最佳指标
        if 'best_cooling_mae' in training_results:
            matlab_data['best_cooling_mae'] = training_results['best_cooling_mae']
        if 'best_reward' in training_results:
            matlab_data['best_reward'] = training_results['best_reward']

        # ========== 4. 保存MATLAB文件 ==========
        output_file = os.path.join(self.output_dir, f'training_{algorithm}.mat')
        savemat(output_file, matlab_data)
        print(f"  ✓ 已保存: training_{algorithm}.mat")
        print(f"    变量数: {len(matlab_data)}")

        # 验证关键数据
        self._verify_training_data(matlab_data)

    def _extract_field(self, data_dict: Dict, field_names: list) -> list:
        """
        从数据字典中提取字段（支持多个可能的字段名）

        Args:
            data_dict: 数据字典
            field_names: 可能的字段名列表

        Returns:
            提取的数据列表，如果都不存在则返回None
        """
        for field_name in field_names:
            if field_name in data_dict:
                data = data_dict[field_name]
                if data and len(data) > 0:
                    return data
        return None

    def _verify_training_data(self, matlab_data: Dict):
        """验证训练数据的完整性"""
        required_fields = ['episode_rewards', 'actor_losses', 'critic_losses']
        optional_fields = ['entropies', 'alphas']

        print(f"\n  📊 数据验证:")
        for field in required_fields:
            if field in matlab_data:
                data_len = len(matlab_data[field])
                print(f"    ✓ {field}: {data_len} 条记录")
            else:
                print(f"    ✗ {field}: 缺失")

        for field in optional_fields:
            if field in matlab_data and len(matlab_data[field]) > 10:  # 大于占位数据
                data_len = len(matlab_data[field])
                print(f"    ✓ {field}: {data_len} 条记录")

    def calculate_total_energy(self, actions: np.ndarray) -> float:
        """计算总能耗"""
        if actions is None or len(actions) == 0:
            return 0.0

        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        # 增压泵能耗
        pump_power = (actions[:, 0] - 2.0) / 3.0 * 100

        # 帕尔贴能耗
        peltier_power = actions[:, 1] * 120

        # 阀门能耗
        valve_power = actions[:, 2] / 100 * 50

        # 总能耗
        total_energy = np.sum(0.5 * pump_power + 0.4 * peltier_power + 0.1 * valve_power)

        return float(total_energy)

    def export_evaluation_data(self, algorithm: str, eval_results: Dict):
        """导出评估数据"""
        print(f"\n导出 {algorithm} 评估数据...")

        matlab_data = {}

        summary = eval_results.get('summary', {})
        metrics = eval_results.get('metrics', {})

        # 降温能力指标
        matlab_data['cooling_mae'] = metrics.get('cooling_mae', 0)
        matlab_data['cooling_rmse'] = metrics.get('cooling_rmse', 0)
        matlab_data['cooling_max_error'] = metrics.get('cooling_max_error', 0)

        # 工业控制指标
        matlab_data['ISE'] = metrics.get('cooling_ise', 0)
        matlab_data['IAE'] = metrics.get('cooling_iae', 0)
        matlab_data['ITAE'] = metrics.get('cooling_itae', 0)

        # 动态性能指标
        matlab_data['settling_time'] = metrics.get('cooling_settling_time', 0)
        matlab_data['peak_overshoot'] = metrics.get('cooling_overshoot', 0)
        matlab_data['steady_state_error'] = metrics.get('cooling_steady_state_error', 0)

        # 控制精度指标
        matlab_data['control_precision_2C'] = metrics.get('cooling_precision_2c', 0)
        matlab_data['control_precision_1C'] = metrics.get('cooling_precision_1c', 0)
        matlab_data['temperature_stability'] = metrics.get('cooling_stability', 0)

        # 能效指标
        matlab_data['total_energy'] = metrics.get('total_energy', 0)
        matlab_data['energy_efficiency_ratio'] = metrics.get('cooling_efficiency', 0)

        # 综合性能指标
        matlab_data['total_performance_index'] = metrics.get('total_cooling_performance_index', 0)
        matlab_data['precision_score'] = metrics.get('precision_score', 0)
        matlab_data['efficiency_score'] = metrics.get('efficiency_score', 0)
        matlab_data['stability_score'] = metrics.get('stability_score', 0)
        matlab_data['speed_score'] = metrics.get('speed_score', 0)

        # RL指标
        matlab_data['avg_reward'] = metrics.get('avg_reward', 0)

        # Episode数据
        episodes = eval_results.get('all_episodes', [])
        if episodes and len(episodes) > 0:
            ep1 = episodes[0]

            # 温度数据
            if 'temperatures' in ep1:
                temps = np.array(ep1['temperatures'], dtype=np.float64)
                matlab_data['episode1_true_temps'] = temps
                print(f"  ✓ 温度数据: {len(temps)} 步")

            # 降温数据
            if 'actual_coolings' in ep1:
                matlab_data['episode1_actual_coolings'] = \
                    np.array(ep1['actual_coolings'], dtype=np.float64)
                print(f"  ✓ 实际降温: {len(ep1['actual_coolings'])} 步")

            if 'target_coolings' in ep1:
                matlab_data['episode1_target_coolings'] = \
                    np.array(ep1['target_coolings'], dtype=np.float64)
                print(f"  ✓ 目标降温: {len(ep1['target_coolings'])} 步")

            # 原始温度（降温前）
            if 'temperatures' in ep1 and 'actual_coolings' in ep1:
                temps = np.array(ep1['temperatures'])
                coolings = np.array(ep1['actual_coolings'])
                original_temps = temps + coolings
                matlab_data['episode1_original_temps'] = original_temps.astype(np.float64)
                print(f"  ✓ 原始温度（降温前）: {len(original_temps)} 步")

            # 动作数据
            if 'actions' in ep1:
                matlab_data['episode1_actions'] = \
                    np.array(ep1['actions'], dtype=np.float64)
                print(f"  ✓ 动作数据: {len(ep1['actions'])} 步")

        # 保存
        output_file = os.path.join(self.output_dir, f'evaluation_{algorithm}.mat')
        savemat(output_file, matlab_data)
        print(f"  ✓ 已保存: evaluation_{algorithm}.mat")
        print(f"    变量数: {len(matlab_data)}")

    def export_all(self):
        """导出所有数据"""
        print("\n" + "=" * 80)
        print("开始导出MATLAB数据（修复版 - 支持training_stats）")
        print("=" * 80)

        try:
            # 1. 加载训练结果
            training_data = self.load_training_results()

            # 🔥 验证数据不为空
            if training_data is None:
                raise ValueError("训练数据加载后为None，请检查训练结果文件")

            if not isinstance(training_data, dict):
                raise TypeError(f"训练数据应该是字典，但得到: {type(training_data)}")

            # 提取results
            if 'results' in training_data:
                results = training_data['results']
                print(f"✓ 从'results'键提取数据")
            else:
                results = training_data
                print(f"✓ 直接使用顶层数据")

            # 再次验证
            if results is None or not isinstance(results, dict):
                raise ValueError(f"results应该是字典，但得到: {type(results)}")

            # 2. 导出训练数据
            print("\n【导出训练数据】")
            for algorithm, algo_results in results.items():
                try:
                    self.export_training_data(algorithm, algo_results)
                except Exception as e:
                    print(f"  ✗ {algorithm} 训练数据导出失败: {e}")
                    import traceback
                    traceback.print_exc()

            # 3. 导出评估数据
            print("\n【导出评估数据】")
            try:
                eval_data = self.load_evaluation_results()
                if eval_data:
                    if 'results' in eval_data:
                        eval_results = eval_data['results']
                    else:
                        eval_results = eval_data

                    for algorithm, algo_eval in eval_results.items():
                        try:
                            self.export_evaluation_data(algorithm, algo_eval)
                        except Exception as e:
                            print(f"  ✗ {algorithm} 评估数据导出失败: {e}")
                            import traceback
                            traceback.print_exc()
            except Exception as e:
                print(f"  ⚠ 评估数据加载失败: {e}")

            print("\n" + "=" * 80)
            print("✓ MATLAB数据导出完成!")
            print("=" * 80)
            print(f"输出目录: {self.output_dir}/")

            print("\n可使用的MATLAB文件:")
            mat_files = [f for f in os.listdir(self.output_dir) if f.endswith('.mat')]
            for file in sorted(mat_files):
                print(f"  • {file}")

            print("\n💡 使用方法:")
            print("  在MATLAB中运行:")
            print("  >> generateImprovedSACDetailedFigures('matlab_data', 'results/figures/ImprovedSAC')")

        except Exception as e:
            print(f"\n✗ 导出失败: {e}")
            import traceback
            traceback.print_exc()


def export_matlab_data(results_dir: str = 'results', output_dir: str = 'matlab_data'):
    """
    便捷函数：导出MATLAB数据（修复版）
    """
    exporter = MatlabDataExporter(results_dir, output_dir)
    exporter.export_all()


if __name__ == "__main__":
    print("=" * 80)
    print("修复说明".center(80))
    print("=" * 80)

    print("\n🔥 核心修复:")
    print("1. ✅ 添加对 'training_stats' 键的检查")
    print("   - 这是 trainer.py 修复版新增的数据结构")
    print("   - 优先从这里提取训练损失、熵、alpha等数据")

    print("\n2. ✅ 数据查找优先级:")
    print("   优先级1: training_stats (修复版)")
    print("   优先级2: agent.get_training_stats() (对象方法)")
    print("   优先级3: training_data (旧版本兼容)")

    print("\n3. ✅ 改进的数据提取:")
    print("   - 使用 _extract_field() 方法支持多个字段名")
    print("   - 添加 _verify_training_data() 验证数据完整性")
    print("   - 更好的错误处理和日志输出")

    print("\n4. ✅ 向后兼容:")
    print("   - 仍然支持旧版本的数据结构")
    print("   - 如果找不到数据，使用占位数据避免MATLAB报错")

    print("\n" + "=" * 80)
    print("使用方法".center(80))
    print("=" * 80)

    print("\n在Python中运行:")
    print("  python compelte_data_lodar.py")

    print("\n或在代码中调用:")
    print("  from compelte_data_lodar import export_matlab_data")
    print("  export_matlab_data(results_dir='results', output_dir='matlab_data')")

    print("\n" + "=" * 80)

    # 运行导出
    export_matlab_data()
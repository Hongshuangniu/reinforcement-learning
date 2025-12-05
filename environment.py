"""
10kV变压器智能冷却控制系统 - 改进环境模拟器 (优化奖励函数版本)
Improved Transformer Cooling Environment Simulator with Optimized Reward

主要改进：
1. ✅ 将奖励范围限制在[-10, 100]之间，避免无界负奖励
2. ✅ 大幅提高温度控制奖励权重（主目标）
3. ✅ 降低能耗惩罚系数（次要目标）
4. ✅ 降低平滑性惩罚系数（辅助目标）
5. ✅ 使用有界的非线性函数（tanh）替代无界线性惩罚
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from config import CONFIG


class ImprovedTransformerCoolingEnv:
    """
    改进的变压器冷却控制环境 - 优化奖励函数版本
    """

    def __init__(self, data: pd.DataFrame, start_idx: int = 0):
        """
        初始化环境

        Args:
            data: 处理后的数据DataFrame
            start_idx: 起始索引
        """
        self.data = data
        self.start_idx = start_idx
        self.current_idx = start_idx
        self.max_steps = CONFIG.env.MAX_STEPS
        self.state_dim = CONFIG.env.STATE_DIM
        self.action_dim = CONFIG.env.ACTION_DIM

        # 环境参数
        # 环境参数（支持自适应目标温度）
        self.target_temp = self._determine_target_temp()
        self.water_temp = CONFIG.env.WATER_TEMP
        self.tank_capacity = CONFIG.env.TANK_CAPACITY
        self.nozzle_count = CONFIG.env.NOZZLE_COUNT
        self.peltier_power = CONFIG.env.PELTIER_POWER

        # 温度区间阈值
        self.temp_low = 55.0
        self.temp_medium = 65.0

        # 状态和动作
        self.current_state = None
        self.last_action = None
        self.step_count = 0

        # 记录
        self.episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'oil_temps': [],
            'predicted_temps': [],
            'ambient_temps': [],
            'reward_components': []
        }

    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_idx = self.start_idx
        self.step_count = 0
        self.last_action = None

        # 清空记录
        self.episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'oil_temps': [],
            'predicted_temps': [],
            'ambient_temps': [],
            'reward_components': []
        }

        # 获取初始状态
        self.current_state = self._get_state()

        return self.current_state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作数组 [pump_pressure, peltier_on, valve_opening]

        Returns:
            next_state, reward, done, info
        """
        # 确保动作在有效范围内
        action = self._clip_action(action)

        # 计算冷却效果
        cooling_effect = self._calculate_cooling_effect(action)

        # 获取当前油温和环境温度
        current_oil_temp = self.data.iloc[self.current_idx]['oil_temp']
        ambient_temp = self.data.iloc[self.current_idx]['ambient_temp']

        # 更新索引
        self.current_idx += 1
        if self.current_idx >= len(self.data):
            self.current_idx = len(self.data) - 1

        next_oil_temp = self.data.iloc[self.current_idx]['oil_temp']

        # 应用冷却效果
        actual_oil_temp = next_oil_temp - cooling_effect + np.random.normal(0, 0.5)

        # 更新数据
        self.data.at[self.data.index[self.current_idx], 'oil_temp'] = actual_oil_temp

        # 获取下一个状态
        next_state = self._get_state()

        # 【核心改进】计算优化的自适应奖励
        reward, reward_info = self._calculate_optimized_reward(
            action, actual_oil_temp, ambient_temp
        )

        # 记录数据
        self.episode_data['states'].append(self.current_state)
        self.episode_data['actions'].append(action)
        self.episode_data['rewards'].append(reward)
        self.episode_data['oil_temps'].append(actual_oil_temp)
        self.episode_data['predicted_temps'].append(next_oil_temp)
        self.episode_data['ambient_temps'].append(ambient_temp)
        self.episode_data['reward_components'].append(reward_info)

        # 更新状态
        self.current_state = next_state
        self.last_action = action.copy()
        self.step_count += 1

        # 判断是否结束
        done = self.step_count >= self.max_steps

        # 附加信息
        info = {
            'oil_temp': actual_oil_temp,
            'predicted_temp': next_oil_temp,
            'ambient_temp': ambient_temp,
            'cooling_effect': cooling_effect,
            'step': self.step_count,
            'temperature_zone': reward_info['zone'],
            **reward_info
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        if self.current_idx >= len(self.data):
            self.current_idx = len(self.data) - 1

        row = self.data.iloc[self.current_idx]

        # 选择数值特征
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        state = row[numeric_cols].values.astype(np.float32)

        # 确保状态维度正确
        if len(state) < CONFIG.env.STATE_DIM:
            state = np.pad(state, (0, CONFIG.env.STATE_DIM - len(state)), 'constant')
        elif len(state) > CONFIG.env.STATE_DIM:
            state = state[:CONFIG.env.STATE_DIM]

        return state

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """裁剪动作到有效范围"""
        clipped_action = np.array([
            np.clip(action[0], CONFIG.env.PUMP_PRESSURE_MIN, CONFIG.env.PUMP_PRESSURE_MAX),
            np.clip(action[1], CONFIG.env.PELTIER_MIN, CONFIG.env.PELTIER_MAX),
            np.clip(action[2], CONFIG.env.VALVE_OPENING_MIN, CONFIG.env.VALVE_OPENING_MAX)
        ])
        return clipped_action

    def _calculate_cooling_effect(self, action: np.ndarray) -> float:
        """计算冷却效果"""
        pump_pressure = action[0]
        peltier_on = action[1]
        valve_opening = action[2]

        # 水冷效果
        water_cooling = (pump_pressure - CONFIG.env.PUMP_PRESSURE_MIN) / \
                        (CONFIG.env.PUMP_PRESSURE_MAX - CONFIG.env.PUMP_PRESSURE_MIN) * \
                        valve_opening / 100.0 * 3.0

        # 帕尔贴冷却效果
        peltier_cooling = peltier_on * self.peltier_power / 100.0 * 1.5

        # 总冷却效果
        total_cooling = water_cooling + peltier_cooling

        return total_cooling

    def _determine_target_temp(self) -> float:
        """根据配置选择目标温度"""
        mode = getattr(CONFIG.env, 'TARGET_MODE', 'fixed')
        if mode == 'adaptive':
            oil_temps = self.data['oil_temp'].astype(float)
            percentile = getattr(CONFIG.env, 'ADAPTIVE_TARGET_PERCENTILE', 55)
            dynamic_target = float(np.percentile(oil_temps, percentile))
            return dynamic_target
        return getattr(CONFIG.env, 'TARGET_TEMP', 60.0)

    def _determine_temperature_zone(self, oil_temp: float) -> str:
        """确定温度区间"""
        if oil_temp < self.temp_low:
            return 'low'
        elif oil_temp < self.temp_medium:
            return 'medium'
        else:
            return 'high'

    def _get_adaptive_weights(self, zone: str) -> Tuple[float, float]:
        """
        获取自适应权重 - 更加强调温度控制

        原始权重：
        - Low Zone:    w_T=0.8,  w_E=0.2
        - Medium Zone: w_T=0.85, w_E=0.15
        - High Zone:   w_T=0.9,  w_E=0.1

        新权重（更加重视温度）：
        - Low Zone:    w_T=0.95, w_E=0.05
        - Medium Zone: w_T=0.97, w_E=0.03
        - High Zone:   w_T=0.99, w_E=0.01
        """
        weight_map = {
            'low': (0.97, 0.03),  # 从(0.95, 0.05)提高
            'medium': (0.98, 0.02),  # 从(0.97, 0.03)提高
            'high': (0.99, 0.01)  # 保持不变
        }
        return weight_map[zone]

    def _get_error_threshold(self, zone: str) -> float:
        """获取误差阈值"""
        thresholds = CONFIG.reward.ERROR_THRESHOLDS
        return thresholds[zone]

    def _calculate_temperature_reward_bounded(self, temp_error: float, threshold: float) -> float:
        """
        计算有界的温度奖励 - 关键改进！

        使用分段函数，但第三段改用tanh保证有界性：
        - ΔT < threshold: 高奖励 100*exp(-0.1*ΔT)
        - threshold ≤ ΔT < 2*threshold: 中等奖励 60*exp(-0.15*ΔT)
        - ΔT ≥ 2*threshold: 有界惩罚 -10*tanh(ΔT/10)

        奖励范围：约 [40, 100] (精确控制) → [10, 40] (可接受) → [-10, 0] (严重偏差)
        """
        if temp_error < threshold:
            # 精确控制区：高奖励，鼓励精细调节
            reward = 100 * np.exp(-0.1 * temp_error)
        elif temp_error < 2 * threshold:
            # 可接受偏差区：中等奖励
            reward = 60 * np.exp(-0.15 * temp_error)
        else:
            # 严重偏差区：有界惩罚（关键改进！）
            # tanh(x/10)将输出限制在[-1, 1]，乘以-10得到[-10, 0]
            reward = -10 * np.tanh(temp_error / 10.0)

        return reward

    def _calculate_energy_penalty_light(self, action: np.ndarray) -> float:
        """
        计算轻量级能耗惩罚 - 降低权重

        原始权重：pump=0.5, peltier=1.0, valve=0.3，系数=0.5
        新权重：pump=0.3, peltier=0.5, valve=0.1，系数=0.05（降低10倍）
        """
        pump_pressure = action[0]
        peltier_on = action[1]
        valve_opening = action[2]

        # 计算各执行器功率
        pump_power = (pump_pressure - CONFIG.env.PUMP_PRESSURE_MIN) / \
                     (CONFIG.env.PUMP_PRESSURE_MAX - CONFIG.env.PUMP_PRESSURE_MIN) * 100
        peltier_power = peltier_on * self.peltier_power
        valve_power = valve_opening / 100 * 50

        # 轻量级加权能耗
        total_energy = (0.3 * pump_power +
                        0.5 * peltier_power +
                        0.1 * valve_power)

        # 返回负的能耗（作为惩罚），系数从0.5降到0.05
        return -total_energy * 0.05

    def _calculate_smoothness_reward_light(self, action: np.ndarray) -> float:
        """
        计算轻量级控制平滑性奖励 - 降低权重

        原始系数：5.0
        新系数：0.5（降低10倍）
        """
        if self.last_action is None:
            return 0.0

        # 计算动作变化的欧氏距离
        action_change = np.linalg.norm(action - self.last_action)

        # 返回负的平滑性惩罚，系数从3.0降到0.5
        smoothness_penalty = -action_change * 0.5

        return smoothness_penalty

    def _calculate_optimized_reward(self, action: np.ndarray,
                                    oil_temp: float,
                                    ambient_temp: float) -> Tuple[float, Dict]:
        """
        【核心方法】计算优化的自适应奖励函数

        改进要点：
        1. 温度奖励：使用有界函数，范围约[-10, 100]
        2. 能耗惩罚：系数从0.5降到0.05（降低10倍）
        3. 平滑性惩罚：系数从3.0降到0.5（降低6倍）
        4. 权重调整：进一步提高温度权重（0.95-0.99）

        单步奖励范围：约[-10, 100]
        Episode回报范围（48步）：约[-500, 4800]

        Args:
            action: 控制动作
            oil_temp: 当前油温
            ambient_temp: 环境温度

        Returns:
            (total_reward, reward_info): 总奖励和详细信息
        """
        # 1. 确定温度区间
        zone = self._determine_temperature_zone(oil_temp)

        # 2. 获取自适应权重（更加重视温度）
        w_T, w_E = self._get_adaptive_weights(zone)

        # 3. 获取误差阈值
        threshold = self._get_error_threshold(zone)

        # 4. 计算温度误差
        temp_error = abs(oil_temp - self.target_temp)

        # 5. 计算各奖励成分
        # 5.1 温度奖励（有界）
        temp_reward = self._calculate_temperature_reward_bounded(temp_error, threshold)

        # 5.2 能耗惩罚（轻量级）
        energy_penalty = self._calculate_energy_penalty_light(action)

        # 5.3 平滑性奖励（轻量级）
        smoothness_reward = self._calculate_smoothness_reward_light(action)

        # 6. 计算总奖励（加权组合）
        # 温度权重0.95-0.99，能耗权重0.01-0.05，平滑性权重固定0.01
        total_reward = (w_T * temp_reward +
                        w_E * energy_penalty +
                        0.01 * smoothness_reward)

        # 7. 详细信息
        reward_info = {
            'zone': zone,
            'w_T': w_T,
            'w_E': w_E,
            'threshold': threshold,
            'temp_error': temp_error,
            'temp_reward': temp_reward,
            'energy_penalty': energy_penalty,
            'smoothness_reward': smoothness_reward,
            'total_reward': total_reward
        }

        return total_reward, reward_info

    def get_episode_data(self) -> Dict:
        """获取episode数据"""
        return self.episode_data

    def render(self):
        """渲染环境"""
        if self.current_state is not None:
            oil_temp = self.data.iloc[self.current_idx]['oil_temp']
            zone = self._determine_temperature_zone(oil_temp)
            print(f"Step: {self.step_count}, Oil Temp: {oil_temp:.2f}°C, "
                  f"Target: {self.target_temp:.2f}°C, Zone: {zone.upper()}")


class MultiEpisodeEnv:
    """多Episode环境管理器"""

    def __init__(self, data: pd.DataFrame, train_dates: list, use_improved=True):
        """
        初始化多Episode环境

        Args:
            data: 完整数据DataFrame
            train_dates: 训练日期列表
            use_improved: 是否使用改进的环境（默认True）
        """
        self.data = data
        self.train_dates = train_dates
        self.envs = {}
        self.use_improved = use_improved

        # 为每个日期创建环境
        for date in train_dates:
            start_idx = self._find_date_index(date)
            if start_idx >= 0:
                self.envs[date] = ImprovedTransformerCoolingEnv(data.copy(), start_idx)

        self.current_date = None
        self.current_env = None

    def _find_date_index(self, date_str: str) -> int:
        """查找日期对应的索引"""
        try:
            target_date = pd.to_datetime(date_str)
            time_diff = abs(self.data.index - target_date)
            idx = time_diff.argmin()
            return idx
        except:
            return -1

    def reset(self, date: Optional[str] = None) -> np.ndarray:
        """重置环境"""
        if date is None:
            date = np.random.choice(self.train_dates)

        self.current_date = date
        self.current_env = self.envs[date]

        return self.current_env.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作"""
        return self.current_env.step(action)

    def get_episode_data(self) -> Dict:
        """获取当前episode数据"""
        return self.current_env.get_episode_data()

    def get_all_envs(self) -> Dict:
        """获取所有环境"""
        return self.envs


if __name__ == "__main__":
    # 测试改进的环境
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("测试优化的奖励函数".center(80))
    print("=" * 80)

    print("\n创建测试数据...")
    n_hours = 48
    time_index = pd.date_range(start='2024-07-01', periods=n_hours, freq='H')
    data = pd.DataFrame(index=time_index)

    # 模拟油温从60°C逐渐升高到70°C（跨越多个温度区间）
    data['oil_temp'] = np.linspace(60, 70, n_hours) + np.random.normal(0, 1, n_hours)
    data['ambient_temp'] = 30 + 5 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    data['humidity'] = 60 + np.random.normal(0, 5, n_hours)
    for i in range(CONFIG.env.STATE_DIM - 3):
        data[f'feature_{i}'] = np.random.randn(n_hours)

    print("✓ 测试数据创建成功")

    print("\n创建优化环境...")
    env = ImprovedTransformerCoolingEnv(data, start_idx=0)
    print("✓ 环境创建成功")

    print("\n执行完整episode测试...")
    state = env.reset()

    rewards = []
    temp_errors = []
    temp_rewards = []
    zones = []

    for i in range(n_hours):
        # 随机动作
        action = np.random.uniform([2.0, 0.0, 0.0], [5.0, 1.0, 100.0])
        next_state, reward, done, info = env.step(action)

        rewards.append(reward)
        temp_errors.append(info['temp_error'])
        temp_rewards.append(info['temp_reward'])
        zones.append(info['temperature_zone'])

        if i < 5 or i >= n_hours - 5:
            print(f"Step {i + 1:2d}: 油温={info['oil_temp']:5.2f}°C, "
                  f"误差={info['temp_error']:5.2f}°C, "
                  f"区间={info['zone']:^6s}, "
                  f"奖励={reward:7.2f} (温度:{info['temp_reward']:6.2f})")
        elif i == 5:
            print("  ...")

        if done:
            break

    print("\n" + "=" * 80)
    print("奖励函数统计".center(80))
    print("=" * 80)
    print(f"Episode总回报:    {sum(rewards):10.2f}")
    print(f"平均单步奖励:    {np.mean(rewards):10.2f}")
    print(f"奖励标准差:      {np.std(rewards):10.2f}")
    print(f"最大单步奖励:    {np.max(rewards):10.2f}")
    print(f"最小单步奖励:    {np.min(rewards):10.2f}")
    print(f"奖励范围:        [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    print("=" * 80)

    print("\n✓ 测试完成！")
    print("\n奖励函数改进要点:")
    print("  1. ✅ 温度奖励使用有界函数，避免无穷负值")
    print("  2. ✅ 能耗惩罚系数降低10倍（0.5 → 0.05）")
    print("  3. ✅ 平滑性惩罚系数降低6倍（3.0 → 0.5）")
    print("  4. ✅ 温度权重提高到0.95-0.99")
    print("  5. ✅ 单步奖励范围控制在约[-10, 100]")
    print("  6. ✅ Episode回报范围合理（约[-500, 4800]）")
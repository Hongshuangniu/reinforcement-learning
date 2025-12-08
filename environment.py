"""
变压器冷却环境 - 完全基于降温能力评价（移除target_temp）

核心改进：
1. ✅ 完全移除target_temp属性
2. ✅ 使用CONFIG中的降温能力参数
3. ✅ 动态降温目标系统
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from config import CONFIG


class ImprovedTransformerCoolingEnv:
    """
    改进的变压器冷却控制环境 - 纯降温能力评价
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

        # 🔥 从CONFIG读取参数
        self.max_steps = CONFIG.env.MAX_STEPS
        self.state_dim = CONFIG.env.STATE_DIM
        self.action_dim = CONFIG.env.ACTION_DIM

        # 环境物理参数（从CONFIG读取）
        self.water_temp = CONFIG.env.WATER_TEMP
        self.tank_capacity = CONFIG.env.TANK_CAPACITY
        self.nozzle_count = CONFIG.env.NOZZLE_COUNT
        self.peltier_power = CONFIG.env.PELTIER_POWER

        # 🔥 温度区间阈值（从CONFIG读取，用于确定降温目标）
        self.temp_low = CONFIG.env.TEMP_LOW
        self.temp_medium = CONFIG.env.TEMP_MEDIUM
        self.temp_high = CONFIG.env.TEMP_HIGH

        # 状态和动作
        self.current_state = None
        self.last_action = None
        self.last_oil_temp = None  # 🔥 记录上一步油温，用于计算降温幅度
        self.step_count = 0

        # 记录
        self.episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'oil_temps': [],
            'cooling_amounts': [],  # 🔥 记录每步实际降温量
            'target_coolings': [],  # 🔥 记录每步目标降温量
            'ambient_temps': [],
            'reward_components': []
        }

    def get_cooling_target(self, oil_temp: float) -> float:
        """
        🔥 根据油温确定降温目标（使用CONFIG）

        Args:
            oil_temp: 当前油温

        Returns:
            目标降温量（°C）
        """
        return CONFIG.env.get_cooling_target(oil_temp)

    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_idx = self.start_idx
        self.step_count = 0
        self.last_action = None
        self.last_oil_temp = None  # 🔥 重置上一步油温

        # 清空记录
        self.episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'oil_temps': [],
            'cooling_amounts': [],
            'target_coolings': [],
            'ambient_temps': [],
            'reward_components': []
        }

        # 获取初始状态
        self.current_state = self._get_state()
        self.last_oil_temp = self.data.iloc[self.current_idx]['oil_temp']

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

        # 🔥 计算实际降温量
        if self.last_oil_temp is not None:
            actual_cooling = self.last_oil_temp - actual_oil_temp
        else:
            actual_cooling = 0

        # 🔥 获取目标降温量（使用CONFIG）
        target_cooling = self.get_cooling_target(
            self.last_oil_temp if self.last_oil_temp is not None else current_oil_temp
        )

        # 获取下一个状态
        next_state = self._get_state()

        # 🔥🔥🔥 核心：计算基于降温能力的奖励（使用CONFIG参数）
        reward, reward_info = self._calculate_cooling_based_reward(
            action, actual_cooling, target_cooling, actual_oil_temp, ambient_temp
        )

        # 记录数据
        self.episode_data['states'].append(self.current_state)
        self.episode_data['actions'].append(action)
        self.episode_data['rewards'].append(reward)
        self.episode_data['oil_temps'].append(actual_oil_temp)
        self.episode_data['cooling_amounts'].append(actual_cooling)  # 🔥 记录实际降温
        self.episode_data['target_coolings'].append(target_cooling)  # 🔥 记录目标降温
        self.episode_data['ambient_temps'].append(ambient_temp)
        self.episode_data['reward_components'].append(reward_info)

        # 更新状态
        self.current_state = next_state
        self.last_action = action.copy()
        self.last_oil_temp = actual_oil_temp  # 🔥 更新上一步油温
        self.step_count += 1

        # 判断是否结束
        done = self.step_count >= self.max_steps

        # 附加信息
        info = {
            'oil_temp': actual_oil_temp,
            'ambient_temp': ambient_temp,
            'cooling_effect': cooling_effect,
            'actual_cooling': actual_cooling,  # 🔥 实际降温量
            'target_cooling': target_cooling,  # 🔥 目标降温量
            'cooling_error': abs(actual_cooling - target_cooling),  # 🔥 降温误差
            'step': self.step_count,
            **reward_info
        }

        return next_state, reward, done, info

    def _calculate_cooling_based_reward(
            self,
            action: np.ndarray,
            actual_cooling: float,
            target_cooling: float,
            oil_temp: float,
            ambient_temp: float
    ) -> Tuple[float, Dict]:
        """
        🔥🔥🔥 核心方法：计算基于降温能力的奖励函数（使用CONFIG参数）

        奖励权重来自CONFIG.reward:
        - 降温效果: 90%
        - 能耗惩罚: 8%
        - 平滑性: 2%

        Args:
            action: 控制动作
            actual_cooling: 实际降温量（°C）
            target_cooling: 目标降温量（°C）
            oil_temp: 当前油温
            ambient_temp: 环境温度

        Returns:
            (total_reward, reward_info)
        """
        # 1. 🔥 降温效果奖励（主要）- 使用CONFIG权重
        cooling_error = abs(actual_cooling - target_cooling)

        if cooling_error < CONFIG.reward.EXCELLENT_COOLING_ERROR:  # 使用CONFIG阈值
            # 非常精确的降温控制
            cooling_reward = 100 * np.exp(-0.5 * cooling_error)  # [90, 100]
        elif cooling_error < CONFIG.reward.GOOD_COOLING_ERROR:  # 使用CONFIG阈值
            # 良好的降温控制
            cooling_reward = 60 * np.exp(-0.3 * cooling_error)  # [20, 60]
        else:
            # 降温偏差较大，有界惩罚
            cooling_reward = -20 * np.tanh(cooling_error / 5.0)  # [-20, 0]

        # 2. 能耗惩罚（次要）
        energy_penalty = self._calculate_energy_penalty_light(action)

        # 3. 平滑性奖励（辅助）
        smoothness_reward = self._calculate_smoothness_reward_light(action)

        # 4. 安全奖励（使用CONFIG阈值）
        if oil_temp > CONFIG.reward.SAFETY_TEMP_THRESHOLD:
            safety_penalty = -10 * (oil_temp - CONFIG.reward.SAFETY_TEMP_THRESHOLD)
        else:
            safety_penalty = 0

        # 5. 加权总奖励（使用CONFIG权重）
        total_reward = (
                CONFIG.reward.COOLING_REWARD_WEIGHT * cooling_reward +
                CONFIG.reward.ENERGY_PENALTY_WEIGHT * energy_penalty +
                CONFIG.reward.SMOOTHNESS_REWARD_WEIGHT * smoothness_reward +
                safety_penalty
        )

        # 详细信息
        reward_info = {
            'cooling_reward': cooling_reward,
            'energy_penalty': energy_penalty,
            'smoothness_reward': smoothness_reward,
            'safety_penalty': safety_penalty,
            'total_reward': total_reward,
            'cooling_error': cooling_error,
            'actual_cooling': actual_cooling,
            'target_cooling': target_cooling
        }

        return total_reward, reward_info

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        if self.current_idx >= len(self.data):
            self.current_idx = len(self.data) - 1

        row = self.data.iloc[self.current_idx]

        # 选择数值特征
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        state = row[numeric_cols].values.astype(np.float32)

        # 确保状态维度正确
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]

        return state

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """裁剪动作到有效范围（使用CONFIG参数）"""
        clipped_action = np.array([
            np.clip(action[0], CONFIG.env.PUMP_PRESSURE_MIN, CONFIG.env.PUMP_PRESSURE_MAX),
            np.clip(action[1], CONFIG.env.PELTIER_MIN, CONFIG.env.PELTIER_MAX),
            np.clip(action[2], CONFIG.env.VALVE_OPENING_MIN, CONFIG.env.VALVE_OPENING_MAX)
        ])
        return clipped_action

    def _calculate_cooling_effect(self, action: np.ndarray) -> float:
        """计算冷却效果（使用CONFIG参数）"""
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

    def _calculate_energy_penalty_light(self, action: np.ndarray) -> float:
        """计算轻量级能耗惩罚（使用CONFIG参数）"""
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

        # 返回负的能耗（作为惩罚）
        return -total_energy * 0.05

    def _calculate_smoothness_reward_light(self, action: np.ndarray) -> float:
        """计算轻量级控制平滑性奖励"""
        if self.last_action is None:
            return 0.0

        # 计算动作变化的欧氏距离
        action_change = np.linalg.norm(action - self.last_action)

        # 返回负的平滑性惩罚
        smoothness_penalty = -action_change * 0.5

        return smoothness_penalty

    def get_episode_data(self) -> Dict:
        """获取episode数据"""
        return self.episode_data

    def render(self):
        """渲染环境"""
        if self.current_state is not None:
            oil_temp = self.data.iloc[self.current_idx]['oil_temp']
            target_cooling = self.get_cooling_target(oil_temp)
            print(f"Step: {self.step_count}, Oil Temp: {oil_temp:.2f}°C, "
                  f"Target Cooling: {target_cooling:.2f}°C (来自CONFIG)")


# 保持向后兼容
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

    def reset(self, date: str = None) -> np.ndarray:
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
    # 测试新的降温能力评价环境
    print("=" * 80)
    print("测试环境模块（完全使用CONFIG，无target_temp）".center(80))
    print("=" * 80)

    print("\n✅ 核心改进：")
    print("  1. ✅ 完全移除target_temp属性")
    print("  2. ✅ 所有参数从CONFIG读取")
    print("  3. ✅ 使用CONFIG.env.get_cooling_target()获取降温目标")
    print("  4. ✅ 奖励权重从CONFIG.reward读取")
    print("  5. ✅ 温度阈值从CONFIG.env读取")

    print("\n📊 CONFIG参数展示：")
    print(f"  MAX_STEPS = {CONFIG.env.MAX_STEPS}")
    print(f"  TEMP_LOW = {CONFIG.env.TEMP_LOW}°C")
    print(f"  TEMP_MEDIUM = {CONFIG.env.TEMP_MEDIUM}°C")
    print(f"  TEMP_HIGH = {CONFIG.env.TEMP_HIGH}°C")
    print(f"  COOLING_REWARD_WEIGHT = {CONFIG.reward.COOLING_REWARD_WEIGHT}")
    print(f"  ENERGY_PENALTY_WEIGHT = {CONFIG.reward.ENERGY_PENALTY_WEIGHT}")
    print(f"  SMOOTHNESS_REWARD_WEIGHT = {CONFIG.reward.SMOOTHNESS_REWARD_WEIGHT}")

    print("\n创建测试数据...")
    n_hours = CONFIG.env.MAX_STEPS
    time_index = pd.date_range(start='2024-07-01', periods=n_hours, freq='H')
    data = pd.DataFrame(index=time_index)

    # 模拟油温从50°C逐渐升高到80°C（跨越所有温度区间）
    data['oil_temp'] = np.linspace(50, 80, n_hours) + np.random.normal(0, 2, n_hours)
    data['ambient_temp'] = 30 + 5 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    data['humidity'] = 60 + np.random.normal(0, 5, n_hours)
    for i in range(CONFIG.env.STATE_DIM - 3):
        data[f'feature_{i}'] = np.random.randn(n_hours)

    print("✓ 测试数据创建成功")

    print("\n创建环境...")
    env = ImprovedTransformerCoolingEnv(data, start_idx=0)
    print("✓ 环境创建成功")
    print(f"✓ 不再有target_temp属性")
    print(f"✓ 使用get_cooling_target()方法动态获取降温目标")

    print("\n执行测试...")
    state = env.reset()

    for i in range(5):
        # 随机动作
        action = np.random.uniform([2.0, 0.0, 0.0], [5.0, 1.0, 100.0])
        next_state, reward, done, info = env.step(action)

        print(f"\nStep {i + 1}:")
        print(f"  油温:         {info['oil_temp']:5.2f}°C")
        print(f"  实际降温:     {info['actual_cooling']:+5.2f}°C")
        print(f"  目标降温:     {info['target_cooling']:5.2f}°C (来自CONFIG)")
        print(f"  降温误差:     {info['cooling_error']:5.2f}°C")
        print(f"  奖励:         {reward:7.2f}")

    print("\n" + "=" * 80)
    print("✓ 环境模块测试完成（完全使用CONFIG，无target_temp）".center(80))
    print("=" * 80)

    print("\n📋 降温目标规则测试（来自CONFIG）:")
    test_temps = [50, 60, 70, 80]
    for temp in test_temps:
        target = env.get_cooling_target(temp)
        print(f"  油温 {temp}°C → 目标降温 {target}°C")
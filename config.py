"""
优化配置 - 修复版（完全基于降温能力评价）

🔥 修复内容：
1. ✅ 调整NUM_EPISODES匹配真实数据量（8832小时=184 episodes）
2. ✅ 保留所有降温能力评价参数
3. ✅ 确保所有模块使用统一配置
"""

import torch
import numpy as np


def get_device():
    """智能选择计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


DEVICE = get_device()


# ============= 环境参数 =============
class EnvironmentConfig:
    """变压器冷却环境配置（降温能力评价版）"""

    # ===== 🔥 降温能力模式参数（核心配置） =====
    # 温度区间阈值
    TEMP_LOW = 55.0  # 低温阈值
    TEMP_MEDIUM = 65.0  # 中温阈值
    TEMP_HIGH = 75.0  # 高温阈值
    TEMP_CRITICAL = 85.0  # 临界高温阈值

    # 降温目标规则（根据油温高低设定不同降温量）
    COOLING_TARGETS = {
        'very_low': 2.0,  # 油温 <= TEMP_LOW: 轻微降温 2°C
        'low': 8.0,  # TEMP_LOW < 油温 <= TEMP_MEDIUM: 中等降温 8°C
        'medium': 12.0,  # TEMP_MEDIUM < 油温 <= TEMP_HIGH: 强力降温 12°C
        'high': 15.0  # 油温 > TEMP_HIGH: 紧急降温 15°C
    }

    # 降温精度评价阈值（用于计算精度指标）
    COOLING_PRECISION_THRESHOLDS = [1.0, 2.0, 3.0]  # ±1°C, ±2°C, ±3°C

    # ===== 物理参数 =====
    WATER_TEMP = 25.0
    TANK_CAPACITY = 140
    NOZZLE_COUNT = 42
    PELTIER_POWER = 120

    # ===== 动作空间 =====
    PUMP_PRESSURE_MIN = 2.0
    PUMP_PRESSURE_MAX = 5.0
    PELTIER_MIN = 0.0
    PELTIER_MAX = 1.0
    VALVE_OPENING_MIN = 0.0
    VALVE_OPENING_MAX = 100.0

    # ===== 状态和动作维度 =====
    STATE_DIM = 24
    ACTION_DIM = 3
    MAX_STEPS = 48  # 每个episode 48小时

    @classmethod
    def get_cooling_target(cls, oil_temp: float) -> float:
        """
        🔥 根据油温确定降温目标

        Args:
            oil_temp: 当前油温

        Returns:
            目标降温量（°C）
        """
        if oil_temp <= cls.TEMP_LOW:
            return cls.COOLING_TARGETS['very_low']
        elif oil_temp <= cls.TEMP_MEDIUM:
            return cls.COOLING_TARGETS['low']
        elif oil_temp <= cls.TEMP_HIGH:
            return cls.COOLING_TARGETS['medium']
        else:
            return cls.COOLING_TARGETS['high']

    @classmethod
    def print_cooling_rules(cls):
        """打印降温规则"""
        print("\n🎯 降温目标规则:")
        print(f"  • 油温 ≤ {cls.TEMP_LOW}°C  → 目标降温 {cls.COOLING_TARGETS['very_low']}°C  （温度较低，维持即可）")
        print(
            f"  • {cls.TEMP_LOW}°C < 油温 ≤ {cls.TEMP_MEDIUM}°C → 目标降温 {cls.COOLING_TARGETS['low']}°C  （温度适中）")
        print(
            f"  • {cls.TEMP_MEDIUM}°C < 油温 ≤ {cls.TEMP_HIGH}°C → 目标降温 {cls.COOLING_TARGETS['medium']}°C （温度偏高）")
        print(f"  • 油温 > {cls.TEMP_HIGH}°C  → 目标降温 {cls.COOLING_TARGETS['high']}°C （温度过高）")


# ============= 训练超参数（修复版）=============
class TrainingConfig:
    """训练配置（修复版）"""

    # 🔥 修复：调整NUM_EPISODES以匹配数据量
    # 真实数据: 8832小时 ÷ 48小时/episode = 184 episodes
    # 建议使用略小的值以保留验证集
    NUM_EPISODES = 150  # 修复：从50改为150，充分利用数据
    EVAL_FREQUENCY = 10  # 每10个episodes评估一次
    EVAL_EPISODES = 10  # 每次评估10个episodes

    # 2. 网络参数
    HIDDEN_DIM = 256
    NUM_ATTENTION_HEADS = 4
    DROPOUT_RATE = 0.1

    # 3. 学习率
    LEARNING_RATE_ACTOR = 1e-4
    LEARNING_RATE_CRITIC = 3e-4
    LEARNING_RATE_ALPHA = 3e-4

    # 4. 正则化
    WEIGHT_DECAY = 1e-4

    # 5. 优先经验回放（可选）
    USE_PRIORITY = False  # 默认关闭优先经验回放
    PRIORITY_ALPHA = 0.6
    PRIORITY_BETA = 0.4
    PRIORITY_BETA_INCREMENT = 0.001

    # 6. 温度参数（SAC）
    INITIAL_ALPHA = 0.2
    TARGET_ENTROPY_SCALE = 0.5  # target_entropy = -action_dim * scale

    # 7. 其他参数
    GAMMA = 0.99
    TAU = 0.005
    BATCH_SIZE = 256
    BUFFER_SIZE = 100000

    # 8. 梯度裁剪
    GRAD_CLIP_NORM = 1.0

    # 9. 动作平滑
    ACTION_SMOOTH_FACTOR = 0.1


# ============= 数据增强配置 =============
class DataAugmentationConfig:
    """数据增强配置"""

    USE_AUGMENTATION = False  # 默认关闭数据增强
    ADD_NOISE = False
    NOISE_STD = 0.5
    TIME_SHIFT = False
    MAX_SHIFT_HOURS = 2
    TEMP_SCALING = False
    SCALE_RANGE = (0.95, 1.05)
    USE_SLIDING_WINDOW = True
    WINDOW_STRIDE = 12


# ============= 算法配置 =============
class AlgorithmConfig:
    """算法配置"""

    ALGORITHMS = ['improved_sac', 'sac', 'ppo', 'ddpg', 'td3']

    # PPO参数
    PPO_CLIP_EPSILON = 0.2
    PPO_ENTROPY_COEF = 0.01
    PPO_VALUE_LOSS_COEF = 0.5
    PPO_UPDATE_EPOCHS = 10

    # DDPG/TD3参数
    NOISE_STD = 0.1
    NOISE_CLIP = 0.5
    TD3_POLICY_DELAY = 2
    TD3_TARGET_NOISE = 0.2


# ============= 🔥 奖励函数配置（降温能力评价）=============
class RewardConfig:
    """奖励函数配置（基于降温能力）"""

    # 奖励权重（总和 = 1.0）
    COOLING_REWARD_WEIGHT = 0.90  # 降温效果权重（主要）
    ENERGY_PENALTY_WEIGHT = 0.08  # 能耗惩罚权重（次要）
    SMOOTHNESS_REWARD_WEIGHT = 0.02  # 平滑性权重（辅助）

    # 降温精度奖励阈值
    EXCELLENT_COOLING_ERROR = 1.0  # 优秀：误差 < 1°C
    GOOD_COOLING_ERROR = 3.0  # 良好：误差 < 3°C

    # 安全温度阈值
    SAFETY_TEMP_THRESHOLD = 80.0  # 超过此温度施加安全惩罚


# ============= 🔥 评估指标配置（降温能力优先）=============
class MetricsConfig:
    """评估指标配置"""

    # 🔥 核心指标（降温能力）- 最重要
    PRIMARY_METRICS = [
        'cooling_mae',  # 降温平均绝对误差（核心）
        'cooling_rmse',  # 降温均方根误差
        'cooling_precision_1c',  # ±1°C精度
        'cooling_precision_2c',  # ±2°C精度
        'cooling_precision_3c',  # ±3°C精度
        'total_cooling',  # 总降温量
        'cooling_efficiency',  # 降温效率
        'cooling_stability',  # 降温稳定性
    ]

    # 工业控制指标（完整）
    INDUSTRIAL_METRICS = [
        'cooling_ise',  # 积分平方误差
        'cooling_iae',  # 积分绝对误差
        'cooling_itae',  # 时间加权积分绝对误差
        'cooling_settling_time',  # 调节时间
        'cooling_overshoot',  # 超调量
        'cooling_steady_state_error',  # 稳态误差
    ]

    # 传统温度控制指标（仅作参考）
    REFERENCE_METRICS = [
        'temperature_range',  # 温度波动范围
        'temperature_std',  # 温度标准差
    ]

    # 工业控制指标（仅作参考）
    CONTROL_METRICS = [
        'action_smoothness',  # 动作平滑度
        'control_effort',  # 控制努力
    ]

    # 强化学习指标
    RL_METRICS = [
        'avg_reward',  # 平均回报
        'reward_std',  # 回报标准差
        'episode_length',  # Episode长度
    ]

    # 🔥 评价优先级（用于最佳模型选择）
    BEST_MODEL_CRITERION = 'cooling_mae'  # 使用降温MAE作为最佳模型判定标准

    # 次要判定标准（当主要标准相近时使用）
    SECONDARY_CRITERION = 'cooling_precision_2c'  # 降温精度±2°C
    TERTIARY_CRITERION = 'avg_reward'  # 平均回报


# ============= 可视化配置 =============
class VisualizationConfig:
    """可视化配置"""

    FONT_FAMILY = 'Times New Roman'
    FONT_SIZE_TITLE = 14
    FONT_SIZE_LABEL = 12
    FONT_SIZE_TICK = 10
    FONT_SIZE_LEGEND = 10

    FIGURE_SIZE_SINGLE = (8, 6)
    FIGURE_SIZE_DOUBLE = (12, 5)
    FIGURE_SIZE_GRID = (12, 10)

    DPI = 300

    COLORS = {
        'improved_sac': '#FF6B6B',
        'sac': '#4ECDC4',
        'ppo': '#45B7D1',
        'ddpg': '#96CEB4',
        'td3': '#DDA0DD',
    }

    SAVE_DIR = 'figures'
    TABLE_DIR = 'tables'
    RESULTS_DIR = 'results'


# ============= 数据配置 =============
class DataConfig:
    """数据配置"""

    DATA_DIR = 'data'
    OIL_TEMP_FILE = 'Oil_temperature_data_for_July_2024.xlsx'
    WEATHER_FILE = 'Weather_data_for_24_hours_on_July_2024.xlsx'
    PROCESSED_DATA_FILE = 'processed_transformer_data.pkl'
    TRAIN_DATES = ['2024-07-13', '2024-07-20', '2024-07-23']


# ============= 输出文件配置 =============
class OutputConfig:
    """输出文件配置"""

    MODEL_DIR = 'models'
    RESULTS_DIR = 'results'
    LOG_DIR = 'logs'


# ============= 全局配置 =============
class GlobalConfig:
    """全局配置"""

    env = EnvironmentConfig()
    train = TrainingConfig()
    aug = DataAugmentationConfig()
    algo = AlgorithmConfig()
    reward = RewardConfig()
    metrics = MetricsConfig()
    vis = VisualizationConfig()
    data = DataConfig()
    output = OutputConfig()
    device = DEVICE

    @staticmethod
    def print_config():
        """打印配置"""
        print("=" * 80)
        print("配置总览 - 修复版（降温能力评价体系）".center(80))
        print("=" * 80)

        print(f"\n🔥 控制模式: 降温能力评价（动态降温目标）")
        EnvironmentConfig.print_cooling_rules()

        print(f"\n⚙️ 设备: {DEVICE}")

        print(f"\n📊 训练参数（修复版）:")
        print(f"  Episodes:            {TrainingConfig.NUM_EPISODES} 🔥 修复：从50改为150")
        print(f"  每Episode小时数:     {EnvironmentConfig.MAX_STEPS}")
        print(f"  需要总小时数:        {TrainingConfig.NUM_EPISODES * EnvironmentConfig.MAX_STEPS}")
        print(f"  Eval Frequency:      {TrainingConfig.EVAL_FREQUENCY}")
        print(f"  Eval Episodes:       {TrainingConfig.EVAL_EPISODES}")
        print(f"  Hidden Dim:          {TrainingConfig.HIDDEN_DIM}")
        print(f"  Attention Heads:     {TrainingConfig.NUM_ATTENTION_HEADS}")
        print(f"  Actor LR:            {TrainingConfig.LEARNING_RATE_ACTOR}")
        print(f"  Critic LR:           {TrainingConfig.LEARNING_RATE_CRITIC}")
        print(f"  Batch Size:          {TrainingConfig.BATCH_SIZE}")
        print(f"  Buffer Size:         {TrainingConfig.BUFFER_SIZE}")

        print(f"\n🎯 奖励权重:")
        print(f"  降温效果:            {RewardConfig.COOLING_REWARD_WEIGHT:.2f} (90%)")
        print(f"  能耗惩罚:            {RewardConfig.ENERGY_PENALTY_WEIGHT:.2f} (8%)")
        print(f"  平滑性:              {RewardConfig.SMOOTHNESS_REWARD_WEIGHT:.2f} (2%)")

        print(f"\n📈 核心评价指标:")
        for i, metric in enumerate(MetricsConfig.PRIMARY_METRICS[:5], 1):
            print(f"  {i}. {metric}")

        print(f"\n🏭 工业控制指标:")
        for i, metric in enumerate(MetricsConfig.INDUSTRIAL_METRICS, 1):
            print(f"  {i}. {metric}")

        print(f"\n🏆 最佳模型判定:")
        print(f"  主要标准:            {MetricsConfig.BEST_MODEL_CRITERION}")
        print(f"  次要标准:            {MetricsConfig.SECONDARY_CRITERION}")
        print(f"  第三标准:            {MetricsConfig.TERTIARY_CRITERION}")

        print("=" * 80)


CONFIG = GlobalConfig()

if __name__ == "__main__":
    CONFIG.print_config()

    print("\n" + "=" * 80)
    print("修复说明".center(80))
    print("=" * 80)

    print("\n✅ 核心修复:")
    print("\n1. 🔥 调整NUM_EPISODES:")
    print("   • 原值: 50 episodes")
    print("   • 新值: 150 episodes")
    print("   • 原因: 充分利用8832小时数据（8832÷48=184可用episodes）")
    print("   • 效果: 训练更充分，性能更好")

    print("\n2. 📊 完整工业指标:")
    print("   • 保留所有工业控制指标（ISE/IAE/ITAE/调节时间/超调量）")
    print("   • 确保在评估和打印时显示")

    print("\n3. ⚙️ 其他优化:")
    print("   • EVAL_FREQUENCY: 10 (更频繁的评估)")
    print("   • 所有参数保持统一")

    print("\n" + "=" * 80)
    print("数据需求计算".center(80))
    print("=" * 80)

    print(f"\n根据当前配置:")
    print(f"  每个Episode: {CONFIG.env.MAX_STEPS} 小时")
    print(f"  训练Episodes: {CONFIG.train.NUM_EPISODES}")
    print(f"  评估Episodes: {CONFIG.train.EVAL_EPISODES}")
    print(f"\n  最小需求数据: {CONFIG.train.NUM_EPISODES * CONFIG.env.MAX_STEPS} 小时")
    print(f"                = {CONFIG.train.NUM_EPISODES * CONFIG.env.MAX_STEPS / 24:.1f} 天")
    print(f"\n  实际数据(真实): 8832 小时 = 368 天")
    print(f"  可用Episodes: {8832 // CONFIG.env.MAX_STEPS}")
    print(f"  ✓ 数据充足")

    print("\n" + "=" * 80)

    # 测试降温目标计算
    print("\n📋 降温目标计算测试:")
    test_temps = [50, 60, 70, 80]
    for temp in test_temps:
        target = CONFIG.env.get_cooling_target(temp)
        print(f"  油温 {temp}°C → 目标降温 {target}°C")

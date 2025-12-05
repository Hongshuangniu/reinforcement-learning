"""
优化配置 - 针对真实数据的改进版本（移除早停机制）

关键改进：
1. 🎯 更激进的正则化防止过拟合
2. 🎯 数据增强策略
3. 🎯 更保守的学习率
4. ✅ 移除早停机制，完整训练
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
    """变压器冷却环境配置"""
    TARGET_TEMP = 50.0
    WATER_TEMP = 25.0
    TANK_CAPACITY = 140
    NOZZLE_COUNT = 42
    PELTIER_POWER = 120

    TEMP_LOW = 55.0
    TEMP_MEDIUM = 65.0

    PUMP_PRESSURE_MIN = 2.0
    PUMP_PRESSURE_MAX = 5.0
    PELTIER_MIN = 0.0
    PELTIER_MAX = 1.0
    VALVE_OPENING_MIN = 0.0
    VALVE_OPENING_MAX = 100.0

    STATE_DIM = 24
    ACTION_DIM = 3
    MAX_STEPS = 48


# ============= 训练超参数（真实数据优化版）=============
class TrainingConfig:
    """训练配置 - 针对真实数据优化"""

    # 1. 训练Episodes（根据数据量调整）
    NUM_EPISODES = 50  # 🔥 从200降到100，减少过拟合风险
    EVAL_FREQUENCY = 5  # 🔥 更频繁评估

    # 2. 网络参数（大幅简化避免过拟合）
    HIDDEN_DIM = 64  # 🔥🔥🔥 从128降到64（减半）
    NUM_ATTENTION_HEADS = 1  # 保持简单
    DROPOUT_RATE = 0.4  # 🔥 从0.35提高到0.4

    # 3. 学习率（更保守）
    LEARNING_RATE_ACTOR = 1e-5  # 🔥🔥 从5e-5降到1e-5
    LEARNING_RATE_CRITIC = 1e-5  # 🔥🔥 从5e-5降到1e-5
    LEARNING_RATE_ALPHA = 1e-4  # 🔥 从2e-4降到1e-4

    # 4. L2正则化（更强）
    WEIGHT_DECAY = 1e-3  # 🔥🔥 从5e-4提高到1e-3（翻倍）

    # 5. 优先经验回放
    PRIORITY_ALPHA = 0.2
    PRIORITY_BETA = 0.3
    PRIORITY_BETA_INCREMENT = 0.0001

    # 6. 温度参数
    INITIAL_ALPHA = 0.05  # 🔥 从0.1降到0.05（更保守探索）
    ALPHA_SCHEDULE = 'adaptive'

    # 7. 其他参数
    GAMMA = 0.99
    TAU = 0.005
    BATCH_SIZE = 128  # 🔥 从256降到128（减小批次）
    BUFFER_SIZE = 50000  # 🔥 从100000降到50000

    # 8. 梯度裁剪
    GRAD_CLIP_NORM = 0.5

    # 9. 动作平滑
    ACTION_SMOOTH_FACTOR = 0.2  # 🔥 从0.1提高到0.2


# ============= 数据增强配置 =============
class DataAugmentationConfig:
    """数据增强配置 - 扩充训练样本"""

    # 是否使用数据增强
    USE_AUGMENTATION = True

    # 噪声增强
    ADD_NOISE = True
    NOISE_STD = 0.5  # 油温噪声标准差（°C）

    # 时间偏移
    TIME_SHIFT = True
    MAX_SHIFT_HOURS = 2

    # 温度缩放
    TEMP_SCALING = True
    SCALE_RANGE = (0.95, 1.05)  # ±5%

    # 滑动窗口（生成更多训练episode）
    USE_SLIDING_WINDOW = True
    WINDOW_STRIDE = 12  # 每12小时开始一个新episode


# ============= 算法配置 =============
class AlgorithmConfig:
    """算法配置"""
    ALGORITHMS = ['improved_sac', 'sac', 'ppo', 'ddpg', 'td3']

    PPO_CLIP_EPSILON = 0.2
    PPO_ENTROPY_COEF = 0.01
    PPO_VALUE_LOSS_COEF = 0.5
    PPO_UPDATE_EPOCHS = 10

    NOISE_STD = 0.1
    NOISE_CLIP = 0.5
    TD3_POLICY_DELAY = 2
    TD3_TARGET_NOISE = 0.2


# ============= 奖励函数配置 =============
class RewardConfig:
    """奖励函数配置"""
    TEMP_WEIGHTS = {
        'low': (0.97, 0.03),
        'medium': (0.98, 0.02),
        'high': (0.99, 0.01)
    }

    ERROR_THRESHOLDS = {
        'low': 4.0,
        'medium': 3.0,
        'high': 2.0
    }

    COMFORT_COEF = 200
    ENERGY_COEF = 0.5
    SMOOTHNESS_COEF = 0.5


# ============= 评估指标配置 =============
class MetricsConfig:
    """评估指标配置"""
    METRICS = [
        'MAE', 'RMSE', 'MAPE', 'R2', 'MaxAE',
        'MeanReward', 'ConvergenceSpeed', 'Stability',
        'PolicyEntropy', 'ActionSmoothness'
    ]


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
        print("配置总览 - 真实数据优化版（完整训练）".center(80))
        print("=" * 80)
        print(f"设备: {DEVICE}")
        print(f"训练Episodes: {TrainingConfig.NUM_EPISODES} 🔥")
        print(f"\n网络参数:")
        print(f"  Hidden Dim: {TrainingConfig.HIDDEN_DIM} 🔥🔥 (减半)")
        print(f"  Dropout: {TrainingConfig.DROPOUT_RATE} 🔥")
        print(f"\n学习参数:")
        print(f"  Learning Rate: {TrainingConfig.LEARNING_RATE_ACTOR} 🔥🔥")
        print(f"  Weight Decay: {TrainingConfig.WEIGHT_DECAY} 🔥🔥")
        print(f"  Batch Size: {TrainingConfig.BATCH_SIZE} 🔥")
        print(f"\n数据增强:")
        print(f"  使用增强: {DataAugmentationConfig.USE_AUGMENTATION}")
        print(f"  滑动窗口: {DataAugmentationConfig.USE_SLIDING_WINDOW}")
        print("=" * 80)


CONFIG = GlobalConfig()

if __name__ == "__main__":
    CONFIG.print_config()

    print("\n" + "=" * 80)
    print("关键改进说明".center(80))
    print("=" * 80)
    print("\n🎯 针对真实数据的优化策略:")
    print("\n1. 🔥🔥 网络简化（避免过拟合）:")
    print("   - Hidden Dim: 128 → 64 (减半)")
    print("   - Dropout: 0.35 → 0.4 (更激进)")
    print("   - Batch Size: 256 → 128 (减小)")

    print("\n2. 🔥🔥 学习率大幅降低:")
    print("   - Actor/Critic LR: 5e-5 → 1e-5 (降低5倍)")
    print("   - Weight Decay: 5e-4 → 1e-3 (翻倍)")

    print("\n3. 🔥 训练策略改进:")
    print("   - Episodes: 200 → 100")
    print("   - 评估: 每10轮 → 每5轮 (更频繁)")
    print("   - ✅ 无早停机制，完整训练所有episodes")

    print("\n4. 🔥 数据增强:")
    print("   - 添加高斯噪声")
    print("   - 时间偏移")
    print("   - 温度缩放")
    print("   - 滑动窗口生成更多样本")

    print("\n预期效果:")
    print("  ✅ 训练回报: -400 到 -450（健康范围）")
    print("  ✅ 评估回报: 接近训练回报（泛化良好）")
    print("  ✅ 评估MAE: < 0.5°C（优于其他算法）")
    print("=" * 80)
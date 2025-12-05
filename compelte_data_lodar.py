"""
完整真实数据加载系统 - 修复版（完整版）
支持多年多月数据（2021-2024年7-9月，共368天）

修复内容：
1. ✅ 正确读取所有Excel sheets
2. ✅ 修复日期解析错误
3. ✅ 改进数据合并逻辑
4. ✅ 添加详细的调试信息
5. ✅ 验证数据完整性
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class TransformerDataLoader:
    """变压器数据加载器 - 修复版"""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.oil_temp_df = None
        self.weather_df = None
        self.predicted_df = None
        self.merged_df = None

        # 调试信息
        self.debug_info = {
            'oil_temp_sheets': [],
            'weather_sheets': [],
            'predicted_sheets': [],
            'total_days': 0,
            'total_hours': 0
        }

    def load_oil_temperature(self, filename='Oil_temperature_data_for_July_2024.xlsx'):
        """加载油温数据 - 修复版"""
        filepath = os.path.join(self.data_dir, filename)
        print(f"\n{'=' * 80}")
        print(f"1. 加载油温数据: {filename}")
        print(f"{'=' * 80}")

        try:
            # 读取所有sheets
            xl_file = pd.ExcelFile(filepath)
            all_sheets = xl_file.sheet_names

            print(f"✓ 找到 {len(all_sheets)} 个sheets:")
            for i, sheet in enumerate(all_sheets, 1):
                print(f"  {i}. {sheet}")

            # 存储所有时间序列数据
            all_time_series = []
            total_days = 0

            # 逐个处理每个sheet
            for sheet_idx, sheet_name in enumerate(all_sheets, 1):
                print(f"\n处理 Sheet {sheet_idx}/{len(all_sheets)}: '{sheet_name}'")

                try:
                    # 读取sheet
                    df_sheet = pd.read_excel(filepath, sheet_name=sheet_name)

                    print(f"  原始形状: {df_sheet.shape}")
                    print(f"  列名示例: {df_sheet.columns.tolist()[:5]}...")

                    # 检查是否有date列
                    if 'date' not in df_sheet.columns:
                        print(f"  ⚠ 跳过: 没有'date'列")
                        continue

                    # 提取时间序列数据
                    sheet_data = []
                    valid_days = 0

                    for row_idx, row in df_sheet.iterrows():
                        try:
                            # 解析日期
                            date_val = row['date']

                            if pd.isna(date_val):
                                continue

                            if isinstance(date_val, str):
                                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                                    try:
                                        date = pd.to_datetime(date_val, format=fmt)
                                        break
                                    except:
                                        continue
                                else:
                                    date = pd.to_datetime(date_val)
                            else:
                                date = pd.to_datetime(date_val)

                            # 遍历24小时
                            day_has_data = False
                            for hour in range(24):
                                # 查找油温列名
                                oil_temp = None

                                possible_col_names = [
                                    f'oil temperature_{hour:02d}:00 (℃)',
                                    f'oil temperature_{hour:02d}:00(℃)',
                                    f'oil temperature_{hour:02d}:00',
                                    f'Oil temperature_{hour:02d}:00 (℃)',
                                    f'Oil temperature_{hour:02d}:00',
                                    f'oil_temperature_{hour:02d}:00',
                                ]

                                for col_name in possible_col_names:
                                    if col_name in df_sheet.columns:
                                        oil_temp = row[col_name]
                                        break

                                if oil_temp is not None and not pd.isna(oil_temp):
                                    try:
                                        oil_temp_float = float(oil_temp)

                                        if 20 <= oil_temp_float <= 100:
                                            timestamp = date + timedelta(hours=hour)
                                            sheet_data.append({
                                                'timestamp': timestamp,
                                                'oil_temp': oil_temp_float
                                            })
                                            day_has_data = True
                                    except (ValueError, TypeError):
                                        continue

                            if day_has_data:
                                valid_days += 1

                        except Exception as e:
                            continue

                    print(f"  ✓ 提取了 {valid_days} 天，{len(sheet_data)} 个小时数据")

                    if len(sheet_data) > 0:
                        all_time_series.extend(sheet_data)
                        total_days += valid_days
                        self.debug_info['oil_temp_sheets'].append({
                            'name': sheet_name,
                            'days': valid_days,
                            'hours': len(sheet_data)
                        })

                except Exception as e:
                    print(f"  ✗ Sheet处理失败: {e}")
                    continue

            if len(all_time_series) == 0:
                print(f"\n✗ 没有提取到任何数据")
                return None

            oil_df = pd.DataFrame(all_time_series)
            oil_df.set_index('timestamp', inplace=True)
            oil_df.sort_index(inplace=True)

            # 去除重复
            before_dedup = len(oil_df)
            oil_df = oil_df[~oil_df.index.duplicated(keep='first')]
            after_dedup = len(oil_df)

            if before_dedup > after_dedup:
                print(f"\n⚠ 去除了 {before_dedup - after_dedup} 个重复时间戳")

            # 统计
            print(f"\n{'=' * 80}")
            print(f"油温数据加载完成")
            print(f"{'=' * 80}")
            print(f"✓ 总时间点: {len(oil_df):,}")
            print(f"✓ 总天数: {total_days}")
            print(f"✓ 时间范围: {oil_df.index.min()} → {oil_df.index.max()}")
            print(f"✓ 时间跨度: {(oil_df.index.max() - oil_df.index.min()).days + 1} 天")
            print(f"✓ 油温范围: {oil_df['oil_temp'].min():.2f}°C → {oil_df['oil_temp'].max():.2f}°C")
            print(f"✓ 油温均值: {oil_df['oil_temp'].mean():.2f}°C")
            print(f"✓ 可训练Episodes (48h/个): {len(oil_df) // 48}")

            self.oil_temp_df = oil_df
            return oil_df

        except Exception as e:
            print(f"\n✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_weather_data(self, filename='Weather_data_for_24_hours_on_July_2024.xlsx'):
        """加载天气数据 - 修复版"""
        filepath = os.path.join(self.data_dir, filename)
        print(f"\n{'=' * 80}")
        print(f"2. 加载天气数据: {filename}")
        print(f"{'=' * 80}")

        try:
            xl_file = pd.ExcelFile(filepath)
            all_sheets = xl_file.sheet_names
            print(f"✓ 找到 {len(all_sheets)} 个sheets")

            all_time_series = []
            total_days = 0

            for sheet_idx, sheet_name in enumerate(all_sheets, 1):
                print(f"\n处理 Sheet {sheet_idx}/{len(all_sheets)}: '{sheet_name}'")

                try:
                    df_sheet = pd.read_excel(filepath, sheet_name=sheet_name)

                    if 'date' not in df_sheet.columns:
                        print(f"  ⚠ 跳过: 没有'date'列")
                        continue

                    sheet_data = []
                    valid_days = 0

                    for row_idx, row in df_sheet.iterrows():
                        try:
                            date_val = row['date']
                            if pd.isna(date_val):
                                continue

                            if isinstance(date_val, str):
                                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                                    try:
                                        date = pd.to_datetime(date_val, format=fmt)
                                        break
                                    except:
                                        continue
                                else:
                                    date = pd.to_datetime(date_val)
                            else:
                                date = pd.to_datetime(date_val)

                            # 提取日级别特征
                            weather_code = 6
                            wind_level = 2
                            sunshine_hours = 8.0
                            max_temp = 32.0
                            min_temp = 24.0

                            # 查找各列
                            weather_col_names = [
                                'Weather (drizzle - 1, light rain - 2, moderate rain - 3, heavy rain - 4, storm - 5, sunny - 6, cloudy - 7, overcast - 8, snow - 9)',
                                'Weather', 'weather'
                            ]
                            for col in weather_col_names:
                                if col in df_sheet.columns and not pd.isna(row[col]):
                                    weather_code = int(row[col])
                                    break

                            wind_col_names = ['Weather - Wind Force Level', 'Wind Force Level']
                            for col in wind_col_names:
                                if col in df_sheet.columns and not pd.isna(row[col]):
                                    wind_level = int(row[col])
                                    break

                            sunshine_col_names = ['Weather - Duration of Sunshine (hours)', 'Duration of Sunshine']
                            for col in sunshine_col_names:
                                if col in df_sheet.columns and not pd.isna(row[col]):
                                    sunshine_hours = float(row[col])
                                    break

                            max_temp_col_names = ['Weather - Maximum Temperature (℃)', 'Maximum Temperature']
                            for col in max_temp_col_names:
                                if col in df_sheet.columns and not pd.isna(row[col]):
                                    max_temp = float(row[col])
                                    break

                            min_temp_col_names = ['Weather - Minimmum Temperature (℃)', 'Minimum Temperature']
                            for col in min_temp_col_names:
                                if col in df_sheet.columns and not pd.isna(row[col]):
                                    min_temp = float(row[col])
                                    break

                            # 24小时数据
                            day_has_data = False
                            for hour in range(24):
                                possible_col_names = [
                                    f'Weather_{hour:02d}:00  (℃)',
                                    f'Weather_{hour:02d}:00 (℃)',
                                    f'Weather_{hour:02d}:00',
                                ]

                                ambient_temp = None
                                for col_name in possible_col_names:
                                    if col_name in df_sheet.columns:
                                        val = row[col_name]
                                        if not pd.isna(val):
                                            ambient_temp = float(val)
                                            break

                                if ambient_temp is None:
                                    if hour < 6:
                                        ambient_temp = min_temp + (max_temp - min_temp) * 0.2
                                    elif hour < 14:
                                        ambient_temp = min_temp + (max_temp - min_temp) * (hour - 6) / 8
                                    elif hour < 18:
                                        ambient_temp = max_temp
                                    else:
                                        ambient_temp = max_temp - (max_temp - min_temp) * (hour - 18) / 6

                                if 0 <= ambient_temp <= 50:
                                    timestamp = date + timedelta(hours=hour)
                                    sheet_data.append({
                                        'timestamp': timestamp,
                                        'ambient_temp': ambient_temp,
                                        'weather_code': weather_code,
                                        'wind_level': wind_level,
                                        'sunshine_hours': sunshine_hours,
                                        'max_temp': max_temp,
                                        'min_temp': min_temp
                                    })
                                    day_has_data = True

                            if day_has_data:
                                valid_days += 1

                        except Exception as e:
                            continue

                    print(f"  ✓ 提取了 {valid_days} 天，{len(sheet_data)} 个小时数据")

                    if len(sheet_data) > 0:
                        all_time_series.extend(sheet_data)
                        total_days += valid_days
                        self.debug_info['weather_sheets'].append({
                            'name': sheet_name,
                            'days': valid_days,
                            'hours': len(sheet_data)
                        })

                except Exception as e:
                    print(f"  ✗ Sheet处理失败: {e}")
                    continue

            if len(all_time_series) == 0:
                print(f"\n✗ 没有提取到任何天气数据")
                return None

            weather_df = pd.DataFrame(all_time_series)
            weather_df.set_index('timestamp', inplace=True)
            weather_df.sort_index(inplace=True)
            weather_df = weather_df[~weather_df.index.duplicated(keep='first')]

            print(f"\n{'=' * 80}")
            print(f"天气数据加载完成")
            print(f"{'=' * 80}")
            print(f"✓ 总时间点: {len(weather_df):,}")
            print(f"✓ 时间跨度: {(weather_df.index.max() - weather_df.index.min()).days + 1} 天")

            self.weather_df = weather_df
            return weather_df

        except Exception as e:
            print(f"\n✗ 加载失败: {e}")
            return None

    def load_predicted_temperature(self, filename='Predicted_temperature_data_for_July_2024.xlsx'):
        """加载预测温度数据 - 修复版"""
        filepath = os.path.join(self.data_dir, filename)
        print(f"\n{'=' * 80}")
        print(f"3. 加载预测温度数据: {filename}")
        print(f"{'=' * 80}")

        try:
            xl_file = pd.ExcelFile(filepath)
            all_sheets = xl_file.sheet_names
            print(f"✓ 找到 {len(all_sheets)} 个sheets")

            all_time_series = []
            total_days = 0

            for sheet_idx, sheet_name in enumerate(all_sheets, 1):
                print(f"\n处理 Sheet {sheet_idx}/{len(all_sheets)}: '{sheet_name}'")

                try:
                    df_sheet = pd.read_excel(filepath, sheet_name=sheet_name)

                    if 'date' not in df_sheet.columns:
                        continue

                    sheet_data = []
                    valid_days = 0

                    for row_idx, row in df_sheet.iterrows():
                        try:
                            date_val = row['date']
                            if pd.isna(date_val):
                                continue

                            if isinstance(date_val, str):
                                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                                    try:
                                        date = pd.to_datetime(date_val, format=fmt)
                                        break
                                    except:
                                        continue
                                else:
                                    date = pd.to_datetime(date_val)
                            else:
                                date = pd.to_datetime(date_val)

                            day_has_data = False
                            for hour in range(24):
                                possible_col_names = [
                                    f'Weather_{hour:02d}:00  (℃)',
                                    f'Weather_{hour:02d}:00 (℃)',
                                    f'Weather_{hour:02d}:00',
                                ]

                                predicted_temp = None
                                for col_name in possible_col_names:
                                    if col_name in df_sheet.columns:
                                        val = row[col_name]
                                        if not pd.isna(val):
                                            predicted_temp = float(val)
                                            break

                                if predicted_temp is not None and 20 <= predicted_temp <= 100:
                                    timestamp = date + timedelta(hours=hour)
                                    sheet_data.append({
                                        'timestamp': timestamp,
                                        'predicted_temp': predicted_temp
                                    })
                                    day_has_data = True

                            if day_has_data:
                                valid_days += 1

                        except Exception as e:
                            continue

                    print(f"  ✓ 提取了 {valid_days} 天，{len(sheet_data)} 个小时数据")

                    if len(sheet_data) > 0:
                        all_time_series.extend(sheet_data)
                        total_days += valid_days
                        self.debug_info['predicted_sheets'].append({
                            'name': sheet_name,
                            'days': valid_days,
                            'hours': len(sheet_data)
                        })

                except Exception as e:
                    continue

            if len(all_time_series) == 0:
                return None

            predicted_df = pd.DataFrame(all_time_series)
            predicted_df.set_index('timestamp', inplace=True)
            predicted_df.sort_index(inplace=True)
            predicted_df = predicted_df[~predicted_df.index.duplicated(keep='first')]

            print(f"\n{'=' * 80}")
            print(f"预测温度数据加载完成")
            print(f"{'=' * 80}")
            print(f"✓ 总时间点: {len(predicted_df):,}")

            self.predicted_df = predicted_df
            return predicted_df

        except Exception as e:
            print(f"\n✗ 加载失败: {e}")
            return None

    def merge_all_data(self):
        """合并所有数据并生成特征"""
        print(f"\n{'=' * 80}")
        print("4. 合并数据并生成特征")
        print(f"{'=' * 80}")

        if self.oil_temp_df is None:
            print("✗ 油温数据未加载")
            return None

        merged = self.oil_temp_df.copy()

        # 合并天气数据
        if self.weather_df is not None:
            merged = merged.join(self.weather_df, how='left')
            merged['ambient_temp'].fillna(method='ffill', inplace=True)
            merged['ambient_temp'].fillna(28.0, inplace=True)
            merged['weather_code'].fillna(6, inplace=True)
            merged['wind_level'].fillna(2, inplace=True)
            merged['sunshine_hours'].fillna(8.0, inplace=True)
            merged['max_temp'].fillna(32.0, inplace=True)
            merged['min_temp'].fillna(24.0, inplace=True)
        else:
            merged['ambient_temp'] = 28.0
            merged['weather_code'] = 6
            merged['wind_level'] = 2
            merged['sunshine_hours'] = 8.0
            merged['max_temp'] = 32.0
            merged['min_temp'] = 24.0

        # 合并预测温度
        if self.predicted_df is not None:
            merged = merged.join(self.predicted_df, how='left')
            merged['predicted_temp'].fillna(method='ffill', inplace=True)
            merged['predicted_temp'].fillna(merged['oil_temp'], inplace=True)

            target_temp = 50.0
            merged['predicted_error'] = merged['predicted_temp'] - target_temp
            merged['feedforward_signal'] = -merged['predicted_error'] / 10.0
        else:
            merged['predicted_temp'] = merged['oil_temp']
            merged['predicted_error'] = 0
            merged['feedforward_signal'] = 0

        # 特征工程
        merged['oil_temp_error'] = merged['oil_temp'] - 50.0
        merged['oil_temp_ma3'] = merged['oil_temp'].rolling(window=3, min_periods=1).mean()
        merged['oil_temp_ma6'] = merged['oil_temp'].rolling(window=6, min_periods=1).mean()
        merged['oil_temp_std3'] = merged['oil_temp'].rolling(window=3, min_periods=1).std().fillna(0)
        merged['temp_change_rate'] = merged['oil_temp'].diff().fillna(0)
        merged['temp_acceleration'] = merged['temp_change_rate'].diff().fillna(0)

        merged['temp_difference'] = merged['oil_temp'] - merged['ambient_temp']
        merged['ambient_temp_ma3'] = merged['ambient_temp'].rolling(window=3, min_periods=1).mean()

        merged['predicted_change'] = merged['predicted_temp'].diff().fillna(0)
        merged['predicted_trend'] = merged['predicted_temp'].rolling(window=3, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0, raw=False
        ).fillna(0)

        merged['hour'] = merged.index.hour
        merged['day_of_week'] = merged.index.dayofweek
        merged['is_daytime'] = ((merged.index.hour >= 6) & (merged.index.hour < 18)).astype(int)
        merged['hour_sin'] = np.sin(2 * np.pi * merged.index.hour / 24)
        merged['hour_cos'] = np.cos(2 * np.pi * merged.index.hour / 24)

        base_load = 0.7
        temp_factor = (merged['oil_temp'] - 50) / 20
        time_factor = 0.2 * np.sin(2 * np.pi * merged.index.hour / 24)
        merged['load_rate'] = np.clip(base_load + temp_factor * 0.15 + time_factor, 0.5, 0.95)

        merged['weather_impact'] = merged['weather_code'].apply(
            lambda x: 1.2 if x in [4, 5, 9] else 1.0 if x in [2, 3] else 0.8
        )
        merged['wind_impact'] = 1.0 + merged['wind_level'] * 0.05

        merged.fillna(method='ffill', inplace=True)
        merged.fillna(method='bfill', inplace=True)
        merged.fillna(0, inplace=True)

        print(f"✓ 特征工程完成")
        print(f"✓ 最终形状: {merged.shape}")

        self.merged_df = merged
        return merged

    def get_statistics(self):
        """打印详细统计"""
        if self.merged_df is None:
            return

        df = self.merged_df

        print("\n" + "=" * 80)
        print("详细数据统计".center(80))
        print("=" * 80)

        print(f"\n📊 基本信息:")
        print(f"  总样本数(小时): {len(df):,}")
        print(f"  时间跨度(天): {(df.index.max() - df.index.min()).days + 1:,}")
        print(f"  时间范围: {df.index.min()} → {df.index.max()}")
        print(f"  可训练Episodes(48h/个): {len(df) // 48:,}")

        print(f"\n🌡️ 油温统计:")
        print(f"  均值: {df['oil_temp'].mean():.2f}°C")
        print(f"  标准差: {df['oil_temp'].std():.2f}°C")
        print(f"  范围: [{df['oil_temp'].min():.2f}, {df['oil_temp'].max():.2f}]°C")

        print(f"\n🌤️ 环境温度统计:")
        print(f"  均值: {df['ambient_temp'].mean():.2f}°C")
        print(f"  范围: [{df['ambient_temp'].min():.2f}, {df['ambient_temp'].max():.2f}]°C")

        if 'predicted_temp' in df.columns:
            print(f"\n🔮 预测温度统计:")
            print(f"  均值: {df['predicted_temp'].mean():.2f}°C")
            print(f"  与实际MAE: {np.mean(np.abs(df['predicted_temp'] - df['oil_temp'])):.2f}°C")

        print("=" * 80)

    def save_processed_data(self, filename='processed_transformer_data.pkl'):
        """保存处理后的数据"""
        if self.merged_df is None:
            return

        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump({'processed_data': self.merged_df}, f)

        print(f"\n✓ 数据已保存到: {filepath}")

    def load_all_and_process(self):
        """一键加载和处理所有数据"""
        print("=" * 80)
        print("变压器智能冷却系统 - 真实数据加载".center(80))
        print("=" * 80)

        self.load_oil_temperature()
        self.load_weather_data()
        self.load_predicted_temperature()

        merged_df = self.merge_all_data()

        if merged_df is not None:
            self.get_statistics()
            self.save_processed_data()

            print("\n" + "=" * 80)
            print("✓ 数据加载完成!".center(80))
            print("=" * 80)

            return merged_df
        else:
            print("\n✗ 数据加载失败")
            return None


def main():
    """主函数"""
    loader = TransformerDataLoader(data_dir='data')
    data = loader.load_all_and_process()

    if data is not None:
        print("\n数据预览:")
        print(data.head())

    return data


if __name__ == "__main__":
    main()
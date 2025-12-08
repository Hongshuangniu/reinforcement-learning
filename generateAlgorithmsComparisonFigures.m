function generateAlgorithmsComparisonFigures(resultsPath, outputPath)
% 生成多算法对比图表（完全基于降温能力评价体系）
%
% 🔥 修复内容：
% 1. ✅ 添加数据有效性检查
% 2. ✅ 修复时序图数据缺失问题
% 3. ✅ 改善错误处理和提示
% 4. ✅ 确保所有图表都能正常显示
%
% 输入:
%   resultsPath - Python导出的matlab_data路径
%   outputPath  - 输出图表路径

if nargin < 1
    resultsPath = 'matlab_data';
end
if nargin < 2
    outputPath = 'results/figures/Comparison';
end

% 创建输出目录
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
if ~exist([outputPath '/Chinese'], 'dir')
    mkdir([outputPath '/Chinese']);
end
if ~exist([outputPath '/English'], 'dir')
    mkdir([outputPath '/English']);
end

fprintf('\n========== 生成算法对比图表（基于降温能力评价）==========\n');

% 加载数据
try
    data = loadPythonComparisonData(resultsPath);
    fprintf('✓ Python数据加载成功\n');
catch ME
    error(['数据加载失败: ' ME.message]);
end

% 生成各类图表
try
    fprintf('\n生成图表序列...\n');
    
    % 1. 训练过程对比
    fprintf('  1/10 训练过程对比...\n');
    generateTrainingComparison(data, outputPath);
    
    % 2. 降温能力指标对比（核心）
    fprintf('  2/10 降温能力指标对比...\n');
    generateCoolingMetricsComparison(data, outputPath);
    
    % 3. 工业控制指标对比
    fprintf('  3/10 工业控制指标对比...\n');
    generateIndustrialControlMetrics(data, outputPath);
    
    % 4. 动态性能指标对比
    fprintf('  4/10 动态性能指标对比...\n');
    generateDynamicPerformanceMetrics(data, outputPath);
    
    % 5. 控制精度指标对比
    fprintf('  5/10 控制精度指标对比...\n');
    generateControlPrecisionMetrics(data, outputPath);
    
    % 6. 能效指标对比
    fprintf('  6/10 能效指标对比...\n');
    generateEnergyEfficiencyMetrics(data, outputPath);
    
    % 7. 综合性能评分对比
    fprintf('  7/10 综合性能评分对比...\n');
    generatePerformanceIndexComparison(data, outputPath);
    
    % 8. 温度控制效果对比
    fprintf('  8/10 温度控制效果对比...\n');
    generateTemperatureControl(data, outputPath);
    
    % 9. 综合性能雷达图
    fprintf('  9/10 综合性能雷达图...\n');
    generateRadarChart(data, outputPath);
    
    % 10. 降温效果时序图
    fprintf('  10/10 降温效果时序图...\n');
    generateCoolingTimeSeriesComparison(data, outputPath);
    
    fprintf('\n✓ 算法对比图表生成完成！\n');
    fprintf('  输出路径: %s\n', outputPath);
catch ME
    warning(['图表生成出错: ' ME.message]);
    if ~isempty(ME.stack)
        fprintf('  错误位置: %s (第 %d 行)\n', ME.stack(1).name, ME.stack(1).line);
    end
end
end

%% ========== 数据加载函数 ==========
function data = loadPythonComparisonData(resultsPath)
    data = struct();
    
    pythonAlgoNames = {'improved_sac', 'sac', 'ppo', 'ddpg', 'td3'};
    displayNames = {'Improved SAC', 'Traditional SAC', 'PPO', 'DDPG', 'TD3'};
    
    data.algorithms = pythonAlgoNames;
    data.algorithmNames = displayNames;
    
    % 1. 加载训练数据
    fprintf('  加载训练数据...\n');
    data.training = struct();
    for i = 1:length(pythonAlgoNames)
        algo = pythonAlgoNames{i};
        trainFile = fullfile(resultsPath, ['training_' algo '.mat']);
        
        if exist(trainFile, 'file')
            trainData = load(trainFile);
            
            if isfield(trainData, 'episode_rewards')
                data.training.(algo).trainInfo.episodeReward = ...
                    double(trainData.episode_rewards(:)');
                data.training.(algo).trainInfo.averageReward = ...
                    movmean(data.training.(algo).trainInfo.episodeReward, 10);
                data.training.(algo).bestReward = ...
                    max(data.training.(algo).trainInfo.episodeReward);
            end
            
            if isfield(trainData, 'actor_losses') && ~isempty(trainData.actor_losses)
                data.training.(algo).actorLosses = double(trainData.actor_losses(:)');
            end
            if isfield(trainData, 'critic_losses') && ~isempty(trainData.critic_losses)
                data.training.(algo).criticLosses = double(trainData.critic_losses(:)');
            end
            
            fprintf('    ✓ %s\n', algo);
        else
            fprintf('    ⚠ 未找到文件: %s\n', trainFile);
        end
    end
    
    % 2. 加载评估数据
    fprintf('  加载评估数据...\n');
    data.evaluation = struct();
    for i = 1:length(pythonAlgoNames)
        algo = pythonAlgoNames{i};
        evalFile = fullfile(resultsPath, ['evaluation_' algo '.mat']);

        if exist(evalFile, 'file')
            evalData = load(evalFile);

            % 初始化该算法的评估数据
            data.evaluation.(algo) = struct();
            
            % ===== 降温能力指标 =====
            data.evaluation.(algo).mae = getFieldOrDefault(evalData, 'cooling_mae', 'MAE', 0);
            data.evaluation.(algo).rmse = getFieldOrDefault(evalData, 'cooling_rmse', 'RMSE', 0);
            data.evaluation.(algo).maxError = getFieldOrDefault(evalData, 'cooling_max_error', 'MaxAE', 0);
            
            % ===== 工业控制指标 =====
            data.evaluation.(algo).ise = getFieldOrDefault(evalData, 'ISE', '', 0);
            data.evaluation.(algo).iae = getFieldOrDefault(evalData, 'IAE', '', 0);
            data.evaluation.(algo).itae = getFieldOrDefault(evalData, 'ITAE', '', 0);
            
            % ===== 动态性能指标 =====
            data.evaluation.(algo).settling_time = getFieldOrDefault(evalData, 'settling_time', '', 0);
            data.evaluation.(algo).peak_overshoot = getFieldOrDefault(evalData, 'peak_overshoot', '', 0);
            data.evaluation.(algo).steady_state_error = getFieldOrDefault(evalData, 'steady_state_error', '', 0);
            
            % ===== 控制精度指标 =====
            data.evaluation.(algo).precision_2c = getFieldOrDefault(evalData, 'control_precision_2C', '', 0);
            data.evaluation.(algo).precision_1c = getFieldOrDefault(evalData, 'control_precision_1C', '', 0);
            data.evaluation.(algo).temp_stability = getFieldOrDefault(evalData, 'temperature_stability', '', 0);
            
            % ===== 能效指标 =====
            data.evaluation.(algo).total_energy = getFieldOrDefault(evalData, 'total_energy', '', 0);
            data.evaluation.(algo).energy_efficiency = getFieldOrDefault(evalData, 'energy_efficiency_ratio', '', 0);
            
            % ===== 综合性能指标 =====
            data.evaluation.(algo).performance_index = getFieldOrDefault(evalData, 'total_performance_index', '', 0);
            data.evaluation.(algo).precision_score = getFieldOrDefault(evalData, 'precision_score', '', 0);
            data.evaluation.(algo).efficiency_score = getFieldOrDefault(evalData, 'efficiency_score', '', 0);
            data.evaluation.(algo).stability_score = getFieldOrDefault(evalData, 'stability_score', '', 0);
            data.evaluation.(algo).speed_score = getFieldOrDefault(evalData, 'speed_score', '', 0);
            
            % ===== RL指标 =====
            data.evaluation.(algo).avg_reward = getFieldOrDefault(evalData, 'avg_reward', '', 0);
            
            % 🔥 修复：加载温度和降温时序数据（用于图8和图10）
            if isfield(evalData, 'episode1_true_temps')
                temps = double(evalData.episode1_true_temps(:));
                data.evaluation.(algo).temperatures = temps;
                data.evaluation.(algo).nSteps = length(temps);
            end
            
            if isfield(evalData, 'episode1_actual_coolings')
                data.evaluation.(algo).actual_coolings = double(evalData.episode1_actual_coolings(:));
            end
            
            if isfield(evalData, 'episode1_target_coolings')
                data.evaluation.(algo).target_coolings = double(evalData.episode1_target_coolings(:));
            end
            
            fprintf('    ✓ %s\n', algo);
        else
            fprintf('    ⚠ 未找到文件: %s\n', evalFile);
        end
    end
end

function value = getFieldOrDefault(s, field1, field2, defaultValue)
    % 辅助函数：尝试从两个可能的字段名获取值，否则返回默认值
    if ~isempty(field1) && isfield(s, field1)
        value = double(s.(field1));
    elseif ~isempty(field2) && isfield(s, field2)
        value = double(s.(field2));
    else
        value = defaultValue;
    end
end

%% ========== 图表生成函数 ==========

function generateTrainingComparison(data, outputPath)
    % 训练过程对比
    try
        fig = figure('Position', [100, 100, 1400, 600], 'Visible', 'off');
        
        colors = lines(length(data.algorithms));
        hold on;
        
        hasData = false;
        for i = 1:length(data.algorithms)
            algo = data.algorithms{i};
            if isfield(data.training, algo) && ...
               isfield(data.training.(algo).trainInfo, 'episodeReward')
                episodeReward = data.training.(algo).trainInfo.episodeReward;
                movingAvg = movmean(episodeReward, 10);
                episodes = 1:length(movingAvg);
                plot(episodes, movingAvg, 'LineWidth', 2.5, 'Color', colors(i,:), ...
                    'DisplayName', data.algorithmNames{i});
                hasData = true;
            end
        end
        
        if hasData
            xlabel('训练回合', 'FontSize', 14);
            ylabel('平均累计奖励', 'FontSize', 14);
            title('训练过程对比（10回合移动平均）', 'FontSize', 16, 'FontWeight', 'bold');
            legend('Location', 'best', 'FontSize', 12);
            grid on;
            
            saveas(fig, fullfile(outputPath, 'Chinese', '01_训练过程对比.png'));
            savefig(fig, fullfile(outputPath, 'Chinese', '01_训练过程对比.fig'));
            
            % 英文版
            xlabel('Episodes', 'FontSize', 14);
            ylabel('Average Cumulative Reward', 'FontSize', 14);
            title('Training Comparison (10-Episode Moving Average)', 'FontSize', 16, 'FontWeight', 'bold');
            
            saveas(fig, fullfile(outputPath, 'English', '01_training_comparison.png'));
            savefig(fig, fullfile(outputPath, 'English', '01_training_comparison.fig'));
        end
        
        close(fig);
    catch ME
        warning(['训练过程对比图生成失败: ' ME.message]);
    end
end

function generateCoolingMetricsComparison(data, outputPath)
    % 降温能力指标对比（核心）
    try
        fig = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
        
        nAlgos = length(data.algorithms);
        mae = zeros(1, nAlgos);
        rmse = zeros(1, nAlgos);
        maxError = zeros(1, nAlgos);
        
        for i = 1:nAlgos
            algo = data.algorithms{i};
            if isfield(data.evaluation, algo)
                mae(i) = data.evaluation.(algo).mae;
                rmse(i) = data.evaluation.(algo).rmse;
                maxError(i) = data.evaluation.(algo).maxError;
            end
        end
        
        % 子图1: MAE
        subplot(1, 3, 1);
        bar(mae, 'FaceColor', [0.25, 0.55, 0.85]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('降温MAE (°C)', 'FontSize', 12);
        title('降温平均绝对误差', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 子图2: RMSE
        subplot(1, 3, 2);
        bar(rmse, 'FaceColor', [0.85, 0.45, 0.25]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('降温RMSE (°C)', 'FontSize', 12);
        title('降温均方根误差', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 子图3: 最大误差
        subplot(1, 3, 3);
        bar(maxError, 'FaceColor', [0.45, 0.75, 0.35]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('最大降温误差 (°C)', 'FontSize', 12);
        title('最大降温误差', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        sgtitle('降温能力指标对比（核心评价）', 'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'Chinese', '02_降温能力指标对比.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '02_降温能力指标对比.fig'));
        
        % 英文版
        subplot(1, 3, 1);
        ylabel('Cooling MAE (°C)', 'FontSize', 12);
        title('Mean Absolute Error', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(1, 3, 2);
        ylabel('Cooling RMSE (°C)', 'FontSize', 12);
        title('Root Mean Square Error', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(1, 3, 3);
        ylabel('Max Cooling Error (°C)', 'FontSize', 12);
        title('Maximum Cooling Error', 'FontSize', 14, 'FontWeight', 'bold');
        
        sgtitle('Cooling Performance Metrics Comparison', 'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'English', '02_cooling_metrics_comparison.png'));
        savefig(fig, fullfile(outputPath, 'English', '02_cooling_metrics_comparison.fig'));
        close(fig);
    catch ME
        warning(['降温能力指标对比图生成失败: ' ME.message]);
    end
end

function generateIndustrialControlMetrics(data, outputPath)
    % 工业控制指标对比
    try
        fig = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
        
        nAlgos = length(data.algorithms);
        ise = zeros(1, nAlgos);
        iae = zeros(1, nAlgos);
        itae = zeros(1, nAlgos);
        
        for i = 1:nAlgos
            algo = data.algorithms{i};
            if isfield(data.evaluation, algo)
                ise(i) = data.evaluation.(algo).ise;
                iae(i) = data.evaluation.(algo).iae;
                itae(i) = data.evaluation.(algo).itae;
            end
        end
        
        % 子图1: ISE
        subplot(1, 3, 1);
        bar(ise, 'FaceColor', [0.85, 0.35, 0.45]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('ISE', 'FontSize', 12);
        title('积分平方误差 (ISE)', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 子图2: IAE
        subplot(1, 3, 2);
        bar(iae, 'FaceColor', [0.45, 0.65, 0.85]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('IAE', 'FontSize', 12);
        title('积分绝对误差 (IAE)', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 子图3: ITAE
        subplot(1, 3, 3);
        bar(itae, 'FaceColor', [0.75, 0.55, 0.25]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('ITAE', 'FontSize', 12);
        title('时间加权积分绝对误差 (ITAE)', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        sgtitle('工业控制指标对比', 'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'Chinese', '03_工业控制指标对比.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '03_工业控制指标对比.fig'));
        
        % 英文版
        subplot(1, 3, 1);
        ylabel('ISE', 'FontSize', 12);
        title('Integral Square Error', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(1, 3, 2);
        ylabel('IAE', 'FontSize', 12);
        title('Integral Absolute Error', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(1, 3, 3);
        ylabel('ITAE', 'FontSize', 12);
        title('Integral Time Absolute Error', 'FontSize', 14, 'FontWeight', 'bold');
        
        sgtitle('Industrial Control Metrics Comparison', 'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'English', '03_industrial_control_metrics.png'));
        savefig(fig, fullfile(outputPath, 'English', '03_industrial_control_metrics.fig'));
        close(fig);
    catch ME
        warning(['工业控制指标对比图生成失败: ' ME.message]);
    end
end

function generateDynamicPerformanceMetrics(data, outputPath)
    % 动态性能指标对比
    try
        fig = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
        
        nAlgos = length(data.algorithms);
        settling_time = zeros(1, nAlgos);
        overshoot = zeros(1, nAlgos);
        ss_error = zeros(1, nAlgos);
        
        for i = 1:nAlgos
            algo = data.algorithms{i};
            if isfield(data.evaluation, algo)
                settling_time(i) = data.evaluation.(algo).settling_time;
                overshoot(i) = data.evaluation.(algo).peak_overshoot;
                ss_error(i) = data.evaluation.(algo).steady_state_error;
            end
        end
        
        % 子图1: 调节时间
        subplot(1, 3, 1);
        bar(settling_time, 'FaceColor', [0.55, 0.35, 0.75]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('调节时间 (步)', 'FontSize', 12);
        title('调节时间', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 子图2: 超调量
        subplot(1, 3, 2);
        bar(overshoot, 'FaceColor', [0.95, 0.55, 0.35]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('超调量 (%)', 'FontSize', 12);
        title('超调量', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 子图3: 稳态误差
        subplot(1, 3, 3);
        bar(ss_error, 'FaceColor', [0.35, 0.75, 0.65]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('稳态误差 (°C)', 'FontSize', 12);
        title('稳态误差', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        sgtitle('动态性能指标对比', 'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'Chinese', '04_动态性能指标对比.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '04_动态性能指标对比.fig'));
        
        % 英文版
        subplot(1, 3, 1);
        ylabel('Settling Time (steps)', 'FontSize', 12);
        title('Settling Time', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(1, 3, 2);
        ylabel('Overshoot (%)', 'FontSize', 12);
        title('Peak Overshoot', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(1, 3, 3);
        ylabel('SS Error (°C)', 'FontSize', 12);
        title('Steady State Error', 'FontSize', 14, 'FontWeight', 'bold');
        
        sgtitle('Dynamic Performance Metrics Comparison', 'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'English', '04_dynamic_performance_metrics.png'));
        savefig(fig, fullfile(outputPath, 'English', '04_dynamic_performance_metrics.fig'));
        close(fig);
    catch ME
        warning(['动态性能指标对比图生成失败: ' ME.message]);
    end
end

function generateControlPrecisionMetrics(data, outputPath)
    % 控制精度指标对比
    try
        fig = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
        
        nAlgos = length(data.algorithms);
        precision_2c = zeros(1, nAlgos);
        precision_1c = zeros(1, nAlgos);
        stability = zeros(1, nAlgos);
        
        for i = 1:nAlgos
            algo = data.algorithms{i};
            if isfield(data.evaluation, algo)
                precision_2c(i) = data.evaluation.(algo).precision_2c;
                precision_1c(i) = data.evaluation.(algo).precision_1c;
                stability(i) = data.evaluation.(algo).temp_stability;
            end
        end
        
        % 子图1: ±2°C精度
        subplot(1, 3, 1);
        bar(precision_2c, 'FaceColor', [0.25, 0.75, 0.55]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('±2°C精度 (%)', 'FontSize', 12);
        title('±2°C控制精度', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        ylim([0 100]);
        
        % 子图2: ±1°C精度
        subplot(1, 3, 2);
        bar(precision_1c, 'FaceColor', [0.65, 0.35, 0.85]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('±1°C精度 (%)', 'FontSize', 12);
        title('±1°C控制精度', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        ylim([0 100]);
        
        % 子图3: 温度稳定性
        subplot(1, 3, 3);
        bar(stability, 'FaceColor', [0.85, 0.65, 0.25]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('稳定性指标', 'FontSize', 12);
        title('温度稳定性', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        sgtitle('控制精度指标对比', 'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'Chinese', '05_控制精度指标对比.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '05_控制精度指标对比.fig'));
        
        % 英文版
        subplot(1, 3, 1);
        ylabel('±2°C Precision (%)', 'FontSize', 12);
        title('±2°C Control Precision', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(1, 3, 2);
        ylabel('±1°C Precision (%)', 'FontSize', 12);
        title('±1°C Control Precision', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(1, 3, 3);
        ylabel('Stability Index', 'FontSize', 12);
        title('Temperature Stability', 'FontSize', 14, 'FontWeight', 'bold');
        
        sgtitle('Control Precision Metrics Comparison', 'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'English', '05_control_precision_metrics.png'));
        savefig(fig, fullfile(outputPath, 'English', '05_control_precision_metrics.fig'));
        close(fig);
    catch ME
        warning(['控制精度指标对比图生成失败: ' ME.message]);
    end
end

function generateEnergyEfficiencyMetrics(data, outputPath)
    % 🔥 能效指标对比（修复版 - 处理数据缺失）
    try
        fig = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
        
        nAlgos = length(data.algorithms);
        total_energy = zeros(1, nAlgos);
        efficiency = zeros(1, nAlgos);
        has_energy_data = false(1, nAlgos);
        
        for i = 1:nAlgos
            algo = data.algorithms{i};
            if isfield(data.evaluation, algo)
                % 总能耗
                energy_val = data.evaluation.(algo).total_energy;
                if energy_val > 0
                    total_energy(i) = energy_val;
                    has_energy_data(i) = true;
                else
                    total_energy(i) = NaN;
                end
                
                % 能效比
                eff_val = data.evaluation.(algo).energy_efficiency;
                if eff_val > 0
                    efficiency(i) = eff_val;
                else
                    efficiency(i) = NaN;
                end
            else
                total_energy(i) = NaN;
                efficiency(i) = NaN;
            end
        end
        
        % 子图1: 总能耗
        subplot(1, 2, 1);
        h1 = bar(total_energy, 'FaceColor', [0.95, 0.45, 0.35]);
        
        % 将NaN值的柱子设为灰色
        if any(isnan(total_energy))
            h1.FaceColor = 'flat';
            for i = 1:length(total_energy)
                if isnan(total_energy(i))
                    h1.CData(i,:) = [0.7, 0.7, 0.7];
                end
            end
        end
        
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('总能耗', 'FontSize', 12);
        title('总能耗对比', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 标注数值或N/A
        for i = 1:length(total_energy)
            if ~isnan(total_energy(i)) && total_energy(i) > 0
                text(i, total_energy(i), sprintf('%.1f', total_energy(i)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                    'FontSize', 9);
            else
                text(i, 0, 'N/A', ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                    'FontSize', 9, 'Color', [0.5, 0.5, 0.5]);
            end
        end
        
        % 子图2: 能效比
        subplot(1, 2, 2);
        h2 = bar(efficiency, 'FaceColor', [0.35, 0.75, 0.45]);
        
        % 将NaN值的柱子设为灰色
        if any(isnan(efficiency))
            h2.FaceColor = 'flat';
            for i = 1:length(efficiency)
                if isnan(efficiency(i))
                    h2.CData(i,:) = [0.7, 0.7, 0.7];
                end
            end
        end
        
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('能效比', 'FontSize', 12);
        title('能效比对比', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 标注数值或N/A
        for i = 1:length(efficiency)
            if ~isnan(efficiency(i)) && efficiency(i) > 0
                text(i, efficiency(i), sprintf('%.4f', efficiency(i)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                    'FontSize', 9);
            else
                text(i, 0, 'N/A', ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                    'FontSize', 9, 'Color', [0.5, 0.5, 0.5]);
            end
        end
        
        sgtitle('能效指标对比', 'FontSize', 16, 'FontWeight', 'bold');
        
        % 添加说明
        if any(~has_energy_data)
            annotation('textbox', [0.15, 0.02, 0.7, 0.03], ...
                'String', '注: 灰色柱表示该算法暂无能耗数据', ...
                'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
                'FontSize', 10, 'Color', [0.5, 0.5, 0.5]);
        end
        
        saveas(fig, fullfile(outputPath, 'Chinese', '06_能效指标对比.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '06_能效指标对比.fig'));
        
        % 英文版
        subplot(1, 2, 1);
        ylabel('Total Energy', 'FontSize', 12);
        title('Total Energy Comparison', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(1, 2, 2);
        ylabel('Energy Efficiency', 'FontSize', 12);
        title('Energy Efficiency Comparison', 'FontSize', 14, 'FontWeight', 'bold');
        
        sgtitle('Energy Efficiency Metrics Comparison', 'FontSize', 16, 'FontWeight', 'bold');
        
        if any(~has_energy_data)
            annotation('textbox', [0.15, 0.02, 0.7, 0.03], ...
                'String', 'Note: Gray bars indicate no energy data available', ...
                'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
                'FontSize', 10, 'Color', [0.5, 0.5, 0.5]);
        end
        
        saveas(fig, fullfile(outputPath, 'English', '06_energy_efficiency_metrics.png'));
        savefig(fig, fullfile(outputPath, 'English', '06_energy_efficiency_metrics.fig'));
        close(fig);
    catch ME
        warning(['能效指标对比图生成失败: ' ME.message]);
    end
end

function generatePerformanceIndexComparison(data, outputPath)
    % 综合性能评分对比
    try
        fig = figure('Position', [100, 100, 1400, 800], 'Visible', 'off');
        
        nAlgos = length(data.algorithms);
        total_index = zeros(1, nAlgos);
        precision_scores = zeros(1, nAlgos);
        efficiency_scores = zeros(1, nAlgos);
        stability_scores = zeros(1, nAlgos);
        speed_scores = zeros(1, nAlgos);
        
        for i = 1:nAlgos
            algo = data.algorithms{i};
            if isfield(data.evaluation, algo)
                total_index(i) = data.evaluation.(algo).performance_index;
                precision_scores(i) = data.evaluation.(algo).precision_score;
                efficiency_scores(i) = data.evaluation.(algo).efficiency_score;
                stability_scores(i) = data.evaluation.(algo).stability_score;
                speed_scores(i) = data.evaluation.(algo).speed_score;
            end
        end
        
        % 子图1: 综合性能指标
        subplot(2, 1, 1);
        bar(total_index, 'FaceColor', [0.25, 0.55, 0.85]);
        set(gca, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('综合评分', 'FontSize', 12);
        title('综合性能指标 (0-100)', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        ylim([0 100]);
        
        % 子图2: 各项分数
        subplot(2, 1, 2);
        x = 1:nAlgos;
        width = 0.2;
        b1 = bar(x - 1.5*width, precision_scores, width, 'FaceColor', [0.85, 0.35, 0.45]);
        hold on;
        b2 = bar(x - 0.5*width, efficiency_scores, width, 'FaceColor', [0.45, 0.75, 0.35]);
        b3 = bar(x + 0.5*width, stability_scores, width, 'FaceColor', [0.75, 0.55, 0.25]);
        b4 = bar(x + 1.5*width, speed_scores, width, 'FaceColor', [0.55, 0.35, 0.75]);
        
        set(gca, 'XTick', x, 'XTickLabel', data.algorithmNames, 'XTickLabelRotation', 25);
        ylabel('分项评分', 'FontSize', 12);
        title('性能分项评分', 'FontSize', 14, 'FontWeight', 'bold');
        legend({'控制精度', '能效', '稳定性', '快速性'}, 'Location', 'best');
        grid on;
        ylim([0 100]);
        
        saveas(fig, fullfile(outputPath, 'Chinese', '07_综合性能评分.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '07_综合性能评分.fig'));
        
        % 英文版
        subplot(2, 1, 1);
        ylabel('Total Score', 'FontSize', 12);
        title('Total Performance Index (0-100)', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(2, 1, 2);
        ylabel('Sub-scores', 'FontSize', 12);
        title('Performance Sub-scores', 'FontSize', 14, 'FontWeight', 'bold');
        legend({'Precision', 'Efficiency', 'Stability', 'Speed'}, 'Location', 'best');
        
        saveas(fig, fullfile(outputPath, 'English', '07_performance_index_comparison.png'));
        savefig(fig, fullfile(outputPath, 'English', '07_performance_index_comparison.fig'));
        close(fig);
    catch ME
        warning(['综合性能评分图生成失败: ' ME.message]);
    end
end

function generateTemperatureControl(data, outputPath)
    % 🔥 修复：温度控制效果对比（添加数据有效性检查）
    try
        fig = figure('Position', [100, 100, 1400, 600], 'Visible', 'off');
        
        colors = lines(length(data.algorithms));
        hold on;
        
        hasData = false;
        for i = 1:length(data.algorithms)
            algo = data.algorithms{i};
            if isfield(data.evaluation, algo) && isfield(data.evaluation.(algo), 'temperatures')
                temps = data.evaluation.(algo).temperatures;
                nSteps = length(temps);
                time = (0:nSteps-1) * 0.5;  % 每步0.5小时
                plot(time, temps, 'LineWidth', 1.5, 'Color', colors(i,:), ...
                    'DisplayName', data.algorithmNames{i});
                hasData = true;
            end
        end
        
        if hasData
            xlabel('时间 (小时)', 'FontSize', 14);
            ylabel('温度 (°C)', 'FontSize', 14);
            title('温度控制质量对比', 'FontSize', 16, 'FontWeight', 'bold');
            legend('Location', 'best', 'FontSize', 10);
            grid on;
            
            saveas(fig, fullfile(outputPath, 'Chinese', '08_控制质量对比.png'));
            savefig(fig, fullfile(outputPath, 'Chinese', '08_控制质量对比.fig'));
            
            % 英文版
            xlabel('Time (hours)', 'FontSize', 14);
            ylabel('Temperature (°C)', 'FontSize', 14);
            title('Control Quality Comparison', 'FontSize', 16, 'FontWeight', 'bold');
            
            saveas(fig, fullfile(outputPath, 'English', '08_control_quality.png'));
            savefig(fig, fullfile(outputPath, 'English', '08_control_quality.fig'));
        else
            warning('  ⚠ 温度控制图: 没有可用的温度数据');
        end
        
        close(fig);
    catch ME
        warning(['控制质量对比图生成失败: ' ME.message]);
    end
end

function generateRadarChart(data, outputPath)
    % 综合性能雷达图
    try
        fig = figure('Position', [100, 100, 800, 800], 'Visible', 'off');
        
        nAlgos = length(data.algorithms);
        nMetrics = 6;
        radarData = zeros(nAlgos, nMetrics);

        for i = 1:nAlgos
            algo = data.algorithms{i};
            if isfield(data.evaluation, algo)
                % 1. 控制精度 (基于MAE，值越小越好，转换为0-1分数)
                mae = data.evaluation.(algo).mae;
                radarData(i, 1) = max(0, min(1, 1 - mae / 5));
                
                % 2. 稳定性 (基于RMSE)
                rmse = data.evaluation.(algo).rmse;
                radarData(i, 2) = max(0, min(1, 1 - rmse / 5));
                
                % 3. 快速性 (基于调节时间)
                settling_time = data.evaluation.(algo).settling_time;
                radarData(i, 3) = max(0, min(1, 1 - settling_time / 30));
                
                % 4. 能效 (归一化能效比)
                eer = data.evaluation.(algo).energy_efficiency;
                radarData(i, 4) = min(1, max(0, eer * 100));
                
                % 5. ±2°C精度
                precision = data.evaluation.(algo).precision_2c;
                radarData(i, 5) = precision / 100;
                
                % 6. 综合性能指标
                pi = data.evaluation.(algo).performance_index;
                radarData(i, 6) = pi / 100;
            end
        end

        metricNamesCN = {'控制精度', '稳定性', '快速性', '能效', '±2°C精度', '综合性能'};
        plotRadarWithPolarAxes(fig, radarData, data.algorithmNames, metricNamesCN, '综合性能雷达图');
        
        saveas(fig, fullfile(outputPath, 'Chinese', '09_综合性能雷达图.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '09_综合性能雷达图.fig'));
        
        clf(fig);
        metricNamesEN = {'Precision', 'Stability', 'Speed', 'Efficiency', '±2°C Precision', 'Performance'};
        plotRadarWithPolarAxes(fig, radarData, data.algorithmNames, metricNamesEN, 'Comprehensive Performance Radar');
        
        saveas(fig, fullfile(outputPath, 'English', '09_performance_radar.png'));
        savefig(fig, fullfile(outputPath, 'English', '09_performance_radar.fig'));
        
        close(fig);
    catch ME
        warning(['雷达图生成失败: ' ME.message]);
    end
end

function plotRadarWithPolarAxes(fig, data, labels, categories, chartTitle)
    nAlgos = size(data, 1);
    nMetrics = size(data, 2);
    
    pax = polaraxes('Parent', fig);
    hold(pax, 'on');
    
    angles = linspace(0, 2*pi, nMetrics+1);
    colors = lines(nAlgos);
    
    for i = 1:nAlgos
        values = [data(i, :), data(i, 1)];
        polarplot(pax, angles, values, 'LineWidth', 2, ...
            'Color', colors(i,:), 'DisplayName', labels{i});
    end
    
    pax.ThetaTick = rad2deg(angles(1:end-1));
    pax.ThetaTickLabel = categories;
    pax.RLim = [0 1];
    pax.RTick = [0.2 0.4 0.6 0.8 1.0];
    pax.FontSize = 12;
    title(pax, chartTitle, 'FontSize', 16, 'FontWeight', 'bold');
    legend(pax, 'Location', 'bestoutside');
    hold(pax, 'off');
end

function generateCoolingTimeSeriesComparison(data, outputPath)
    % 🔥 修复：降温效果时序图（添加数据有效性检查）
    try
        fig = figure('Position', [100, 100, 1400, 1000], 'Visible', 'off');
        
        nAlgos = length(data.algorithms);
        plotCount = 0;
        
        for i = 1:nAlgos
            algo = data.algorithms{i};
            
            % 检查是否有必要的数据
            if isfield(data.evaluation, algo) && ...
               isfield(data.evaluation.(algo), 'actual_coolings') && ...
               isfield(data.evaluation.(algo), 'target_coolings')
                
                plotCount = plotCount + 1;
                actual = data.evaluation.(algo).actual_coolings;
                target = data.evaluation.(algo).target_coolings;
                nSteps = length(actual);
                time = (0:nSteps-1) * 0.5;  % 每步0.5小时
                
                subplot(nAlgos, 1, i);
                plot(time, target, 'r--', 'LineWidth', 2, 'DisplayName', '目标降温');
                hold on;
                plot(time, actual, 'b-', 'LineWidth', 1.5, 'DisplayName', '实际降温');
                
                % 添加误差带
                fill([time fliplr(time)], ...
                     [target'+1 fliplr(target'-1)], ...
                     'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', ...
                     'DisplayName', '±1°C误差带');
                
                xlabel('时间 (小时)', 'FontSize', 11);
                ylabel('降温量 (°C)', 'FontSize', 11);
                title([data.algorithmNames{i} ' - 降温效果'], ...
                      'FontSize', 12, 'FontWeight', 'bold');
                legend('Location', 'best', 'FontSize', 9);
                grid on;
            end
        end
        
        if plotCount > 0
            sgtitle('各算法降温效果时序对比', 'FontSize', 14, 'FontWeight', 'bold');
            
            saveas(fig, fullfile(outputPath, 'Chinese', '10_降温效果时序对比.png'));
            savefig(fig, fullfile(outputPath, 'Chinese', '10_降温效果时序对比.fig'));
            
            % 英文版
            for i = 1:nAlgos
                if isfield(data.evaluation, data.algorithms{i}) && ...
                   isfield(data.evaluation.(data.algorithms{i}), 'actual_coolings')
                    subplot(nAlgos, 1, i);
                    h = get(gca, 'Children');
                    for j = 1:length(h)
                        if strcmp(get(h(j), 'Type'), 'line')
                            name = get(h(j), 'DisplayName');
                            if strcmp(name, '目标降温')
                                set(h(j), 'DisplayName', 'Target Cooling');
                            elseif strcmp(name, '实际降温')
                                set(h(j), 'DisplayName', 'Actual Cooling');
                            elseif contains(name, '误差带')
                                set(h(j), 'DisplayName', '±1°C Error Band');
                            end
                        end
                    end
                    xlabel('Time (hours)', 'FontSize', 11);
                    ylabel('Cooling Amount (°C)', 'FontSize', 11);
                    title([data.algorithmNames{i} ' - Cooling Performance'], ...
                          'FontSize', 12, 'FontWeight', 'bold');
                end
            end
            
            sgtitle('Cooling Performance Time Series Comparison', 'FontSize', 14, 'FontWeight', 'bold');
            
            saveas(fig, fullfile(outputPath, 'English', '10_cooling_time_series_comparison.png'));
            savefig(fig, fullfile(outputPath, 'English', '10_cooling_time_series_comparison.fig'));
        else
            warning('  ⚠ 降温时序图: 没有可用的降温数据');
        end
        
        close(fig);
    catch ME
        warning(['降温效果时序图生成失败: ' ME.message]);
        if ~isempty(ME.stack)
            fprintf('  错误位置: %s (第 %d 行)\n', ME.stack(1).name, ME.stack(1).line);
        end
    end
end
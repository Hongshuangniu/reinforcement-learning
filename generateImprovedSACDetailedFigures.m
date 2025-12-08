function generateImprovedSACDetailedFigures(resultsPath, outputPath)
% 为 Improved SAC 算法生成详细的单独分析图表（基于降温能力评价）
%
% 🔥 修复内容：
% 1. ✅ 修复 nSteps 未定义错误
% 2. ✅ 添加数据有效性检查
% 3. ✅ 改善错误处理
%
% 输入:
%   resultsPath - Python导出的matlab_data路径
%   outputPath  - 输出图表路径

if nargin < 1
    resultsPath = 'matlab_data';
end
if nargin < 2
    outputPath = 'results/figures/ImprovedSAC';
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

fprintf('\n========== 生成 Improved SAC 详细分析图表（降温能力评价）==========\n');

% 加载数据
try
    data = loadImprovedSACDataFromPython(resultsPath);
    fprintf('✓ Python数据加载成功\n');
catch ME
    error(['数据加载失败: ' ME.message]);
end

% 生成各类图表
try
    fprintf('\n生成图表序列...\n');
    
    % 1. 训练动态特性
    fprintf('  1/8 训练动态特性...\n');
    generateTrainingDynamics(data, outputPath);
    
    % 2. 降温能力分析（核心）
    fprintf('  2/8 降温能力分析...\n');
    generateCoolingPerformanceAnalysis(data, outputPath);
    
    % 3. 温度控制效果
    fprintf('  3/8 温度控制效果...\n');
    generateTemperatureControl(data, outputPath);
    
    % 4. 降温时序分析
    fprintf('  4/8 降温时序分析...\n');
    generateCoolingTimeSeries(data, outputPath);
    
    % 5. 控制动作分析
    fprintf('  5/8 控制动作分析...\n');
    generateActionAnalysis(data, outputPath);
    
    % 6. 学习曲线
    fprintf('  6/8 学习曲线...\n');
    generateLearningCurves(data, outputPath);
    
    % 7. 熵调节分析
    fprintf('  7/8 熵调节分析...\n');
    generateEntropyAnalysis(data, outputPath);
    
    % 8. 综合性能指标
    fprintf('  8/8 综合性能指标...\n');
    generatePerformanceMetrics(data, outputPath);
    
    fprintf('\n✓ Improved SAC 详细分析图表生成完成！\n');
    fprintf('  输出路径: %s\n', outputPath);
catch ME
    warning(['图表生成出错: ' ME.message]);
    if ~isempty(ME.stack)
        fprintf('  错误位置: %s (第 %d 行)\n', ME.stack(1).name, ME.stack(1).line);
    end
end
end

%% ========== 数据加载函数 ==========
function data = loadImprovedSACDataFromPython(resultsPath)
    data = struct();
    
    % 1. 加载训练数据
    trainFile = fullfile(resultsPath, 'training_improved_sac.mat');
    if exist(trainFile, 'file')
        trainData = load(trainFile);
        
        data.stats = struct();
        
        if isfield(trainData, 'episode_rewards')
            data.stats.episodeReward = double(trainData.episode_rewards(:)');
            data.stats.qValue = data.stats.episodeReward;
            nEpisodes = length(data.stats.qValue);
        else
            nEpisodes = 0;
        end
        
        if isfield(trainData, 'critic_losses') && ~isempty(trainData.critic_losses)
            data.stats.criticLoss = double(trainData.critic_losses(:)');
        else
            data.stats.criticLoss = [];
        end
        
        if isfield(trainData, 'actor_losses') && ~isempty(trainData.actor_losses)
            data.stats.actorLoss = double(trainData.actor_losses(:)');
        else
            data.stats.actorLoss = [];
        end
        
        if isfield(trainData, 'entropies') && ~isempty(trainData.entropies)
            data.stats.entropy = double(trainData.entropies(:)');
        else
            data.stats.entropy = [];
        end
        
        if isfield(trainData, 'alphas') && ~isempty(trainData.alphas)
            data.stats.alpha = double(trainData.alphas(:)');
        else
            data.stats.alpha = [];
        end
        
        fprintf('  ✓ 加载训练数据: %d episodes\n', nEpisodes);
    else
        error('未找到训练文件: %s', trainFile);
    end
    
    % 2. 加载评估数据
    evalFile = fullfile(resultsPath, 'evaluation_improved_sac.mat');
    if exist(evalFile, 'file')
        evalData = load(evalFile);
        
        % 🔥 修复：初始化evaluation结构和nSteps
        data.evaluation = struct();
        data.nSteps = 0;  % 默认值
        
        % 温度数据
        if isfield(evalData, 'episode1_true_temps')
            actualTemps = double(evalData.episode1_true_temps(:));
            data.nSteps = length(actualTemps);  % 🔥 保存到data结构中
            timeVec = (0:data.nSteps-1)' * 0.5;
            
            data.evaluation.Time = timeVec;
            data.evaluation.Actual = actualTemps;
        end
        
        % 降温数据
        if isfield(evalData, 'episode1_actual_coolings')
            data.evaluation.ActualCooling = double(evalData.episode1_actual_coolings(:));
        end
        
        if isfield(evalData, 'episode1_target_coolings')
            data.evaluation.TargetCooling = double(evalData.episode1_target_coolings(:));
        end
        
        % 动作数据
        if isfield(evalData, 'episode1_actions')
            actionMat = double(evalData.episode1_actions);
            % 🔥 修复：确保是正确的维度 (nSteps x 3)
            if size(actionMat, 1) < size(actionMat, 2)
                actionMat = actionMat';
            end
            data.evaluation.Actions = actionMat;
        end

        % ===== 计算完整的metrics =====
        data.metrics = struct();
        
        % 基础误差指标
        if isfield(evalData, 'cooling_mae')
            data.metrics.mae = double(evalData.cooling_mae);
        elseif isfield(evalData, 'MAE')
            data.metrics.mae = double(evalData.MAE);
        else
            data.metrics.mae = 0;
        end
        
        if isfield(evalData, 'cooling_rmse')
            data.metrics.rmse = double(evalData.cooling_rmse);
        elseif isfield(evalData, 'RMSE')
            data.metrics.rmse = double(evalData.RMSE);
        else
            data.metrics.rmse = 0;
        end
        
        if isfield(evalData, 'cooling_max_error')
            data.metrics.maxError = double(evalData.cooling_max_error);
        elseif isfield(evalData, 'MaxAE')
            data.metrics.maxError = double(evalData.MaxAE);
        else
            data.metrics.maxError = 0;
        end
        
        % 工业控制指标
        data.metrics.ise = getFieldOrDefault(evalData, 'ISE', 0);
        data.metrics.iae = getFieldOrDefault(evalData, 'IAE', 0);
        data.metrics.itae = getFieldOrDefault(evalData, 'ITAE', 0);
        
        % 动态性能指标
        data.metrics.settling_time = getFieldOrDefault(evalData, 'settling_time', 0);
        data.metrics.overshoot = getFieldOrDefault(evalData, 'peak_overshoot', 0);
        data.metrics.steadyStateError = getFieldOrDefault(evalData, 'steady_state_error', 0);
        
        % 控制精度指标
        data.metrics.precision_2c = getFieldOrDefault(evalData, 'control_precision_2C', 0);
        data.metrics.precision_1c = getFieldOrDefault(evalData, 'control_precision_1C', 0);
        data.metrics.tempStability = getFieldOrDefault(evalData, 'temperature_stability', 0);
        
        % 能效指标
        data.metrics.totalEnergy = getFieldOrDefault(evalData, 'total_energy', 0);
        data.metrics.energyEfficiency = getFieldOrDefault(evalData, 'energy_efficiency_ratio', 0);
        
        % 综合性能指标
        data.metrics.performanceIndex = getFieldOrDefault(evalData, 'total_performance_index', 0);
        data.metrics.precisionScore = getFieldOrDefault(evalData, 'precision_score', 0);
        data.metrics.efficiencyScore = getFieldOrDefault(evalData, 'efficiency_score', 0);
        data.metrics.stabilityScore = getFieldOrDefault(evalData, 'stability_score', 0);
        data.metrics.speedScore = getFieldOrDefault(evalData, 'speed_score', 0);
        
        fprintf('  ✓ 加载评估数据: %d 时间步\n', data.nSteps);
    else
        warning('未找到评估文件: %s', evalFile);
    end
end

function value = getFieldOrDefault(s, fieldName, defaultValue)
    % 辅助函数：获取字段值或默认值
    if isfield(s, fieldName)
        value = double(s.(fieldName));
    else
        value = defaultValue;
    end
end

%% ========== 图表生成函数 ==========

function generateTrainingDynamics(data, outputPath)
    % 训练动态特性
    try
        if ~isfield(data, 'stats') || ~isfield(data.stats, 'episodeReward')
            warning('没有训练统计数据');
            return;
        end
        
        fig = figure('Position', [100, 100, 1400, 600], 'Visible', 'off');
        
        episodes = 1:length(data.stats.qValue);
        episodeReward = data.stats.episodeReward;
        movingAvg = movmean(episodeReward, 10);
        
        % 绘制原始奖励和移动平均
        plot(episodes, episodeReward, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1, ...
            'DisplayName', '每回合奖励');
        hold on;
        plot(episodes, movingAvg, 'LineWidth', 2.5, 'Color', [0.2, 0.4, 0.8], ...
            'DisplayName', '10回合移动平均');
        
        xlabel('训练回合', 'FontSize', 14);
        ylabel('累计奖励', 'FontSize', 14);
        title('Improved SAC 训练动态特性', 'FontSize', 16, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 12);
        grid on;
        
        saveas(fig, fullfile(outputPath, 'Chinese', '01_训练动态特性.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '01_训练动态特性.fig'));
        
        % 英文版
        xlabel('Episodes', 'FontSize', 14);
        ylabel('Cumulative Reward', 'FontSize', 14);
        title('Improved SAC Training Dynamics', 'FontSize', 16, 'FontWeight', 'bold');
        h = legend;
        h.String{1} = 'Episode Reward';
        h.String{2} = '10-Episode Moving Average';
        
        saveas(fig, fullfile(outputPath, 'English', '01_training_dynamics.png'));
        savefig(fig, fullfile(outputPath, 'English', '01_training_dynamics.fig'));
        close(fig);
    catch ME
        warning(['训练动态特性图生成失败: ' ME.message]);
    end
end

function generateCoolingPerformanceAnalysis(data, outputPath)
    % 降温能力分析
    try
        if ~isfield(data, 'metrics')
            warning('没有性能指标数据');
            return;
        end
        
        fig = figure('Position', [100, 100, 1200, 800], 'Visible', 'off');
        
        % 1. 基础误差指标
        subplot(2, 3, 1);
        metrics1 = [data.metrics.mae, data.metrics.rmse, data.metrics.maxError];
        bar(metrics1, 'FaceColor', [0.25, 0.55, 0.85]);
        set(gca, 'XTickLabel', {'MAE', 'RMSE', 'MaxAE'});
        ylabel('误差 (°C)', 'FontSize', 11);
        title('基础误差指标', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        
        % 2. 工业控制指标
        subplot(2, 3, 2);
        metrics2 = [data.metrics.ise, data.metrics.iae, data.metrics.itae];
        bar(metrics2, 'FaceColor', [0.85, 0.45, 0.25]);
        set(gca, 'XTickLabel', {'ISE', 'IAE', 'ITAE'});
        ylabel('指标值', 'FontSize', 11);
        title('工业控制指标', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        
        % 3. 动态性能指标
        subplot(2, 3, 3);
        metrics3 = [data.metrics.settling_time, data.metrics.overshoot, ...
                    data.metrics.steadyStateError];
        bar(metrics3, 'FaceColor', [0.45, 0.75, 0.35]);
        set(gca, 'XTickLabel', {'调节时间', '超调量', '稳态误差'});
        ylabel('指标值', 'FontSize', 11);
        title('动态性能指标', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        
        % 4. 控制精度指标
        subplot(2, 3, 4);
        metrics4 = [data.metrics.precision_2c, data.metrics.precision_1c, ...
                    data.metrics.tempStability * 100];
        bar(metrics4, 'FaceColor', [0.75, 0.25, 0.65]);
        set(gca, 'XTickLabel', {'±2°C精度', '±1°C精度', '稳定性'});
        ylabel('百分比 (%)', 'FontSize', 11);
        title('控制精度指标', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        
        % 5. 能效指标
        subplot(2, 3, 5);
        metrics5 = [data.metrics.totalEnergy, data.metrics.energyEfficiency * 1000];
        bar(metrics5, 'FaceColor', [0.95, 0.65, 0.15]);
        set(gca, 'XTickLabel', {'总能耗', '能效比×1000'});
        ylabel('指标值', 'FontSize', 11);
        title('能效指标', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        
        % 6. 综合性能评分
        subplot(2, 3, 6);
        metrics6 = [data.metrics.precisionScore, data.metrics.efficiencyScore, ...
                    data.metrics.stabilityScore, data.metrics.speedScore];
        bar(metrics6, 'FaceColor', [0.35, 0.65, 0.95]);
        set(gca, 'XTickLabel', {'精度', '能效', '稳定', '速度'});
        ylabel('评分', 'FontSize', 11);
        title('综合性能评分', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        ylim([0 100]);
        
        sgtitle('Improved SAC 降温能力完整分析', 'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'Chinese', '02_降温能力分析.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '02_降温能力分析.fig'));
        
        % 英文版标题
        subplot(2, 3, 1);
        set(gca, 'XTickLabel', {'MAE', 'RMSE', 'MaxAE'});
        ylabel('Error (°C)', 'FontSize', 11);
        title('Basic Error Metrics', 'FontSize', 12, 'FontWeight', 'bold');
        
        subplot(2, 3, 2);
        title('Industrial Control Metrics', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Metric Value', 'FontSize', 11);
        
        subplot(2, 3, 3);
        set(gca, 'XTickLabel', {'Settling Time', 'Overshoot', 'SS Error'});
        ylabel('Metric Value', 'FontSize', 11);
        title('Dynamic Performance', 'FontSize', 12, 'FontWeight', 'bold');
        
        subplot(2, 3, 4);
        set(gca, 'XTickLabel', {'±2°C', '±1°C', 'Stability'});
        ylabel('Percentage (%)', 'FontSize', 11);
        title('Control Precision', 'FontSize', 12, 'FontWeight', 'bold');
        
        subplot(2, 3, 5);
        set(gca, 'XTickLabel', {'Total Energy', 'Efficiency×1000'});
        ylabel('Metric Value', 'FontSize', 11);
        title('Energy Efficiency', 'FontSize', 12, 'FontWeight', 'bold');
        
        subplot(2, 3, 6);
        set(gca, 'XTickLabel', {'Precision', 'Efficiency', 'Stability', 'Speed'});
        ylabel('Score', 'FontSize', 11);
        title('Performance Scores', 'FontSize', 12, 'FontWeight', 'bold');
        
        sgtitle('Improved SAC Cooling Performance Analysis', 'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'English', '02_cooling_analysis.png'));
        savefig(fig, fullfile(outputPath, 'English', '02_cooling_analysis.fig'));
        close(fig);
    catch ME
        warning(['降温能力分析图生成失败: ' ME.message]);
    end
end

function generateTemperatureControl(data, outputPath)
    % 🔥 温度控制效果（修复版 - 添加原始温度对比）
    try
        if ~isfield(data.evaluation, 'Time') || ~isfield(data.evaluation, 'Actual')
            warning('没有温度数据');
            return;
        end
        
        fig = figure('Position', [100, 100, 1400, 600], 'Visible', 'off');
        
        time = data.evaluation.Time;
        actual_temps = data.evaluation.Actual;
        
        % 🔥 计算原始温度（降温前）
        if isfield(data.evaluation, 'ActualCooling')
            original_temps = actual_temps + data.evaluation.ActualCooling;
            
            % 绘制原始温度和降温后温度的对比
            plot(time, original_temps, 'r--', 'LineWidth', 2, 'DisplayName', '原始温度（降温前）');
            hold on;
            plot(time, actual_temps, 'b-', 'LineWidth', 2.5, 'DisplayName', '实际温度（降温后）');
        else
            % 如果没有降温数据，只绘制实际温度
            plot(time, actual_temps, 'b-', 'LineWidth', 2.5, 'DisplayName', '实际温度');
            hold on;
        end
        
        % 添加温度区间标记
        yLimits = ylim;
        plot([min(time), max(time)], [75, 75], ...
            'Color', [0.8, 0.2, 0.2], 'LineStyle', '-.', 'LineWidth', 1.5, ...
            'DisplayName', '高温阈值 (75°C)');
        plot([min(time), max(time)], [65, 65], ...
            'Color', [1, 0.5, 0], 'LineStyle', '-.', 'LineWidth', 1.5, ...
            'DisplayName', '中温阈值 (65°C)');
        
        xlabel('时间 (小时)', 'FontSize', 14);
        ylabel('油温 (°C)', 'FontSize', 14);
        title('Improved SAC 温度控制效果对比', 'FontSize', 16, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 11);
        grid on;
        ylim(yLimits);
        
        saveas(fig, fullfile(outputPath, 'Chinese', '03_温度控制效果.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '03_温度控制效果.fig'));
        
        % 英文版
        xlabel('Time (hours)', 'FontSize', 14);
        ylabel('Oil Temperature (°C)', 'FontSize', 14);
        title('Improved SAC Temperature Control Comparison', 'FontSize', 16, 'FontWeight', 'bold');
        h = legend;
        if length(h.String) >= 4
            h.String{1} = 'Original Temp (Before Cooling)';
            h.String{2} = 'Actual Temp (After Cooling)';
            h.String{3} = 'High Temp Threshold (75°C)';
            h.String{4} = 'Medium Temp Threshold (65°C)';
        end
        
        saveas(fig, fullfile(outputPath, 'English', '03_temperature_control.png'));
        savefig(fig, fullfile(outputPath, 'English', '03_temperature_control.fig'));
        close(fig);
    catch ME
        warning(['温度控制效果图生成失败: ' ME.message]);
    end
end
function generateCoolingTimeSeries(data, outputPath)
    % 降温时序分析
    try
        % 🔥 修复：添加数据有效性检查
        if ~isfield(data.evaluation, 'Time') || ...
           ~isfield(data.evaluation, 'ActualCooling') || ...
           ~isfield(data.evaluation, 'TargetCooling')
            warning('没有完整的降温数据');
            return;
        end
        
        fig = figure('Position', [100, 100, 1400, 600], 'Visible', 'off');
        
        % 🔥 修复：使用data.evaluation.Time而不是未定义的nSteps
        time = data.evaluation.Time;
        actual = data.evaluation.ActualCooling;
        target = data.evaluation.TargetCooling;
        
        % 绘制目标和实际降温
        plot(time, target, 'r--', 'LineWidth', 2.5, 'DisplayName', '目标降温');
        hold on;
        plot(time, actual, 'b-', 'LineWidth', 2, 'DisplayName', '实际降温');
        
        % 添加误差带
        fill([time; flipud(time)], [target+1; flipud(target-1)], ...
            'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', '±1°C误差带');
        
        xlabel('时间 (小时)', 'FontSize', 14);
        ylabel('降温量 (°C)', 'FontSize', 14);
        title('Improved SAC 降温效果时序分析', 'FontSize', 16, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 12);
        grid on;
        
        saveas(fig, fullfile(outputPath, 'Chinese', '04_降温时序分析.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '04_降温时序分析.fig'));
        
        % 英文版
        xlabel('Time (hours)', 'FontSize', 14);
        ylabel('Cooling Amount (°C)', 'FontSize', 14);
        title('Improved SAC Cooling Performance Time Series', 'FontSize', 16, 'FontWeight', 'bold');
        h = legend;
        h.String{1} = 'Target Cooling';
        h.String{2} = 'Actual Cooling';
        h.String{3} = '±1°C Error Band';
        
        saveas(fig, fullfile(outputPath, 'English', '04_cooling_time_series.png'));
        savefig(fig, fullfile(outputPath, 'English', '04_cooling_time_series.fig'));
        close(fig);
    catch ME
        warning(['降温时序分析图生成失败: ' ME.message]);
        fprintf('错误详情: %s\n', ME.message);
    end
end

function generateActionAnalysis(data, outputPath)
    % 控制动作分析
    try
        if ~isfield(data.evaluation, 'Actions') || ~isfield(data.evaluation, 'Time')
            warning('没有动作数据');
            return;
        end
        
        fig = figure('Position', [100, 100, 1400, 900], 'Visible', 'off');
        
        actions = data.evaluation.Actions;
        time = data.evaluation.Time;
        
        % 1. 泵压力
        subplot(3, 1, 1);
        plot(time, actions(:, 1), 'LineWidth', 2);
        xlabel('时间 (小时)', 'FontSize', 11);
        ylabel('压力 (kPa)', 'FontSize', 11);
        title('泵压力', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        ylim([2 5]);
        
        % 2. 帕尔贴开度
        subplot(3, 1, 2);
        plot(time, actions(:, 2), 'LineWidth', 2);
        xlabel('时间 (小时)', 'FontSize', 11);
        ylabel('开度 (0-1)', 'FontSize', 11);
        title('帕尔贴开度', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        ylim([0 1]);
        
        % 3. 阀门开度
        subplot(3, 1, 3);
        plot(time, actions(:, 3), 'LineWidth', 2);
        xlabel('时间 (小时)', 'FontSize', 11);
        ylabel('开度 (%)', 'FontSize', 11);
        title('阀门开度', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        ylim([0 100]);
        
        saveas(fig, fullfile(outputPath, 'Chinese', '05_控制动作分析.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '05_控制动作分析.fig'));
        
        % 英文版
        subplot(3, 1, 1);
        ylabel('Pressure (kPa)', 'FontSize', 11);
        title('Pump Pressure', 'FontSize', 12, 'FontWeight', 'bold');
        
        subplot(3, 1, 2);
        ylabel('Opening (0-1)', 'FontSize', 11);
        title('Peltier Opening', 'FontSize', 12, 'FontWeight', 'bold');
        
        subplot(3, 1, 3);
        xlabel('Time (hours)', 'FontSize', 11);
        ylabel('Opening (%)', 'FontSize', 11);
        title('Valve Opening', 'FontSize', 12, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'English', '05_action_analysis.png'));
        savefig(fig, fullfile(outputPath, 'English', '05_action_analysis.fig'));
        close(fig);
    catch ME
        warning(['控制动作分析图生成失败: ' ME.message]);
    end
end

function generateLearningCurves(data, outputPath)
    % 学习曲线
    try
        hasCriticLoss = isfield(data.stats, 'criticLoss') && ~isempty(data.stats.criticLoss);
        hasActorLoss = isfield(data.stats, 'actorLoss') && ~isempty(data.stats.actorLoss);
        
        if ~hasCriticLoss && ~hasActorLoss
            warning('没有损失数据');
            return;
        end
        
        fig = figure('Position', [100, 100, 1200, 500], 'Visible', 'off');
        
        hold on;
        if hasCriticLoss
            episodes = 1:length(data.stats.criticLoss);
            plot(episodes, movmean(data.stats.criticLoss, 20), 'LineWidth', 2, ...
                'DisplayName', 'Critic Loss');
        end
        
        if hasActorLoss
            episodes = 1:length(data.stats.actorLoss);
            plot(episodes, movmean(data.stats.actorLoss, 20), 'LineWidth', 2, ...
                'DisplayName', 'Actor Loss');
        end
        
        xlabel('训练步数', 'FontSize', 12);
        ylabel('损失', 'FontSize', 12);
        title('学习曲线（20步移动平均）', 'FontSize', 14, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 10);
        grid on;
        
        saveas(fig, fullfile(outputPath, 'Chinese', '06_学习曲线.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '06_学习曲线.fig'));
        
        % 英文版
        xlabel('Training Steps', 'FontSize', 12);
        ylabel('Loss', 'FontSize', 12);
        title('Learning Curves (20-Step Moving Average)', 'FontSize', 14, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'English', '06_learning_curves.png'));
        savefig(fig, fullfile(outputPath, 'English', '06_learning_curves.fig'));
        close(fig);
    catch ME
        warning(['学习曲线图生成失败: ' ME.message]);
    end
end

function generateEntropyAnalysis(data, outputPath)
    % 熵调节分析
    try
        hasEntropy = isfield(data.stats, 'entropy') && ~isempty(data.stats.entropy);
        hasAlpha = isfield(data.stats, 'alpha') && ~isempty(data.stats.alpha);
        
        if ~hasEntropy || ~hasAlpha
            warning('没有熵或alpha数据');
            return;
        end
        
        fig = figure('Position', [100, 100, 1200, 500], 'Visible', 'off');
        episodes = 1:length(data.stats.entropy);
        
        yyaxis left;
        plot(episodes, data.stats.entropy, 'LineWidth', 2.5);
        ylabel('策略熵', 'FontSize', 12);
        
        yyaxis right;
        plot(episodes, data.stats.alpha, 'LineWidth', 2.5);
        ylabel('熵系数 α', 'FontSize', 12);
        
        xlabel('训练步数', 'FontSize', 12);
        title('自适应熵调节', 'FontSize', 14, 'FontWeight', 'bold');
        legend({'策略熵', '熵系数'}, 'FontSize', 10);
        grid on;
        
        saveas(fig, fullfile(outputPath, 'Chinese', '07_熵调节分析.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '07_熵调节分析.fig'));
        
        % 英文版
        yyaxis left;
        ylabel('Entropy', 'FontSize', 12);
        yyaxis right;
        ylabel('Alpha', 'FontSize', 12);
        xlabel('Training Steps', 'FontSize', 12);
        title('Adaptive Entropy Tuning', 'FontSize', 14, 'FontWeight', 'bold');
        legend({'Entropy', 'Alpha'}, 'FontSize', 10);
        
        saveas(fig, fullfile(outputPath, 'English', '07_entropy_analysis.png'));
        savefig(fig, fullfile(outputPath, 'English', '07_entropy_analysis.fig'));
        close(fig);
    catch ME
        warning(['熵调节图生成失败: ' ME.message]);
    end
end

function generatePerformanceMetrics(data, outputPath)
    % 综合性能指标
    try
        if ~isfield(data, 'metrics')
            warning('没有指标数据');
            return;
        end
        
        fig = figure('Position', [100, 100, 1600, 1200], 'Visible', 'off');

        % 指标分组
        metrics_groups = {
            {'mae', 'rmse', 'maxError'}, ...
            {'ise', 'iae', 'itae'}, ...
            {'settling_time', 'overshoot', 'steadyStateError'}, ...
            {'precision_2c', 'precision_1c', 'tempStability'}, ...
            {'totalEnergy', 'energyEfficiency'}, ...
            {'precisionScore', 'efficiencyScore', 'stabilityScore', 'speedScore'}
        };
        
        group_titles_cn = {
            '基础误差指标', '工业控制指标', '动态性能指标', ...
            '控制精度指标', '能效指标', '综合性能评分'
        };
        
        group_titles_en = {
            'Basic Error Metrics', 'Industrial Control', 'Dynamic Performance', ...
            'Control Precision', 'Energy Efficiency', 'Performance Scores'
        };
        
        metric_names_cn = {
            {'MAE (°C)', 'RMSE (°C)', 'MaxAE (°C)'}, ...
            {'ISE', 'IAE', 'ITAE'}, ...
            {'调节时间', '超调量(%)', '稳态误差(°C)'}, ...
            {'±2°C(%)', '±1°C(%)', '稳定性×100'}, ...
            {'总能耗', '能效比×1000'}, ...
            {'精度分', '能效分', '稳定分', '速度分'}
        };
        
        metric_names_en = {
            {'MAE (°C)', 'RMSE (°C)', 'MaxAE (°C)'}, ...
            {'ISE', 'IAE', 'ITAE'}, ...
            {'Settling', 'Overshoot', 'SS Error'}, ...
            {'±2°C(%)', '±1°C(%)', 'Stability×100'}, ...
            {'Energy', 'Efficiency×1000'}, ...
            {'Precision', 'Efficiency', 'Stability', 'Speed'}
        };
        
        % 绘制6个子图
        for g = 1:6
            subplot(3, 2, g);
            
            current_metrics = metrics_groups{g};
            values = zeros(1, length(current_metrics));
            
            for m = 1:length(current_metrics)
                metric_name = current_metrics{m};
                if isfield(data.metrics, metric_name)
                    val = data.metrics.(metric_name);
                    % 特殊处理：能效比和稳定性需要放大
                    if strcmp(metric_name, 'energyEfficiency')
                        val = val * 1000;
                    elseif strcmp(metric_name, 'tempStability')
                        val = val * 100;
                    end
                    values(m) = val;
                end
            end
            
            bar(values, 'FaceColor', [0.25, 0.55, 0.85]);
            set(gca, 'XTickLabel', metric_names_cn{g}, 'XTickLabelRotation', 20);
            ylabel('指标值', 'FontSize', 11);
            title(group_titles_cn{g}, 'FontSize', 13, 'FontWeight', 'bold');
            grid on;
            
            % 在柱子上显示数值
            for m = 1:length(values)
                if values(m) ~= 0
                    text(m, values(m), sprintf('%.2f', values(m)), ...
                        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                        'FontSize', 9);
                end
            end
        end
        
        sgtitle('Improved SAC 综合性能指标（完整降温能力评价体系）', ...
            'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'Chinese', '08_综合性能指标.png'));
        savefig(fig, fullfile(outputPath, 'Chinese', '08_综合性能指标.fig'));
        
        % 英文版
        for g = 1:6
            subplot(3, 2, g);
            set(gca, 'XTickLabel', metric_names_en{g});
            ylabel('Metric Value', 'FontSize', 11);
            title(group_titles_en{g}, 'FontSize', 13, 'FontWeight', 'bold');
        end
        sgtitle('Improved SAC Performance Metrics (Cooling-Based Evaluation)', ...
            'FontSize', 16, 'FontWeight', 'bold');
        
        saveas(fig, fullfile(outputPath, 'English', '08_performance_metrics.png'));
        savefig(fig, fullfile(outputPath, 'English', '08_performance_metrics.fig'));
        close(fig);
    catch ME
        warning(['综合性能指标图生成失败: ' ME.message]);
        fprintf('错误详情: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('错误位置: %s (第 %d 行)\n', ME.stack(1).name, ME.stack(1).line);
        end
    end
end
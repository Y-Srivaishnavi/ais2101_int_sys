%loading data
data = readtable('winequality-red.csv', 'Delimiter', ';', VariableNamingRule='preserve');

head(data);

X = data{:, 1:end-1}; % All columns except the last one
Y = data{:, end}; % Last column is the output (wine quality)

%splitting into training and testing sets
N = size(X, 1);
trainRatio = 0.8;
trainInd = 1:round(trainRatio*N);
testInd = round(trainRatio*N)+1:N;

X_train = X(trainInd, :);
Y_train = Y(trainInd);
X_test = X(testInd, :);
Y_test = Y(testInd);

function rmse = tuneFIS(fis, X_train, Y_train, params)
    fis = updateFIS(fis, params);
    Y_pred = evalfis(fis, X_train);
    rmse = sqrt(mean((Y_pred - Y_train).^2));
end

function fisUpdated = updateFIS(fis, params)
    fisUpdated = fis; 
    
    numInputs = length(fis.Inputs);
    for i = 1:numInputs
        lowerRange = params(2*i - 1); 
        upperRange = params(2*i);
        
        if upperRange <= lowerRange
            upperRange = lowerRange + 1;
        end
        
        fisUpdated.Inputs(i).Range = [lowerRange, upperRange];
    end
end


%% Generating FIS using genfis
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = 3;  %limiting to 3 mfs per input
fis = genfis(X_train, Y_train, opt);

disp(fis);

%% 4a: Applying GA to tune FIS
optionsGA = optimoptions('ga', 'Display', 'iter', 'MaxGenerations', 50);
lb = [0, 0, 0, 0, 0, 0];   % Lower bounds for input range
ub = [10, 10, 10, 10, 10, 10];   % Upper bounds

objectiveGA = @(params) tuneFIS(fis, X_train, Y_train, params);

[optimalParamsGA, ~] = ga(objectiveGA, 6, [], [], [], [], lb, ub, [], optionsGA);

%% Update FIS with tuned params
fisGA = updateFIS(fis, optimalParamsGA);

%evaluating performance on test data
Y_predGA = evalfis(fisGA, X_test);
RMSE_GA = sqrt(mean((Y_predGA - Y_test).^2));
fprintf('GA Tuning RMSE: %.4f\n', RMSE_GA);

%% Visualizing GA-Tuned FIS
Y_pred_GA = evalfis(fisGA, X_test);

figure;
plot(Y_test, Y_pred_GA, 'o');
hold on;
plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--');
xlabel('Actual Output');
ylabel('Predicted Output (GA-Tuned FIS)');
title('Actual vs Predicted Output (GA-Tuned FIS)');
grid on;
hold off;

%% 4b: Applying PSO to tune FIS
optionsPSO = optimoptions('particleswarm', 'Display', 'iter', 'MaxIterations', 50);
objectivePSO = @(params) tuneFIS(fis, X_train, Y_train, params);

[optimalParamsPSO, ~] = particleswarm(objectivePSO, length(fis.Inputs), [], [], optionsPSO);

%% Update FIS with tuned params
fisPSO = updateFIS(fis, optimalParamsPSO);

Y_predPSO = evalfis(fisPSO, X_test);
RMSE_PSO = sqrt(mean((Y_predPSO - Y_test).^2));
fprintf('PSO Tuning RMSE: %.4f\n', RMSE_PSO);

%% Visualize PSO-Tuned FIS
Y_pred_PSO = evalfis(fisPSO, X_test);
figure;
plot(Y_test, Y_pred_PSO, 'o');
hold on;
plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--');
xlabel('Actual Output');
ylabel('Predicted Output (PSO-Tuned FIS)');
title('Actual vs Predicted Output (PSO-Tuned FIS)');
grid on;
hold off;

%% 4c: Applying ANFIS to tune FIS
optionsANFIS = anfisOptions('InitialFIS', fis, 'EpochNumber', 50, 'OptimizationMethod', 2);
fisANFIS = anfis([X_train, Y_train], optionsANFIS);
Y_predANFIS = evalfis(fisANFIS, X_test);
RMSE_ANFIS = sqrt(mean((Y_predANFIS - Y_test).^2));
fprintf('ANFIS Tuning RMSE: %.4f\n', RMSE_ANFIS);

%% Visualize ANFIS-Tuned FIS
Y_pred_ANFIS = evalfis(fisANFIS, X_test);
figure;
plot(Y_test, Y_pred_ANFIS, 'o');
hold on;
plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--');
xlabel('Actual Output');
ylabel('Predicted Output (ANFIS-Tuned FIS)');
title('Actual vs Predicted Output (ANFIS-Tuned FIS)');
grid on;
hold off;

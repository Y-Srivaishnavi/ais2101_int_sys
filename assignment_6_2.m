% Step 1: Load Wine Quality Dataset
data = readtable('winequality-red.csv', 'Delimiter', ';', VariableNamingRule='preserve');

% Display the first few rows of the data
head(data);

% Separate input variables (features) and output (quality)
X = data{:, 1:end-1}; % All columns except the last one
Y = data{:, end}; % Last column is the output (wine quality)

% Split data into training and testing sets
N = size(X, 1);
trainRatio = 0.8;
trainInd = 1:round(trainRatio*N);
testInd = round(trainRatio*N)+1:N;

X_train = X(trainInd, :);
Y_train = Y(trainInd);
X_test = X(testInd, :);
Y_test = Y(testInd);

% Step 3: Generate Fuzzy Inference System (FIS) using genfis
% Set grid partition for FIS generation
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = 3;  % Limit to 3 membership functions per input
fis = genfis(X_train, Y_train, opt);

% Display the FIS
disp(fis);

% Step 4a: Apply Genetic Algorithm to tune the FIS
% GA Tuning (if Global Optimization Toolbox is installed)
optionsGA = optimoptions('ga', 'Display', 'iter', 'MaxGenerations', 50);

% Define the lower and upper bounds for the parameters (e.g., for 3 inputs)
lb = [0, 0, 0, 0, 0, 0];   % Lower bounds for each input range
ub = [10, 10, 10, 10, 10, 10];   % Upper bounds for each input range

% Objective function for GA
objectiveGA = @(params) tuneFIS(fis, X_train, Y_train, params);

% Run GA with the correct number of variables and bounds
[optimalParamsGA, ~] = ga(objectiveGA, 6, [], [], [], [], lb, ub, [], optionsGA);

% Update FIS with the tuned parameters
fisGA = updateFIS(fis, optimalParamsGA);

% Evaluate the performance on the test data
Y_predGA = evalfis(fisGA, X_test);
RMSE_GA = sqrt(mean((Y_predGA - Y_test).^2));
fprintf('GA Tuning RMSE: %.4f\n', RMSE_GA);


% Step 4b: Apply Particle Swarm Optimization (PSO) to tune the FIS
optionsPSO = optimoptions('particleswarm', 'Display', 'iter', 'MaxIterations', 50);

% Objective function for PSO
objectivePSO = @(params) tuneFIS(fis, X_train, Y_train, params);

% Run PSO
[optimalParamsPSO, ~] = particleswarm(objectivePSO, length(fis.Inputs), [], [], optionsPSO);

% Update FIS with the tuned parameters
fisPSO = updateFIS(fis, optimalParamsPSO);

% Evaluate the performance on the test data
Y_predPSO = evalfis(fisPSO, X_test);
RMSE_PSO = sqrt(mean((Y_predPSO - Y_test).^2));
fprintf('PSO Tuning RMSE: %.4f\n', RMSE_PSO);

% Step 4c: Apply ANFIS to tune the FIS
optionsANFIS = anfisOptions('InitialFIS', fis, 'EpochNumber', 50, 'OptimizationMethod', 2);
fisANFIS = anfis([X_train, Y_train], optionsANFIS);

% Evaluate performance on test data
Y_predANFIS = evalfis(fisANFIS, X_test);
RMSE_ANFIS = sqrt(mean((Y_predANFIS - Y_test).^2));
fprintf('ANFIS Tuning RMSE: %.4f\n', RMSE_ANFIS);

function rmse = tuneFIS(fis, X_train, Y_train, params)
    % Modify the FIS parameters using the provided params
    fis = updateFIS(fis, params);
    
    % Train the FIS on training data
    Y_pred = evalfis(fis, X_train);
    
    % Calculate the RMSE for the training data
    rmse = sqrt(mean((Y_pred - Y_train).^2));
end

function fisUpdated = updateFIS(fis, params)
    % Ensure params are valid and set proper ranges
    fisUpdated = fis;  % Copy original FIS to modify
    
    numInputs = length(fis.Inputs);  % Number of inputs in FIS
    
    % Loop through each input and set the range
    for i = 1:numInputs
        % Extract the lower and upper range for each input
        lowerRange = params(2*i - 1);   % Lower bound for input i
        upperRange = params(2*i);       % Upper bound for input i
        
        % Ensure the upper range is greater than the lower range
        if upperRange <= lowerRange
            % If invalid, set default values or adjust the range
            upperRange = lowerRange + 1;  % Ensure valid range
        end
        
        % Set the range for the i-th input
        fisUpdated.Inputs(i).Range = [lowerRange, upperRange];
    end
end


%% Visulaization

% Predictions for GA-Tuned FIS
Y_pred_GA = evalfis(fisGA, X_test);

% Predictions for PSO-Tuned FIS
Y_pred_PSO = evalfis(fisPSO, X_test);

% Predictions for ANFIS-Tuned FIS
Y_pred_ANFIS = evalfis(fisANFIS, X_test);

% Actual vs Predicted (GA)
figure;
plot(Y_test, Y_pred_GA, 'o');
hold on;
plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--');
xlabel('Actual Output');
ylabel('Predicted Output (GA-Tuned FIS)');
title('Actual vs Predicted Output (GA-Tuned FIS)');
grid on;
hold off;

% Actual vs Predicted (PSO)
figure;
plot(Y_test, Y_pred_PSO, 'o');
hold on;
plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--');
xlabel('Actual Output');
ylabel('Predicted Output (PSO-Tuned FIS)');
title('Actual vs Predicted Output (PSO-Tuned FIS)');
grid on;
hold off;

% Actual vs Predicted (ANFIS)
figure;
plot(Y_test, Y_pred_ANFIS, 'o');
hold on;
plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--');
xlabel('Actual Output');
ylabel('Predicted Output (ANFIS-Tuned FIS)');
title('Actual vs Predicted Output (ANFIS-Tuned FIS)');
grid on;
hold off;

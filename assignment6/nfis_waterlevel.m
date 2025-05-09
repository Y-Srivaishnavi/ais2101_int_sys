clc; clear; close all;

%% 1:Creating random input data
inputData = zeros(1000, 2);
inputData(:,1) = 10 * rand(1000, 1);      % Level: 0 to 10
inputData(:,2) = -5 + 10 * rand(1000, 1);  % Rate: -5 to 5

%% 2:Building water tank FIS
fis = mamfis('Name', 'WaterTankController');

%input 1- Error
fis = addInput(fis, [-50 50], 'Name', 'Error');  % assuming setpoint = 0
fis = addMF(fis, 'Error', 'trimf', [-50 -50 0], 'Name', 'Low');
fis = addMF(fis, 'Error', 'trimf', [-10 0 10], 'Name', 'Zero');
fis = addMF(fis, 'Error', 'trimf', [0 50 50], 'Name', 'High');

%input 2- Derivative of error
fis = addInput(fis, [-10 10], 'Name', 'dError');
fis = addMF(fis, 'dError', 'trimf', [-10 -10 0], 'Name', 'Negative');
fis = addMF(fis, 'dError', 'trimf', [-5 0 5], 'Name', 'Zero');
fis = addMF(fis, 'dError', 'trimf', [0 10 10], 'Name', 'Positive');

%output- Valve Adjustment
fis = addOutput(fis, [0 100], 'Name', 'Valve');
fis = addMF(fis, 'Valve', 'trimf', [0 0 25], 'Name', 'CloseFast');
fis = addMF(fis, 'Valve', 'trimf', [10 25 40], 'Name', 'CloseSlow');
fis = addMF(fis, 'Valve', 'trimf', [40 50 60], 'Name', 'NoChange');
fis = addMF(fis, 'Valve', 'trimf', [60 75 90], 'Name', 'OpenSlow');
fis = addMF(fis, 'Valve', 'trimf', [75 100 100], 'Name', 'OpenFast');

%rules- same as simulink
ruleList = [
    "If Error is Zero and dError is Zero then Valve is NoChange"
    "If Error is Low and dError is Zero then Valve is OpenFast"
    "If Error is High and dError is Zero then Valve is CloseFast"
    "If Error is Zero and dError is Positive then Valve is CloseSlow"
    "If Error is Zero and dError is Negative then Valve is OpenSlow"
];

fis = addRule(fis, ruleList);

%% 3:Generating targets using FIS
targets = evalfis(fis, inputData);

%% 4:Preparing dataset 
dataset = [inputData, targets];
trainingRatio = 0.8;

cv = cvpartition(1000, 'HoldOut', 1 - trainingRatio);
trainIDX = cv.training;
testIDX = ~trainIDX;

trainData = dataset(trainIDX, :);
testData = dataset(testIDX, :);

trainInput = trainData(:, 1:end-1);
trainTarget = trainData(:, end);
testInput = testData(:, 1:end-1);
testTarget = testData(:, end);

%% 5:Creating initial neuro-fuzzy system
fis_init = genfis1([trainInput trainTarget], 3, 'trimf', 'linear');

%% 6:Training Neuro-FIS
TrainOptions = [200 0 0.01 0.9 1.1];
DisplayOptions = [1 1 1 1];
OptimizationMethod = 1;

anfisFIS = anfis([trainInput trainTarget], fis_init, TrainOptions, DisplayOptions, [], OptimizationMethod);

%% 7:Generating outputs for train and test Sets
trainOutputs = evalfis(trainInput, anfisFIS);
testOutputs = evalfis(testInput, anfisFIS);

%% 8:Calculating errors for test data
testErrors = testTarget - testOutputs;
MSE = mean(testErrors.^2);
RMSE = sqrt(MSE);
errorMean = mean(testErrors);
errorSTD = std(testErrors);

fprintf('Testing MSE: %.6f\n', MSE);
fprintf('Testing RMSE: %.6f\n', RMSE);
fprintf('Testing Error Mean: %.6f\n', errorMean);
fprintf('Testing Error STD: %.6f\n', errorSTD);

%% 9:Plotting test Target vs Output
figure;
plot(testTarget, testOutputs, 'bo');
xlabel('Test Target');
ylabel('Test Output');
title('Test Target vs Test Output (Neuro-Fuzzy)');
grid on;

%% 10:Testing inputs beyond UOD
inputBeyondUOD = [15 7; 20 -10; 25 5];

fisOutputs = evalfis(fis, inputBeyondUOD);
anfisOutputs = evalfis(anfisFIS, inputBeyondUOD);

figure;
subplot(2,1,1);
plot(fisOutputs, 'ro-', 'DisplayName', 'FIS Outputs');
hold on;
plot(anfisOutputs, 'bx--', 'DisplayName', 'Neuro-Fuzzy Outputs');
xlabel('Data Point');
ylabel('Valve Signal');
title('Comparison: FIS vs Neuro-Fuzzy (Extrapolated Inputs)');
legend('Location', 'Best');
grid on;

%% Plotting surface plot of trained neuro-fuzzy system
[x, y] = meshgrid(linspace(0,10,50), linspace(-5,5,50)); %Creating grid over input domain
xy = [x(:), y(:)]; 
z = evalfis(anfisFIS, xy);
z = reshape(z, size(x)); %Reshaping output to grid format

figure;
surf(x, y, z);
xlabel('Water Level (cm)');
ylabel('Inflow Rate (cm/min)');
zlabel('Valve Signal');
title('Surface View of Trained Neuro-Fuzzy Inference System');
grid on;
shading interp;

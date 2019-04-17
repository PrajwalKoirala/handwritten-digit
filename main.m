
%% Initialization
clear ; close all;

input_layer_size  = 784;  % 29x28 Input Images of Digits
hidden_layer_size = 200;   % 2000 hidden units
num_labels = 10;           % 10 labels for 10 digits

%% =========== Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading  Data ...\n')


% 
% X = csvread('features.csv');
% y = csvread('labels.csv');

% data1 = csvread('mnist1.csv');
% data2 = csvread('mnist2.csv');
% data = [data1; data2];
data = csvread('mnist.csv');
X = data(:, 2:785);
X = X/255;
yy = data(:, 1);
yy = yy + (yy==0)*10;
y = zeros(length(yy),10);
for i = 1: length(yy)
    y(i, yy(i)) = 1;
end


m = size(X, 1);





fprintf('Program paused. Press enter to continue.\n');
pause;




%% ================ Initializing Pameters ================


fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);


% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];




%% =================== Training NN ===================

fprintf('\nTraining Neural Network... \n')


options = optimset('MaxIter', 400);

lambda = 0.05;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;





%% =================   Predict & compute accuracy =================

pred = predict(Theta1, Theta2, X);
[asdf yy] = max(y, [], 2);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == yy)) * 100);

csvwrite("param1.csv",Theta1);
csvwrite("param2.csv",Theta2);

wih = Theta1(:,2:785);
bih = Theta1(:,1);
who = Theta2(:,2:201);
bho = Theta2(:,1);

csvwrite('who.csv', who);
csvwrite('wih.csv', wih);
csvwrite('bho.csv', bho);
csvwrite('bih.csv', bih);
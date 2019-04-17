function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

             
% Setup of some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


yy = y;



X = [ones(m,1) X];
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2 * Theta2';
h = sigmoid(z3);
j = yy .* log(h) + (1-yy) .* log(1-h);
J = -1/m * sum(sum(j));

regCost = lambda/2/m * ( sum(sum(Theta1(:,2:input_layer_size+1).^2)) + sum(sum(Theta2(:,2:hidden_layer_size+1) .^2)) );
J = J + regCost;





%%%%%%%%%%%%%%%%%%%%%%%%%

%output_error = h - y;
%hidden_error =  output_error * Theta2 .* [ones(m,1) sigmoidGradient(z2)];

%hidden_error = hidden_error(:,2:end);

%delta1 = hidden_error' * X;
%delta2 = output_error' * a2;

%Theta1_grad = 1/m * delta1;
%Theta2_grad = 1/m * delta2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for i = 1:m
y_3 = yy(i,:);
a_1 = X(i,:);
z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2,1),1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);
error3 = a_3 - y_3;
error2 =  error3 * Theta2 .* [1 sigmoidGradient(z_2)];
error2_trans = error2';
delta1 = delta1 + error2_trans(2:end,:) * a_1;
delta2 = delta2 + error3.' * a_2;
end

reg_1 = zeros(size(delta1));
reg_2 = zeros(size(delta2));
reg_1(:,2:end) = lambda/m * Theta1(:,2:end);
reg_2(:,2:end) = lambda/m * Theta2(:,2:end);
Theta1_grad = 1/m * delta1 + reg_1;
Theta2_grad = 1/m * delta2 + reg_2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

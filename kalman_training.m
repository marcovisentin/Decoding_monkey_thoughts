% -- TRAINING KALMAN ALGORITHM  --

%Preprocessed data (spikes = observation and positions = state) given as
%input to calculate the matrices A,W,H,Q necessary to prediction

function [A,W,H,Q] = kalman_training(spikes_train, xy_positions_train)

% Initialize
n_dim = 2;
n_neurons = 98;
A1 = zeros(n_dim);
A2 = zeros(n_dim);
W1 = zeros(n_dim);
W2 = zeros(n_dim);
H1 = zeros(n_neurons,n_dim);
H2 = zeros(n_dim);
Q1 = zeros(n_neurons);
Q2 = zeros(n_dim,n_neurons);

for i = 1:length(spikes_train)
    %Hand Kinematics X:
    X=xy_positions_train{i,1}; %states: the positions we predicting (dimension x n_bin)
    M = length(X); 
    %Firing rates Z:
    Z=spikes_train{i,1}; %observations: neural data (n_neuron x n_bin)

    %Compute the parameters according to the equations found in Wu et.al:
    %A: Transition matrix of state (from xt to xt+1) using least squares
    X2=X(:,2:M); 
    X1=X(:,1:M-1);
    A1 = A1 + X2*X1';
    A2 = A2 + X1*X1';
    %W: noise covariance of state 
    W1 = W1 + (1/(M-1)) * (X2*X2');
    W2 = W2 + (1/(M-1)) * (X1*X2');

    %H: transition matrix of observation (transformation from position to spikes)
    H1 = H1 + Z*X';
    H2 = H2 + X*X';

    %Q: Noise covariance of observation
    Q1 = Q1 + (1/M) * (Z*Z');
    Q2 = Q2 + (1/M) * (X*Z');
end
A = A1 / A2;
W = W1 - A * W2;
H = H1/H2;
Q = Q1 - H*Q2;
end

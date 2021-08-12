% -- TRAINING ALGORITHM --

function [modelParameters] = positionEstimatorTraining(trainingData)
    
% Initialization

 bin_size = 20;             % 20 ms 
 lag = 60/bin_size;
 bias = 300/bin_size;
 n_angles = size(trainingData,2);  
 mean_firing_sum = 0;
 n_neurons = size(trainingData(1,1).spikes,1);
 response=cell(size(trainingData,1),size(trainingData,2));    % Cell for spikes data
 positions_train=cell(size(trainingData,1),size(trainingData,2));  % Cell for position data
 
for i=1:size(trainingData,1)
    for k=1:size(trainingData,2)
         
         % For each trial, sum of spikes across 20 ms is calculated (firing rate)
         % Sqrt applied to make more Gaussian the distribution of firing rate
         % this is useful to improve the performance of Kalman filter
         
         spikes_train = sqrt(make_spikes_matrix(trainingData,i,k,bin_size, n_neurons,lag, bias));
         response{i,k}=spikes_train;
         
         % this is to calculate average
         mean_firing = mean(spikes_train, 'all');
         mean_firing_sum = mean_firing_sum + mean_firing;
         
         % track positions for each trial every 20 ms
         xy_positions_train = make_position_matrix(trainingData,i,k, bias);
         positions_train{i,k}=xy_positions_train;
     end
end
firing_average = mean_firing_sum / (size(trainingData,1)*size(trainingData,2));
 
% Remove average from firing rate to make the distribution of firing rate zero-mean
for i=1:size(trainingData,1)
    for k=1:size(trainingData,2)
         response{i,k}=response{i,k} - firing_average;     
     end
end
 
%Get parameters of the Kalman filter for the 8 different angles
 A_ = cell(n_angles,1);
 W_ = cell(n_angles,1);
 H_ = cell(n_angles,1);
 Q_ = cell(n_angles,1);
 P_ = cell(n_angles,1);
 
 for i = 1:n_angles
     [A,W,H,Q] = kalman_training(response(:,i), positions_train(:,i));
     A_{i,1}=A; 
     W_{i,1}=W;
     H_{i,1}=H;
     Q_{i,1}=Q;
     P_{i,1}=zeros(2);
     
 end
 
 % Classifier training
 [training_Data, responseData] = transformData(trainingData);
 NN = trainClassifier(training_Data,responseData);
 
 % Save parameters
 modelParameters = struct ('A', A_, 'W', W_, 'H', H_, 'Q', Q_, 'P', P_, 'NN',NN);
 modelParameters(1).firing_average = firing_average;
end
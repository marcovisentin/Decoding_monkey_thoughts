% -- PREDICT --

function[decodedPosX,decodedPosY, newParameters]=positionEstimator(past_current_trial, modelParameters)

%Classification of the angle using pre-trained classificator once a new
%trajectory has to be calculated

% Pre-processing
if isempty(past_current_trial.decodedHandPos)== 1
training_Data = [];
t_max = 320;
for i = 1:98
   num_spikes = length(find(past_current_trial.spikes(i,1:t_max)==1));
   spikes_count(i)=num_spikes;
end
training_Data = cat(1, training_Data, spikes_count);

% Actual classification
NN = modelParameters(1).NN;
i_angle = NN.predict(training_Data);

% Save angle classified
modelParameters(1).angle = i_angle;
end

% Now predict trajectory
% Take A,W,H,Q associated to the angle identified. P come from the previous
% update during prediction

A = modelParameters(modelParameters(1).angle).A;
W = modelParameters(modelParameters(1).angle).W;
H = modelParameters(modelParameters(1).angle).H;
Q = modelParameters(modelParameters(1).angle).Q;
P = modelParameters(modelParameters(1).angle).P;

if isempty(past_current_trial.decodedHandPos)== 1 % first prediction, use the starting point
    decodedPosX = past_current_trial.startHandPos(1,1);
    decodedPosY = past_current_trial.startHandPos(2,1); 
    newParameters = modelParameters; 

else 
    decoded_position = past_current_trial.decodedHandPos(:,end); % all other, I use the previous position to calculate the successive
    Z = past_current_trial.spikes; % n_neuron x n_bins(actual), every time increase of one bin

    % Step 1: Time update
    X_m=A*decoded_position; 
    P_m=A*P*A'+W;
    %Step 2: Measurement update
    K=P_m*H'*pinv(H*P_m*H'+Q); %Kalman gain
    I = eye(2);
    P = (I-K*H)*P_m;
    decoded_position=X_m+K*((sqrt(sum(Z(:, 1:(end)),2) - sum(Z(:,(1:(end-20))),2))- modelParameters(1).firing_average)-H*X_m);
   
    decodedPosX = decoded_position(1,1);
    decodedPosY = decoded_position(2,1);
    modelParameters(modelParameters(1).angle).P = P ; % update P 
    newParameters = modelParameters; 
end
end

% -- KALMAN FILTER --

tic;
clc
clear
close all

% Load data
load monkeydata_training.mat

% Set random number generator
rng(2018);
ix = randperm(length(trial));
%addpath(teamName);

% Select training and testing data 
trainingData = trial(ix(1:80),:);
testData = trial(ix(80:end),:);
fprintf('Testing the continuous position estimator...')
meanSqError = 0;
n_predictions = 0;  
figure
hold on
axis square
grid

% Train Model
[modelParameters] = positionEstimatorTraining(trainingData);

% Estimate position
for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8)
        decodedHandPos = [];
        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;
            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); %Why one?
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            if t < size(testData(tr,direc).spikes,2)
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            end
        end
        n_predictions = n_predictions+length(times);
        hold on
        if tr == 6
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
        xlabel('X (cm)')
        ylabel('Y (cm)')
        end
    end
end
legend('Decoded Position', 'Actual Position')
RMSE = sqrt(meanSqError/n_predictions); 
disp(RMSE)
toc;

%% classifier
function [training_Data, responseData] = transformData(training_data)
%%Transform the training_data into a matrix and vector simulating predictors and response:
% Input:
% - training_data:
%     training_data(n,k)              (n = trial id,  k = reaching angle)
%     training_data(n,k).trialId      unique number of the trial
%     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
%     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
%  Output:
%      trainingData: size(trial,1)*size(trial,2),size(trial(1,1).spikes,1)
%      responseData: response vector of same row size as trainingData
training_Data = [];
responseData = [];
spikes_count = zeros(length(training_data),98);
t_max=340; %Only 320 ms 
for k = 1:8
    for i = 1:98
        for n = 1:length(training_data)           
                %num_spikes = length(find(training_data(n,k).spikes(i,1:t_max)==1));
                %spikes_count(n,i) = num_spikes;
                spikes_count(n,i)=sum(training_data(n,k).spikes(i,1:t_max));
        end
    end
    training_Data = cat(1, training_Data, spikes_count); 
    reaching_angle(1:length(training_data)) = k;
    responseData =cat(2,responseData, reaching_angle);
    
end
end
function [trainedClassifier] = trainClassifier(trainingData, responseData)
% Returns a trained classifier and its accuracy in percentage.
%  Input:
%      trainedData
%      responseData
%  Output:
%      trained_classifier: a struct containing the trained classifier. 
%
%      trained_classifier.predict: a function to make predictions on new
%       data.
%
%      validation_percent: a double containing the accuracy in percent.


%%Process the data: response and predictors:
name = 'cell';
predictorNames={98};
for i = 1 : 98
    predictorNames{i} = [name '_' num2str(i,'%d')]; % Cell Array
end
inputTable = array2table(trainingData, 'VariableNames',predictorNames);
predictors = inputTable(:, predictorNames);
response = responseData(:);
%isCategoricalPredictor = ones(1,98)*false;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationDiscriminant = fitcdiscr(...
    predictors, ...
    response, ...
    'DiscrimType', 'pseudolinear', ...
    'Gamma', 0.49, ...
    'Delta',0.002,...
    'FillCoeffs', 'off', ...
    'ClassNames', [1; 2; 3; 4; 5; 6; 7; 8]);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
trainedClassifier.predict = @(x) discriminantPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;
end

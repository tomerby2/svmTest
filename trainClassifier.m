function [trainedClassifier, validationAccuracy] = trainClassifier(datasetTable)
% Convert input to table
datasetTable = table(datasetTable');
datasetTable.Properties.VariableNames = {'row'};
% Split matrices in the input table into vectors
datasetTable.row_1 = datasetTable.row(:,1);
datasetTable.row_2 = datasetTable.row(:,2);
datasetTable.row_3 = datasetTable.row(:,3);
datasetTable.row = [];
% Extract predictors and response
predictorNames = {'row_1', 'row_2'};
predictors = datasetTable(:,predictorNames);
predictors = table2array(varfun(@double, predictors));
response = datasetTable.row_3;
% Train a classifier
trainedClassifier = fitcsvm(predictors, response, 'KernelFunction', 'linear', 'PolynomialOrder', [], 'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', 1, 'PredictorNames', {'row_1' 'row_2'}, 'ResponseName', 'row_3', 'ClassNames', [0 1]);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier, 'KFold', 5);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

%% Uncomment this section to compute validation predictions and scores:
% % Compute validation predictions and scores
% [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
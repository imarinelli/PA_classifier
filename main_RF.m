function [Class,testE, trainE, count_tot, count_rk]=main_RF(data, label, iter, k, nb_samples_per_class, nb_folds, Sel, nTrees)


count_tot = zeros(k,1); count_rk = zeros(iter,k);
testE = NaN.*ones(iter,1);
trainE = NaN.*ones(iter,1);


parfor i=1:iter
    compute(data, label, k, nb_samples_per_class, nb_folds, Sel, nTrees, i);
end


for i=1:iter
    
    load(sprintf('Output/Iteration_%d.mat',i))
    
    Class.(sprintf('Train_Idx%d',i)) = indices;
    Class.(sprintf('Test_Idx%d',i)) = testIdx ;
    Class.(sprintf('Train_Iter%d',i)) = TrainC;
    Class.(sprintf('Test_Iter%d',i)) = TestC;
    Class.(sprintf('trainError_Iter%d',i)) = trainError;
    Class.(sprintf('testError_Iter%d',i)) = testError;

    
    count_rk(i,:) = x1;
    count_tot = count_tot + Count;
    testE(i) = testError;
    trainE(i) = trainError;

end

fname = sprintf('Output/Iteration_*.mat');
delete(fname)
end


function compute(data, label, k, nb_samples_per_class, nb_folds, Sel, nTrees, i)

Count = zeros(k,1);    actSet = 1;
uu = find(label==2); data_Cu = data(uu,:); lab_Cu = label(uu,:);
indices_Cu = nFoldCrossValidation(data_Cu,'labels',lab_Cu,'splits','random','nb_samples',nb_samples_per_class,'nb_folds',nb_folds,'comparable',0);
idx_Cu = uu(indices_Cu{actSet});


uu = find(label==1); data_Con = data(uu,:); lab_Con = label(uu,:);
indices_Con = nFoldCrossValidation(data_Con,'labels',lab_Con,'splits','random','nb_samples',nb_samples_per_class,'nb_folds',nb_folds,'comparable',0);
idx_Con = uu(indices_Con{actSet});

indices = [idx_Cu; idx_Con];

% extract the training set
trainSet = data(indices,:);
trainLab = label(indices);

% extract the test set
testIdx = 1:length(label);
testIdx(indices) = [];
testIdx = testIdx(:);
testSet = data(testIdx,:);
testLab = label(testIdx);


%%%%%%%%%% Random Forest

B = TreeBagger(nTrees,trainSet,trainLab, 'Method', 'classification','OOBPredictorImportance','on');
imp = B.OOBPermutedPredictorCountRaiseMargin;

[u,x1] = sort(imp);
x = x1(end-Sel+1:end);
Count(x) = Count(x)+1;

trainSet = trainSet(:,x);  testSet = testSet(:,x);
B = TreeBagger(nTrees,trainSet,trainLab, 'Method', 'classification','OOBPredictorImportance','on');

% 1 = control; 2 = PA
gPosition1 = find(strcmp('1',B.ClassNames));
gPosition2 = find(strcmp('2',B.ClassNames));
[Yfit,Sfit] = oobPredict(B);
Y = B.Y; 


predChar_train = B.predict(trainSet);  % Predictions is a char though. We want it to be a number.
estimatedTrainLabels = str2double(predChar_train);

predChar_test = B.predict(testSet);
estimatedTestLabels = str2double(predChar_test);

trainError  = mean( trainLab ~= estimatedTrainLabels);
testError  = mean( testLab ~= estimatedTestLabels );


TrainC = [trainLab estimatedTrainLabels];
TestC = [testLab estimatedTestLabels];

save(sprintf('Output/Iteration_%d',i), 'indices', 'testIdx', 'TrainC', 'TestC', 'trainError', 'testError', 'x1', 'Count', 'Y', 'Sfit', 'gPosition1', 'gPosition2');

end
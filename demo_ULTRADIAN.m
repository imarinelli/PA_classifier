close all; clc;
clear all

iter = 1e3;

load('dBM_z_Set_IX.mat');  % file containing the dBMs

data = dBM_CT(:,4:end); 

label = dBM_CT(:,3);
data(label==4,:) = []; label(label==4) = [];
label(label==10)=1;
label(label==7)=2;

l = 1:7:size(data,2);
n = size(data,1);

ex = find(sum(isnan(data),2));
data(ex,:) = []; label(ex) = [];
data(data==0) = 1e-6;


nb_samples_per_class = 51;
k = size(data,2);


nb_folds = 1; Sel = 3;
nTrees=500;

data = zscoreTransformation(log(data));

[Class,testError, trainError, count_tot, count_rk] = main_RF(data, label, iter, k, nb_samples_per_class, nb_folds, Sel, nTrees);
count = 100.*count_tot./iter;

fname = strcat('Class_some.mat');
save(fname, 'Class', 'testError', 'trainError', 'count', 'count_rk', 'ROC_Y', 'ROC_Sfit', 'ROC_gPosition', 'n', 'iter');



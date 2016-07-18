clear all; clc;

if ~isdir('./build'),
    compile;
end
addpath('./build/'); 
               
% load data
if ~exist('./ToyAnomaly.mat'),
    genSyntheticAnomalyData;
end
load('./ToyAnomaly.mat');

dataset.seqs = D.seqs;
dataset.labels = cellfun(@(x) int32(unique(x)), D.labels);

% divide data into half
dataset.splits{1}.train = 1:200;
dataset.splits{1}.valid = 300:400;
dataset.splits{1}.test = 300:400;

%% Common parameter values
params.common.optimizer = 'nrbm';
params.common.nbHiddenStates = 4;
params.common.regFactorL2 = 0.1;
params.common.seed = 02139;
params.common.verbose = true;

%% Experiment with OCHCRF
% See Song et al., IJCAI 2013
params.ochcrf = params.common;
params.ochcrf.modelType = 'ochcrf';
params.ochcrf.rho = 0.5;
[bRc.ochcrf,Rc.ochcrf] = experimentOCHCRF(dataset,params.ochcrf);


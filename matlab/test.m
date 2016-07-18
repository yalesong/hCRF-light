clear all; clc;

if ~isdir('./build'),
    compile;
end
addpath('./build/'); 

% load data
dataset=load('../data/natops/NATOPS6.mat');
dataset.labels = cellfun(@(x) int32(unique(x)), dataset.labels);

% divide data into half
dataset.splits
for fold=1:5,
    dataset.splits{fold}.train = dataset.splits{fold}.train(1:5:end);
end
dataset.splits(2:5)=[];

%% Common parameter values
params.common.optimizer = 'lbfgs';
params.common.nbHiddenStates = 4;
params.common.regFactorL2 = 10;
params.common.seed = 02139;
params.common.verbose = false;

%% Experiment with HCRF
params.hcrf = params.common;
params.hcrf.modelType = 'hcrf';
[bRc.hcrf,Rc.hcrf] = experimentHCRF(dataset,params.hcrf);

%% Experiment with HCNF
params.hcnf = params.hcrf;
params.hcnf.nbGates = 12;
[bRc.hcnf,Rc.hcnf] = experimentHCNF(dataset,params.hcnf);
 
%% Experiment with OCHCRF
% See Song et al., IJCAI 2013
dataset_oneclass = loadOneClassToyData('../data/toy-oneclass/');
params.ochcrf = params.common;
params.ochcrf.modelType = 'ochcrf';
params.ochcrf.regFactorL2 = 0.01;
params.ochcrf.rho = 0.5;
[bRc.ochcrf,Rc.ochcrf] = experimentOCHCRF(dataset_oneclass,params.ochcrf);

%% Experiment with HSS-HCRF
% See Song et al., CVPR 2013
params.hsshcrf = params.common;
params.hsshcrf.modelType = 'hsshcrf';
params.hsshcrf.nbHiddenStates = 4;
params.hsshcrf.nbGates = 12;
params.hsshcrf.segmentTau = .5; 
params.hsshcrf.maxFeatureLayer = 3;
[bRc.hsshcrf,Rc.hsshcrf] = experimentHSSHCRF(dataset,params.hsshcrf); 

%% Experiment with Multi-view HCRF (Linked topology)
% See Song et al., CVPR 2012
% This part may take longer
params.lhcrf = params.common;
params.lhcrf.modelType = 'mvhcrf';
params.lhcrf.graphType = 'linked'; 
params.lhcrf.inferenceEngine = 'JT'; 
params.lhcrf.nbViews = 2;
params.lhcrf.nbHiddenStates = {[4 4]}; 
params.lhcrf.rawFeatureIndex = {[0:3,8:13], [4:7,14:19]}; % idx of L and R arms
[bRc.lhcrf,Rc.lhcrf] = experimentLHCRF(dataset,params.lhcrf); 

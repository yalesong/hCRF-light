% Tested on Windows 7 64bit. Visual Studio 8.0
addpath('../bin/openMP/'); 

% load data
load('../toydata/ArmGesture.mat');
dataset.labels = cellfun(@(x) int32(unique(x)), dataset.labels);

% For demonstration purpose, use one fifth of the training split.
for fold=1:5,
    dataset.splits{fold}.train = dataset.splits{fold}.train(1:5:end);
end
dataset.splits(2:5)=[];

%% Common parameter values
params.common.optimizer = 'lbfgs';
params.common.nbHiddenStates = 4:4:12;
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
 
%% Experiment with HSS-HCRF
params.hsshcrf = params.common;
params.hsshcrf.modelType = 'hsshcrf';
params.hsshcrf.nbHiddenStates = 8;
params.hsshcrf.nbGates = 12;
params.hsshcrf.segmentTau = .5; 
params.hsshcrf.maxFeatureLayer = 4;
[bRc.hsshcrf,Rc.hsshcrf] = experimentHSSHCRF(dataset,params.hsshcrf); 


%% Experiment with Multi-view HCRF (Linked topology)
% This part may take longer
params.lhcrf = params.common;
params.lhcrf.modelType = 'mvhcrf';
params.lhcrf.graphType = 'linked'; 
params.lhcrf.inferenceEngine = 'JT'; 
params.lhcrf.nbViews = 2;
params.lhcrf.nbHiddenStates = {[6 6]}; 
params.lhcrf.rawFeatureIndex = {[0:3,8:13], [4:7,14:19]}; % idx of L and R arms
[bRc.lhcrf,Rc.lhcrf] = experimentLHCRF(dataset,params.lhcrf); 

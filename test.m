clear all; clc;

if ~isdir('./build'),
    compile;
end
addpath('./build/'); 

% load data
if ~exist('../data/natops/NATOPS6.mat'),
    system('../data/natops/download_natops6.sh');
end
dataset=load('../data/natops/NATOPS6.mat');

% Labels are required to be of type int32
dataset.labels = cellfun(@(x) int32(unique(x)), dataset.labels);

% Split the dataset. The NATOPS6 dataset contains 6 gestures,
% performed by 20 people, each person repeating each gesture
% 20 times. The data is organized by subject-gesture, so the
% first 20 samples correspond to the first gesture performed
% by the first subject, and the first 120 samples correspond
% to all six gestures performed by the first subject. For 
% simplicity, we split the data into train (first 6 subjects), 
% validation (next 3 subjects) and test (last 3 subjects).
dataset.splits{1}.train = 1:2:1200;
dataset.splits{1}.valid = 1201:1800;
dataset.splits{1}.test = 1801:2400;

%% Common parameter values
params.common.optimizer = 'lbfgs';
params.common.nbHiddenStates = 8;
params.common.regFactorL2 = 1;
params.common.seed = 02139;
params.common.verbose = false;

%% Experiment with HCRF
fprintf('Training Hidden Conditional Random Fields (HCRF)\n');
params.hcrf = params.common;
params.hcrf.modelType = 'hcrf';
[bRc.hcrf,Rc.hcrf] = experimentHCRF(dataset,params.hcrf);

%% Experiment with HCNF
fprintf('Training Hidden Conditional Neural Fields (HCNF)\n');
params.hcnf = params.hcrf;
params.hcnf.nbGates = 12;
[bRc.hcnf,Rc.hcnf] = experimentHCNF(dataset,params.hcnf);
 
%% Experiment with HSS-HCRF
% See Song et al., CVPR 2013
fprintf('Training Hierarchical Sequence Summarization HCNF (HSS-HCNF)\n');
params.hsshcrf = params.common;
params.hsshcrf.modelType = 'hsshcrf';
params.hsshcrf.nbHiddenStates = 8;
params.hsshcrf.nbGates = 12;
params.hsshcrf.segmentTau = .5; 
params.hsshcrf.maxFeatureLayer = 4;
[bRc.hsshcrf,Rc.hsshcrf] = experimentHSSHCRF(dataset,params.hsshcrf); 

%% Experiment with Multi-view HCRF (Linked topology)
% See Song et al., CVPR 2012
fprintf('Training Multi-view Hidden Conditional Random Fields (MV-HCRF)\n');
params.lhcrf = params.common;
params.lhcrf.modelType = 'mvhcrf';
params.lhcrf.graphType = 'linked'; 
params.lhcrf.inferenceEngine = 'LBP'; % use JT for exact inference
params.lhcrf.nbViews = 2;
params.lhcrf.nbHiddenStates = {[8 8]}; 
params.lhcrf.rawFeatureIndex = {[0:3,8:13], [4:7,14:19]}; % idx of L and R arms
[bRc.lhcrf,Rc.lhcrf] = experimentLHCRF(dataset,params.lhcrf); 

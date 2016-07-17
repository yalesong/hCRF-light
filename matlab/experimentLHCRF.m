function [ bRc, Rc ] = experimentLHCRF( dataset, params, H )

if ~exist('H','var')
    H = params.nbHiddenStates;
end

seqs = dataset.seqs;
labels = dataset.labels;
splits = dataset.splits;

bRc = cell(1,numel(splits)); % best results
Rc = cell(1,numel(splits)); % all results
for fold=1:numel(splits)
    best_accuracy = -1;
    best_idx = -1;
    R = cell(1,numel(H));
    for i=1:numel(H)
        params.nbHiddenStates = H{i}; 

        % Create toolbox
        matHCRF('createToolbox', params.modelType, params.graphType, ...
            params.nbViews, params.nbHiddenStates, params.rawFeatureIndex);
        matHCRF('setOptimizer',params.optimizer);
        matHCRF('setInferenceEngine',params.inferenceEngine);
        matHCRF('setParam','regularizationL2',params.regFactorL2); 
        matHCRF('setParam','randomSeed', params.seed); 

        % Train
        matHCRF('setData',seqs(splits{fold}.train),[],labels((splits{fold}.train)));
        id=tic(); matHCRF('train'); time=toc(id); 
        [model,features] = matHCRF('getModel');    
        matHCRF('clearToolbox');

        % Load model 
        matHCRF('createToolbox', params.modelType, params.graphType, ...
            params.nbViews, params.nbHiddenStates, params.rawFeatureIndex);
        matHCRF('setOptimizer',params.optimizer);
        matHCRF('setInferenceEngine',params.inferenceEngine);
        matHCRF('initToolbox');
        matHCRF('setModel',model,features);

        % Test on validatation split
        pYstar = cell(1,numel(splits{fold}.valid));
        for sample=1:numel(splits{fold}.valid), 
            matHCRF('setData',seqs(splits{fold}.valid(sample)),[],labels(splits{fold}.valid(sample)));
            matHCRF('test');
            ll = matHCRF('getResults');
            pYstar{sample} = ll{1};
        end
        matHCRF('clearToolbox');

        [~,Ystar] = max(cell2mat(pYstar)); Ystar = Ystar-1;
        accuracy = sum(Ystar==labels(splits{fold}.valid))/numel(splits{fold}.valid);
        if accuracy >= best_accuracy
            best_idx = i;
            best_accuracy = accuracy;
        end 

        if params.verbose,
            fprintf('[fold %d] H=[%d %d], accuracy = %f, time = %.2f mins\n', ...
                fold, params.nbHiddenStates(1), params.nbHiddenStates(2), accuracy, time/60);
        end
        
        R{i}.model = model;
        R{i}.features = features;
        R{i}.params = params;
        R{i}.accuracy_valid = accuracy;
        R{i}.time = time;
    end
    Rc{fold} = R;
    bRc{fold} = R{best_idx};
end

for fold=1:numel(splits)
    bR = bRc{fold};

    % Load the best model
     matHCRF('createToolbox', bR.params.modelType, ...
         bR.params.graphType, bR.params.nbViews, ...
         bR.params.nbHiddenStates, bR.params.rawFeatureIndex);
    matHCRF('setOptimizer',bR.params.optimizer);
    matHCRF('setInferenceEngine',bR.params.inferenceEngine);
    matHCRF('initToolbox');
    matHCRF('setModel',bR.model,bR.features);

     % Test on test split
    pYstar = cell(1,numel(splits{fold}.test));
    for j=1:numel(splits{fold}.test), 
        matHCRF('setData',seqs(splits{fold}.test(j)),[],labels(splits{fold}.test(j)));
        matHCRF('test');
        ll = matHCRF('getResults');
        pYstar{j} = ll{1};
    end
    matHCRF('clearToolbox');

    [~,Ystar] = max(cell2mat(pYstar)); Ystar = Ystar-1;
    accuracy = sum(Ystar==labels(splits{fold}.test))/numel(splits{fold}.test);

    fprintf('Best> Linked-HCRF [%d %d] acc_test = %f, time = %.2f mins\n', ...
        bR.params.nbHiddenStates(1), bR.params.nbHiddenStates(2),...
        accuracy, bR.time/60);
    bRc{fold}.accuracy_test = accuracy;
end


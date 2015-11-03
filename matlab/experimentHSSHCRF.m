function [ bRc, Rc ] = experimentHSSHCRF( dataset, params, G,tau )

if ~exist('H','var')
    H = params.nbHiddenStates;
end
if ~exist('G','var')
    G = params.nbGates;
end
if ~exist('tau','var')
    tau = params.segmentTau;
end
if ~exist('sigma','var')
    sigma = params.regFactorL2;
end

seqs = dataset.seqs;
labels = dataset.labels;
splits = dataset.splits;

Rc = cell(1,numel(splits)); % all results
bRc = cell(1,numel(splits)); % best results

for fold=1:numel(splits)
    best_idx = -1;
    best_accuracy = -1;
    R = cell(numel(H),numel(G),numel(tau),numel(sigma));   
    
    for i=1:numel(H)          
        for j=1:numel(G)
            for k=1:numel(tau)
                for l=1:numel(sigma)
                    params.nbHiddenStates = H(i);
                    params.nbGates = G(j);
                    params.segmentTau = tau(k);
                    params.regFactorL2 = sigma(l);

                    % Create model
                    matHCRF('createToolbox',params.modelType,params.nbHiddenStates);
                    matHCRF('setOptimizer',params.optimizer);
                    matHCRF('set','regularizationL2',params.regFactorL2);  
                    matHCRF('set','randomSeed',params.seed); 
                    matHCRF('set','nbGates',params.nbGates);
                    matHCRF('set','maxFeatureLayer',params.maxFeatureLayer);
                    matHCRF('set','segmentTau',params.segmentTau);

                    % Train
                    matHCRF('setData',seqs(splits{fold}.train),[],labels((splits{fold}.train)));
                    id=tic(); matHCRF('train'); time=toc(id); 
                    [model,features] = matHCRF('getModel');    
                    matHCRF('clearToolbox');

                    % Load model
                    matHCRF('createToolbox',params.modelType,params.nbHiddenStates);
                    matHCRF('setOptimizer',params.optimizer);
                    matHCRF('set','nbGates',params.nbGates); % nbGates needs to be set before initToolbox
                    matHCRF('set','maxFeatureLayer',params.maxFeatureLayer);
                    matHCRF('set','segmentTau',params.segmentTau);
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
                    if accuracy > best_accuracy
                        best_idx = [i j k l];
                        best_accuracy = accuracy;
                    end 

                    if params.verbose,
                        fprintf('[fold %d] H=%d, G=%d, tau = %.1f, sigma=%.2f, acc_valid = %f, time = %.2f mins\n', ...
                            fold, params.nbHiddenStates, params.nbGates, params.segmentTau, params.regFactorL2, accuracy, time/60);
                    end

                    R{i,j,k,l}.model = model;
                    R{i,j,k,l}.features = features;
                    R{i,j,k,l}.params = params;
                    R{i,j,k,l}.accuracy_valid = accuracy;
                    R{i,j,k,l}.time = time;            
                end
            end
        end
    end
    Rc{fold} = R;
    bRc{fold} = R{best_idx(1),best_idx(2),best_idx(3),best_idx(4)};
end

for fold=1:numel(splits)
    bR = bRc{fold};
    % Load the best model
    matHCRF('createToolbox',bR.params.modelType,bR.params.nbHiddenStates);
    matHCRF('setOptimizer',bR.params.optimizer);
    matHCRF('set','nbGates',bR.params.nbGates); % nbGates needs to be set before initToolbox
    matHCRF('set','maxFeatureLayer',bR.params.maxFeatureLayer);
    matHCRF('set','segmentTau',bR.params.segmentTau);
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

    fprintf('Best> [fold %d] HSS-HCRF%d H=%d, G=%d, tau=%.1f, acc_test = %f, time = %.2f mins \n', ...
        fold, bR.params.maxFeatureLayer, bR.params.nbHiddenStates, bR.params.nbGates, bR.params.segmentTau, accuracy, bR.time/60);
    bRc{fold}.accuracy_test = accuracy;
end

end
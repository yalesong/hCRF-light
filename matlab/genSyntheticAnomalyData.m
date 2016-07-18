clear all;
rng default;
seqs={}; labels={};

SNR = 10; % in DB
nb_pos = 300; 
nb_neg = 100;
len = 10;

% Positive (normal) class samples
for i=1:nb_pos
    x = [awgn(1:len,SNR,0); 0.5*len*awgn(ones(1,len),2*SNR,0)];
    seqs{end+1} = x;
    labels{end+1} = zeros(1,size(x,2));
end

% Negative (abnormal) class samples
for i=1:nb_neg
    x = [0.5*len*awgn(ones(1,len),2*SNR,0); awgn(1:len,SNR,0)];
    seqs{end+1} = x;
    labels{end+1} = ones(1,size(x,2));
end

D.seqs = seqs;
D.labels = labels;
D.trainSplitParams = {{1:nb_pos-nb_neg}};
D.validateSplitParams = {{1:nb_pos-nb_neg}};
D.testSplitParams = {nb_pos-nb_neg+1:nb_pos+nb_neg};


%% Save data
save('./ToyAnomaly.mat', 'D');  

fid_x = fopen('../data/toy/dataAnomalyTrain.csv','w+');
fid_y = fopen('../data/toy/seqLabelsAnomalyTrain.csv','w+');
for idx=D.trainSplitParams{1}{1},
    y = unique(D.labels{idx});
    fprintf(fid_y,'%d\n',y);

    x = D.seqs{idx};
    fprintf(fid_x,'1,%d\n', numel(x));
    for i=1:numel(x),
        fprintf(fid_x,'%f',x(i));
        if i==numel(x),
            fprintf(fid_x,'\n');
        else
            fprintf(fid_x,',');
        end
    end
end
fclose(fid_x);
fclose(fid_y);

fid_x = fopen('../data/toy/dataAnomalyTest.csv','w+');
fid_y = fopen('../data/toy/seqLabelsAnomalyTest.csv','w+');
for idx=D.testSplitParams{1},
    y = unique(D.labels{idx});    
    fprintf(fid_y,'%d\n',y);
    
    x = D.seqs{idx};
    fprintf(fid_x,'1,%d\n', numel(x));
    for i=1:numel(x),
        fprintf(fid_x,'%f',x(i));
        if i==numel(x),
            fprintf(fid_x,'\n');
        else
            fprintf(fid_x,',');
        end
    end
end
fclose(fid_x);
fclose(fid_y);


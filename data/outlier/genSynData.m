clear all;
rng default;
seqs={}; labels={};

SNR = 10; % in DB
nb_pos = 300; 
nb_neg = 100;
max_amp = 1; 
len = 10;
freq = 10000;
t = [0:1/(len*freq):1]; 

% Positive (normal) class samples
for i=1:nb_pos
    amp = randi(max_amp);
    x = awgn(amp*sin(2*pi*freq*t(1:len)),SNR,0);
    seqs{end+1} = x;
    labels{end+1} = zeros(1,size(x,2));
end

% Negative (abnormal) class samples
for i=1:nb_neg
    amp = max_amp + randi(max_amp);
    x = awgn(amp*sin(2*pi*freq*t(1:len)),SNR,0); 
    seqs{end+1} = x;
    labels{end+1} = ones(1,size(x,2));
end

D.seqs = seqs;
D.labels = labels;
D.trainSplitParams = {{1:nb_pos-nb_neg}};
D.validateSplitParams = {{1:nb_pos-nb_neg}};
D.testSplitParams = {nb_pos-nb_neg+1:nb_pos+nb_neg};
save('./synData.mat', 'D');  


%% Visualize
my_axis = [1 max(cellfun(@(x) size(x,2), D.seqs)) ...
             min(cellfun(@(x) min(x), D.seqs)) ...
             max(cellfun(@(x) max(x), D.seqs))];

subplot(1,2,1); hold on;
for i=1:numel(D.trainSplitParams{1}{1})
    j =  D.trainSplitParams{1}{1}(i); 
    if D.labels{j}(1)==0, color='b'; else color='r'; end;
    plot(D.seqs{j}', color); 
end 
axis(my_axis); title('Train set'); xlabel('time'); ylabel('value'); hold off;

subplot(1,2,2); hold on;
for i=1:numel(D.testSplitParams{1})
    j =  D.testSplitParams{1}(i); 
    if D.labels{j}(1)==0, color='b'; else color='r'; end;
    plot(D.seqs{j}', color); 
end  
axis(my_axis); title('Test set'); xlabel('time'); ylabel('value'); hold off;
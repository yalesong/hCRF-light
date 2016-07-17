classdef hcrfToolbox < handle
    %HCRFTOOLBOX a wrapper around matHCRF to be used with matlab

    properties(Access = private)
        % This data should never be accessed by the client code. It contains
        % pointer to data allocated by the underlaying c++ code.
        data = uint64([0, 0]);
    end
   
    methods
        function obj = hcrfToolbox(model, optimiser, nbr_hiddenstate, window_size )
            % Construct a toolbox of the given type. The following parameter can
            % be given
            % model : The type of model ('crf', 'hcrf', 'ldcrf'). Default: crf
            % optimiser: The optimiser for selecting the weight ('cg', 'bfgs',
            % 'asa', 'owlqn' or 'lbfgs'). Default: cg
            % Number of hidden states (default 3)
            % Window size (number of observation to take into account before and
            % after the current time). Default 0
            
            % Set the default value
            if nargin<1
                model = 'crf';
            end
            if nargin<2
                optimiser='cg';
            end
            if nargin<3
                nbr_hiddenstate = 3;
            end
            if nargin<4
                window_size=0;
            end
            % Call the mex file. No call with a string  as the first parameter
            % should have been made into the mex. 
            obj.data = matHCRF(obj.data, 'createToolbox', model, optimiser, nbr_hiddenstate, window_size);
        end
        
        function delete(obj)
            % Called when the object is destroyed. We release the memory
            % allocated by the mex file
            obj.unloadData()
            obj.clearToolbox()          
        end
        
        function clearToolbox(obj)
            % Release the memory used by the toolbox
            obj.data = matHCRF(obj.data, 'clearToolbox');
        end
  
        function set(obj, param, value)
            % This function can be used to change a pameter of the toolbox. 
            obj.data = matHCRF(obj.data, 'set', param, value);
        end
        
        function value = get(obj, param)
            % This function can be used to get a parameter from the toolbox
            [obj.data value] =  matHCRF(obj.data, 'get', param);
        end
        
        function setData(obj, features, labels)
            % This function can be used to load data into the module. The data
            % are copied into memory allocated. So after loading the data can be
            % cleared from the matlab space if memory is an issue. A best way
            % would be to use loadData to load directly from file without a copy
            % in Matlab.
            % Features is a cell of sequence. Every sequence is a matrix of
            % double with one column per observation.
            % Label is a cell containing one vector per observation. The vector
            % contains the label for each observation. This is converted to
            % int32. Label must be zero based
            if length(features) ~= length(labels)
                error('Labels and feature should have the same length')
            end
            if (cellfun(@length, features) ~= cellfun(@length, labels))
                error('Labels and features must ahve the same length for all observation')
            end
            converted_labels = cellfun(@int32, labels,'UniformOutput', false);
            if ~isequal(converted_labels, labels)
                error('Labels seems to be not integers');
            end
            obj.data = matHCRF(obj.data, 'setData', features, converted_labels);
        end
        
        function loadData(obj, featuresFile, labelFile)
            % Load data directly from csv file. A file for features and one for
            % label ( See the C++ documention for the format)
            obj.data = matHCRF(obj.data, 'loadData', featuresFile, labelFile);
        end
        
        function unloadData(obj)
            % This function free the space use by the data
            obj.data = matHCRF(obj.data, 'unloadData');
        end
        
        function train(obj)
            % Train the model
            obj.data = matHCRF(obj.data, 'train');
        end
        
        function test(obj)
            % Test the model.
            obj.data = matHCRF(obj.data, 'test');
        end
        
        function varargout = getAllFeatures(obj)
           temp = cell(nargout,1);
            [obj.data temp{1:end}] = matHCRF(obj.data, 'getAllFeatures');
            for i=1:length(temp)
                varargout(i) = temp(i); %#ok<AGROW>
            end
        end
        
        function varargout = getResults(obj)
            % This return the result.
            temp = cell(nargout,1);
            [obj.data temp{1:end}] = matHCRF(obj.data, 'getResults');
            for i=1:length(temp)
                varargout(i) = temp(i); %#ok<AGROW>
            end
        end
        
        function loadModel(obj, modelFile,  featureFile)
            % This function load the model from a text file (and the features)
            obj.data = matHCRF(obj.data, 'loadModel', modelFile, featureFile);
        end
        
        function saveModel(obj, modelFile,  featureFile)
            % This function save the model and the features function to two text
            % files. 
            obj.data = matHCRF(obj.data, 'saveModel', modelFile, featureFile);
        end

        function setModel(obj, model, feature)
            % This function can be used to set the model. From a matlab object
             obj.data = matHCRF(obj.data, 'setModel', model, feature);
        end
        
        function varargout = getModel(obj)
            % This function return up to two object describing the model and the
            % features function used.
            temp = cell(nargout,1);
            [obj.data temp{1:end}] = matHCRF(obj.data, 'getModel');
            for i=1:length(temp)
                varargout(i) = temp(i); %#ok<AGROW>
            end
        end
    end
end


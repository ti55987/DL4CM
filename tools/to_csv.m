updatedTrial = TrialData;
updatedTrial = renamevars( ...
    updatedTrial, ...
    {'id', 'trial', 'selected_box', 'correct_box', 'correct', 'reward'}, ...
    {'agentid','trials', 'actions', 'correct_actions', 'iscorrectaction', 'rewards'});

updatedTrial = removevars(updatedTrial, {'rt', 'imp_score', 'pos_urg', 'neg_urg', 'sens', 'lack_pers','lack_prem', 'lack_persev', 'lack_premed', 'imp_2', 'imp_3', 'imp_4', 'imp_5'});
%disp(updatedTrial.Properties.VariableNames);
% save trial data
writetable(updatedTrial, '/Users/ccnlab/Development/dl4rl/data/prl_zou2022/trial_data.csv');
% --------
% Save real params
% Define custom column names
columnNames = param_names(8);
% Add column names to the matrix
dataStruct.matrix = RL2apc.real_params;
dataStruct.columnNames = columnNames{1};
% Save the matrix with column names to a CSV file
%saveStructToCSV('/Users/ccnlab/Development/dl4rl/data/prl_zou2022/output.csv', dataStruct);

function saveStructToCSV(filename, dataStruct)
    % Save a structure containing a matrix and column names to a CSV file
    
    % Open the file for writing
    fileID = fopen(filename, 'w');

    % Write the column names to the file
    fprintf(fileID, '%s,', dataStruct.columnNames{1:end-1});
    fprintf(fileID, '%s\n', dataStruct.columnNames{end});

    % Write the matrix data to the file
    for i = 1:size(dataStruct.matrix, 1)
        fprintf(fileID, '%f,', dataStruct.matrix(i, 1:end-1));
        fprintf(fileID, '%f\n', dataStruct.matrix(i, end));
    end

    % Close the file
    fclose(fileID);
end

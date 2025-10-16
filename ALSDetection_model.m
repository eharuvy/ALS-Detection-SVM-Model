%% Section 1 — Set working directory within submission folder
cd("data/")

%% Section 2 — Define file lists
als_files = {'A01.csv','A02.csv','A03.csv','A04.csv','A05.csv', ...
             'A06.csv','A07.csv','A08.csv','A09.csv','A10.csv','A11.csv'};
healthy_files = {'N01.csv','N02.csv','N03.csv','N04.csv','N05.csv', ...
                 'N06.csv','N07.csv','N08.csv','N09.csv','N10.csv','N11.csv'};

%% Section 3 — Load CSVs and add labels
all_data = table();  % empty table

% ALS data
for i = 1:numel(als_files)
    df = readtable(als_files{i}, 'ReadVariableNames', false);
    df.Label = ones(height(df),1); % ALS = 1
    all_data = [all_data; df];
end

% Healthy data
for i = 1:numel(healthy_files)
    df = readtable(healthy_files{i}, 'ReadVariableNames', false);
    df.Label = zeros(height(df),1); % Healthy = 0
    all_data = [all_data; df];
end

% Convert Label to categorical
all_data.Label = categorical(all_data.Label);  % '0' = Healthy, '1' = ALS

%% Section 4 — Half-Half Cross-Validation Setup
rng(2025);  % reproducibility

als_index = find(all_data.Label == categorical(1));
healthy_index = find(all_data.Label == categorical(0));

% Shuffle indices
als_index = als_index(randperm(numel(als_index)));
healthy_index = healthy_index(randperm(numel(healthy_index)));

% Split each class in half
half_als = floor(numel(als_index)/2);
half_healthy = floor(numel(healthy_index)/2);

als_half1 = als_index(1:half_als);
als_half2 = als_index(half_als+1:end);
healthy_half1 = healthy_index(1:half_healthy);
healthy_half2 = healthy_index(half_healthy+1:end);

%% Section 5 — Half-half SVM Cross-Validation
BoxC = 80;  % SVM cost parameter
metrics = zeros(2,3);  % [accuracy, sensitivity, specificity]

% Initialize summed confusion matrix
total_cm = zeros(2,2);

for iter = 1:2
    if iter == 1
        train_idx = [als_half1; healthy_half1];
        test_idx  = [als_half2; healthy_half2];
    else
        train_idx = [als_half2; healthy_half2];
        test_idx  = [als_half1; healthy_half1];
    end
    
    train_data = all_data(train_idx,:);
    test_data  = all_data(test_idx,:);
    
    % Compute class weights to balance ALS vs Healthy
    numALS = sum(train_data.Label == categorical(1));
    numHealthy = sum(train_data.Label == categorical(0));
    w = ones(height(train_data),1);
    w(train_data.Label==categorical(1)) = numHealthy/numALS;
    
    % Train SVM
    svm_model = fitcsvm(train_data(:,1:end-1), train_data.Label, ...
                        'KernelFunction','rbf', ...
                        'BoxConstraint', BoxC, ...
                        'Standardize', true, ...
                        'KernelScale','auto', ...
                        'ClassNames', categorical({'0','1'}), ...
                        'Weights', w);
    
    % Predict
    pred = predict(svm_model, test_data(:,1:end-1));
    
    % Confusion matrix
    cm = confusionmat(test_data.Label, pred, 'Order', categorical({'0','1'}));
    
    % Add to total confusion matrix
    total_cm = total_cm + cm;
    
    % Convert to table with labeled axes: Actual rows, Predicted columns
    predicted_labels = {'Pred_Healthy','Pred_ALS'};   % columns = predicted
    actual_labels    = {'Actual_Healthy','Actual_ALS'}; % rows = actual
    cm_table = array2table(cm, 'VariableNames', predicted_labels, 'RowNames', actual_labels);
    
    disp(['Iteration ', num2str(iter), ' — Confusion Matrix (Actual rows, Predicted columns):']);
    disp(cm_table);
    
    % Compute metrics
    tp = cm(2,2); tn = cm(1,1); fp = cm(1,2); fn = cm(2,1);
    accuracy = (tp+tn)/sum(cm,'all');
    sensitivity = tp/(tp+fn);
    specificity = tn/(tn+fp);
    
    metrics(iter,:) = [accuracy, sensitivity, specificity];
end
%% Section 6 — Summed Confusion Matrix
predicted_labels = {'Pred_Healthy','Pred_ALS'};   % columns = predicted
actual_labels    = {'Actual_Healthy','Actual_ALS'}; % rows = actual
total_cm_table = array2table(total_cm, 'VariableNames', predicted_labels, 'RowNames', actual_labels);

disp('Summed Confusion Matrix Across Both Cross-Validation Iterations:');
disp(total_cm_table);

%% Section 7 — Average metrics
avg_accuracy = mean(metrics(:,1));
avg_sensitivity = mean(metrics(:,2));
avg_specificity = mean(metrics(:,3));

fprintf('Average Accuracy: %.2f%%\n', avg_accuracy*100);
fprintf('Average Sensitivity: %.2f%%\n', avg_sensitivity*100);
fprintf('Average Specificity: %.2f%%\n', avg_specificity*100);
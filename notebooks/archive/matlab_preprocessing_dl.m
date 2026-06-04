%% Lateral Eye Movement Detection - MATLAB Preprocessing & DL Model
% Goal: 
% 1. Load without_eog_channels.set in EEGLAB
% 2. Apply same preprocessing as Python (filter 0.5-40Hz, Extended Infomax ICA)
% 3. Train a simple DL model to detect lateral eye movements

%% 1. Setup and Load Data
clear; clc;

% Add EEGLAB to path (adjust path to your EEGLAB installation)
% addpath('C:\path\to\eeglab');  % Uncomment and set your EEGLAB path
% eeglab nogui;  % Start EEGLAB in no GUI mode

% Load the dataset
EEG = pop_loadset('without_eog_channels.set');

fprintf('Dataset loaded:\n');
fprintf('  Channels: %d\n', EEG.nbchan);
fprintf('  Epochs: %d\n', EEG.trials);
fprintf('  Points per epoch: %d\n', EEG.pnts);
fprintf('  Sampling rate: %d Hz\n', EEG.srate);

%% 2. Extract Event Labels
% Get all events and show lateral eye movement labels
fprintf('\nEvent types in dataset:\n');
event_types = unique({EEG.event.type});
for i = 1:length(event_types)
    count = sum(strcmp({EEG.event.type}, event_types{i}));
    fprintf('  %s: %d events\n', event_types{i}, count);
end

% Find events in epoch 4
epoch4_events = EEG.event([EEG.event.epoch] == 4);
fprintf('\nEpoch 4 events:\n');
for i = 1:length(epoch4_events)
    fprintf('  %s at latency %.1f (%.3f sec)\n', ...
        epoch4_events(i).type, ...
        epoch4_events(i).latency, ...
        epoch4_events(i).latency / EEG.srate);
end

%% 3. Apply FIR Filter (0.5 - 40 Hz)
fprintf('\nApplying FIR filter (0.5-40 Hz)...\n');
EEG_filt = pop_eegfiltnew(EEG, 0.5, 40);
fprintf('✓ Filtering complete\n');

%% 4. Run Extended Infomax ICA
fprintf('\nRunning Extended Infomax ICA...\n');
% Extended Infomax with parameters matching Python
EEG_ica = pop_runica(EEG_filt, 'icatype', 'runica', ...
    'extended', 1, ...
    'lrate', 1e-5, ...
    'maxsteps', 2000);

fprintf('✓ ICA complete\n');
fprintf('  Number of ICs: %d\n', size(EEG_ica.icaweights, 1));

%% 5. Extract IC activations for epoch 4
epoch_idx = 4;  % 1-based indexing in MATLAB
ic_activations = EEG_ica.icaact(:, :, epoch_idx);  % (ICs × samples)

fprintf('\nIC activations for epoch 4:\n');
fprintf('  Shape: %d ICs × %d samples\n', size(ic_activations));

% Display IC values at 1 second (sample 200 @ 200Hz)
sample_idx = 200;
fprintf('  IC values at t=1s (sample %d):\n', sample_idx);
for ic = 1:min(5, size(ic_activations, 1))
    fprintf('    IC%d: %.6f\n', ic, ic_activations(ic, sample_idx));
end

%% 6. Prepare Training Data for DL Model
% Create training labels: 1 for lateral movement (eye-l, eye-r), 0 for others

% Initialize labels for all samples in all epochs
num_epochs = EEG_ica.trials;
num_samples = EEG_ica.pnts;
num_ics = size(EEG_ica.icaweights, 1);

% Extract all IC activations (ICs × samples × epochs)
all_ic_activations = EEG_ica.icaact;

% Create labels array
labels = zeros(num_epochs, num_samples);

% Mark samples with lateral eye movements
for i = 1:length(EEG.event)
    if strcmp(EEG.event(i).type, 'eye-l') || strcmp(EEG.event(i).type, 'eye-r')
        ep = EEG.event(i).epoch;
        % Convert global latency to epoch-relative sample
        epoch_start_sample = (ep - 1) * num_samples;
        sample_in_epoch = round(EEG.event(i).latency - epoch_start_sample);
        
        % Mark the event duration as lateral movement
        if sample_in_epoch > 0 && sample_in_epoch <= num_samples
            duration_samples = round(EEG.event(i).duration);
            end_idx = min(sample_in_epoch + duration_samples, num_samples);
            labels(ep, sample_in_epoch:end_idx) = 1;
        end
    end
end

fprintf('\nTraining data prepared:\n');
fprintf('  Total samples: %d\n', num_epochs * num_samples);
fprintf('  Lateral movement samples: %d (%.2f%%)\n', ...
    sum(labels(:)), 100 * sum(labels(:)) / numel(labels));

%% 7. Simple LSTM Model for Lateral Eye Movement Detection
% Reshape data: (samples × features × time)
% For simplicity, use first 5 ICs as features

num_ics_model = 5;
X_train = [];
Y_train = [];

% Create sequences for training (sliding window approach)
seq_length = 100;  % 0.5 second window @ 200Hz
stride = 50;       % 0.25 second stride

for ep = 1:num_epochs
    ic_data = all_ic_activations(1:num_ics_model, :, ep)';  % (samples × ICs)
    label_data = labels(ep, :)';  % (samples × 1)
    
    % Create sequences
    for s = 1:stride:(num_samples - seq_length + 1)
        seq = ic_data(s:s+seq_length-1, :);
        target = max(label_data(s:s+seq_length-1));  % 1 if any lateral movement in window
        
        X_train = cat(3, X_train, seq');  % (features × seq_length × num_sequences)
        Y_train = [Y_train; target];
    end
end

fprintf('\nSequences created:\n');
fprintf('  Number of sequences: %d\n', length(Y_train));
fprintf('  Positive samples: %d (%.2f%%)\n', ...
    sum(Y_train), 100 * sum(Y_train) / length(Y_train));

%% 8. Define and Train LSTM Network
fprintf('\nDefining LSTM network...\n');

layers = [ ...
    sequenceInputLayer(num_ics_model)
    lstmLayer(50, 'OutputMode', 'last')
    dropoutLayer(0.5)
    fullyConnectedLayer(2)  % 2 classes: no movement, lateral movement
    softmaxLayer
    classificationLayer];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Convert labels to categorical
Y_train_cat = categorical(Y_train);

% Split data into train/validation (80/20)
num_train = round(0.8 * length(Y_train));
idx = randperm(length(Y_train));

X_train_split = X_train(:, :, idx(1:num_train));
Y_train_split = Y_train_cat(idx(1:num_train));
X_val_split = X_train(:, :, idx(num_train+1:end));
Y_val_split = Y_train_cat(idx(num_train+1:end));

fprintf('Training LSTM model...\n');
fprintf('  Training samples: %d\n', num_train);
fprintf('  Validation samples: %d\n', length(Y_train) - num_train);

% Convert to cell arrays for sequence input
X_train_cell = {};
for i = 1:size(X_train_split, 3)
    X_train_cell{i} = X_train_split(:, :, i);
end

X_val_cell = {};
for i = 1:size(X_val_split, 3)
    X_val_cell{i} = X_val_split(:, :, i);
end

% Train the network
net = trainNetwork(X_train_cell', Y_train_split, layers, options);

fprintf('\n✓ Model training complete!\n');

%% 9. Test Model on Validation Set
Y_pred = classify(net, X_val_cell');
accuracy = sum(Y_pred == Y_val_split) / length(Y_val_split);

fprintf('\nModel Performance:\n');
fprintf('  Validation Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
figure;
confusionchart(Y_val_split, Y_pred);
title('Lateral Eye Movement Detection - Confusion Matrix');

fprintf('\n✓ MATLAB preprocessing and DL model complete!\n');
fprintf('  Model can be used to detect lateral eye movements in new data\n');

%% AW Passive Voice Task
clear all
close all

% We use PTB-3
AssertOpenGL;

% Open Datapixx, and stop any schedules which might already be running
Datapixx('Open');
Datapixx('StopAllSchedules');
Datapixx('InitAudio');
Datapixx('SetAudioVolume', 0.1);    % Not too loud
Datapixx('RegWrRd');    % Synchronize Datapixx registers to local register cache

% Show how many TTL output bits are in the Datapixx
nBits = Datapixx('GetDoutNumBits');
fprintf('\nDATAPixx has %d TTL output bits\n\n', nBits);

%% Setup experiment parameters and timers
KbName('UnifyKeyNames');  % This ensures consistent key names across platforms
escapeKey = KbName('ESCAPE');

% Get participant info using input dialog
prompt = {'Participant ID:'};
dlgtitle = 'Experiment Info';
dims = [1 35];
definput = {sprintf('%06d', randi([0 999999]))};
answer = inputdlg(prompt, dlgtitle, dims, definput);

participant = answer{1};
date_str = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

% Setup data file
dataFile = fullfile(pwd, 'data', ...
    sprintf('%s_PassiveEmoVoice_%s.mat', participant, date_str));
[filepath,~,~] = fileparts(dataFile);
if ~exist(filepath, 'dir')
    mkdir(filepath);
end

nBlocks = 5;
nTrialsPerBlock = 60;

% Load stimulus information
% Load stimulus information
[~, ~, stimInfo] = xlsread('TrialStimInfo.xlsx');
% Remove header row if it exists
if ischar(stimInfo{1,1}) || iscell(stimInfo{1,1})
    stimInfo = stimInfo(2:end,:);
end
% Convert emotion types and sentence numbers to numeric arrays
for i = 1:size(stimInfo,1)
    if iscell(stimInfo{i,2})
        stimInfo{i,2} = str2double(stimInfo{i,2});
    end
    if iscell(stimInfo{i,3})
        stimInfo{i,3} = str2double(stimInfo{i,3});
    end
end

% Initialize experiment info structure
expInfo = struct();
expInfo.participant = participant;
expInfo.date = date_str;
expInfo.nBlocks = nBlocks;
expInfo.nTrialsPerBlock = nTrialsPerBlock;

% Create backup directory
saveDir = fullfile(pwd, 'data');
backupDir = fullfile(saveDir, 'backups');
if ~exist(backupDir, 'dir')
    mkdir(backupDir)
end

% Initial TTL
deliverTTL(6);

%% Run experiment blocks
%try
% Run blocks

overallStart = GetSecs;
expInfo.experimentStartTime = overallStart;
allBlockData = [];  % Initialize empty array to store all blocks

%% add in Vpixx time recovery
for blockNum = 1:nBlocks
    fprintf('Starting Block %d\n', blockNum);
    deliverTTL(6);
    % Generate trial sequence for this block
    [emotions, trials, sentences] = trialRandomiser(stimInfo);
    
    %     % Save sequence information
    %     blockData.blockNum = blockNum;
    %     blockData.emotionSequence = emotions;
    %     blockData.trialOrder = trials;
    %     blockData.sentenceNumbers = sentences;
    
    blockData = zeros(nTrialsPerBlock,5);  % Note: changed to 5 columns
    blockData(:,1) = blockNum;
    blockData(:,2) = 1:1:length(trials);
    blockData(:,3) = emotions;
    blockData(:,4) = sentences;
    blockData(:,5) = trials;
    
    % Accumulate blockData
    allBlockData = [allBlockData; blockData];  % Append current block data
    
    % Convert to table
    blockDataTable = array2table(allBlockData, ...
        'VariableNames', {'BlockNum', 'TrialNumWithinBlock', 'EmotionType', ...
        'SentenceNum', 'GlobalTrialNum'});
    
    % Save block backup
    backupFile = sprintf('%s_backup_block%d.mat', participant, blockNum);
    backupPath = fullfile(backupDir, backupFile);
    save(backupPath, 'blockDataTable');
    
    %     % Save block data
    %     save(sprintf('%s_block%d_order.mat', dataFile, blockNum), 'blockData');
    %     save(sprintf('blockData.mat',dataFile, 'blockData'));
    
    blockStart = GetSecs;
    blockTimes.blockNum = blockNum;
    blockTimes.blockStart=blockStart;
    
    % Run trials in this block
    for trialNum = 1:nTrialsPerBlock
        fprintf('Starting trial %d\n', trialNum);
        
        % Reset timing for this trial
        %trialStart = toc(globalClock);
        
        % Get stimulus information
        rowTrial = trials(trialNum);
        filepathStim = stimInfo{rowTrial, 1};
        emotionType = stimInfo{rowTrial, 2};
        sentenceNum = stimInfo{rowTrial, 3};
        
        % Record trial start
        deliverTTL(4);
        startTrialPTB = GetSecs;
        
        % Save trial data
        trialData.blockNum = blockNum;
        trialData.trialNum = trialNum;
        trialData.stimFile = filepathStim;
        trialData.stimNum = rowTrial;
        trialData.emotionType = emotionType;
        trialData.sentenceNum = sentenceNum;
        trialData.trialStartTime = startTrialPTB;
        
        % Jitter period
        jitter = 2.0 + 0.5*rand;
        trialData.jitter = jitter;
        pause(jitter);
        
        % Play audio
        [y, fs] = audioread(filepathStim);
        % Ensure stereo output (2 channels)
        if size(y, 2) == 1
            y = repmat(y, 1, 2);  % Duplicate mono to both channels
        end
        y = y';  % DataPixx expects channels-by-samples format
        
        % Configure audio playback
        maxScheduleFrames = size(y, 2);  % Number of samples
        bufferAddress = 16e6;
        
        % Initialize audio system
        Datapixx('InitAudio');
        Datapixx('RegWrRd');
        
        % Write audio to buffer
        Datapixx('WriteAudioBuffer', y, bufferAddress);
        Datapixx('RegWrRd');
        
        % Set and start audio schedule
        Datapixx('SetAudioSchedule', 0, fs, maxScheduleFrames, 3, bufferAddress); %3 in 0 position
        %Datapixx('SetAudioSchedule', 0, sample_freq, nTotalFrames_O, 3, 0, nTotalFrames_O);
        
        % Start the audio
        Datapixx('StartAudioSchedule');
        audioStart = GetSecs;
        
        % Set emotion type TTL at start of audio
        switch trialData.emotionType
            case 1
                doutValue = bin2dec('000000000000000000010000');
            case 2
                doutValue = bin2dec('000000000000000000001000');
            case 3
                doutValue = bin2dec('000000000000000000000100');
        end
        
        bitMask = hex2dec('ffffff');
        Datapixx('SetDoutValues', doutValue, bitMask);
        Datapixx('RegWrRd');
        
        % Wait for the duration of the audio
        WaitSecs(maxScheduleFrames/fs);
        
        % Record audio end time and send emotion type TTL again
        audioEnd = GetSecs;
        Datapixx('SetDoutValues', doutValue, bitMask);
        Datapixx('RegWrRd');
        
        % Reset TTL to zero
        Datapixx('SetDoutValues', 0, bitMask);
        Datapixx('RegWrRd');
        
        % Wait until 2 seconds from audio start (if not already passed)
        timeUntil2s = 2 - (GetSecs - audioStart);
        if timeUntil2s > 0
            WaitSecs(timeUntil2s);
        end
        
        % Send trigger 5 at 2 seconds post-audio-start
        trigger5Value = bin2dec('000000000000000000100000'); % Trigger 5
        Datapixx('SetDoutValues', trigger5Value, bitMask);
        Datapixx('RegWrRd');
        
        % Quick reset of trigger 5
        WaitSecs(0.001);  % Brief delay to ensure trigger is registered
        Datapixx('SetDoutValues', 0, bitMask);
        Datapixx('RegWrRd');
        
        % Stop audio schedule
        Datapixx('StopAudioSchedule');
        Datapixx('RegWrRd');
        
        % Record trial end time
        endTrlPTB = GetSecs;
        
        % Save timing data
        audioLength = audioEnd-audioStart;
        trialData.audioStartTime = audioStart;
        trialData.audioEndTime = audioEnd;
        trialData.audioDuration = audioLength;
        trialData.trialEndTime = endTrlPTB;
        
        % Check for escape key
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown && keyCode(escapeKey)%KbName('ESCAPE'))
            fprintf('Escape pressed. Ending experiment...\n');
            Datapixx('Close');
            return;
        end
        
        % Save trial data
        if ~exist('experimentData', 'var')
            experimentData = trialData;
        else
            experimentData(end+1) = trialData;
            
        end
        
        % Convert to table and save backup
        experimentDataTable = struct2table(experimentData);
        save(backupPath, 'experimentDataTable', '-append');
        
        %save(dataFile, 'experimentData');
        %deliverTTL(5); %may not need this if delivered before
    end
    
    % Show break screen if not last block
    if blockNum < nBlocks
        
        blockEnd = GetSecs;
        blockTimes.blockEnd = blockEnd;
        blockTimes.blockDuration = blockEnd-blockStart;
        
        % After collecting block times
        if ~exist('blockTimeData', 'var')
            blockTimeData = blockTimes;
        else
            blockTimeData(end+1) = blockTimes;
        end
        % Convert to table and save with backup
        blockTimesTable = struct2table(blockTimeData);
        save(backupPath, 'blockTimesTable', '-append');
        fprintf('Block break - Press R to resume or Q to quit\n');
        
        while true
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown
                if keyCode(KbName('r'))
                    fprintf('Resuming experiment...\n');
                    blockBreakEnd = GetSecs;
                    blockBreakDuration = blockBreakEnd-blockEnd;
                    blockTimes.blockBreakDuration = blockBreakDuration;
                    % Update the break duration in the existing entry
                    blockTimeData(end).blockBreakDuration = blockBreakDuration;
                    % Save updated timing info
                    blockTimesTable = struct2table(blockTimeData);
                    save(backupPath, 'blockTimesTable', '-append');
                    
                    break
                    
                elseif keyCode(KbName('q')) || keyCode(KbName('ESCAPE'))
                    fprintf('Quitting experiment...\n');
                    Datapixx('Close');
                    return
                end
            end
            pause(0.01);
        end
    end
end

% At the very end of your experiment

%Save last block data
blockTimes.blockNum = nBlocks;
blockTimes.blockStart = blockStart;
blockEnd = GetSecs;
blockTimes.blockEnd = blockEnd;
blockTimes.blockDuration = blockEnd - blockStart;
blockTimes.blockBreakDuration = NaN;

% Add to blockTimeData
if ~exist('blockTimeData', 'var')
    blockTimeData = blockTimes;
else
    blockTimeData(end+1) = blockTimes;
end

% Convert to table with the added information
blockTimesTable = struct2table(blockTimeData);

% Final TTL
deliverTTL(6);
expEnd = GetSecs;
expInfo.experimentEndTime = expEnd;
expInfo.experimentDuration = expEnd - overallStart;

% Add these to the last row of blockTimes
blockTimes.blockNum = nBlocks + 1;  % or however you're numbering your last block
blockTimes.blockStart = overallStart;
blockTimes.blockEnd = expEnd;
blockTimes.blockDuration = expEnd - overallStart;
blockTimes.blockBreakDuration = NaN;  % No break after the last block

% Add to blockTimeData
if ~exist('blockTimeData', 'var')
    blockTimeData = blockTimes;
else
    blockTimeData(end+1) = blockTimes;
end

% Convert to table with the added information
blockTimesTable = struct2table(blockTimeData);

if ~exist('experimentDataTable', 'var')
    experimentDataTable = struct2table(experimentData);
end

% Save all information in a single MAT file
fileName = sprintf('%s_PassiveEmoVoice_%s.mat', participant, date_str);
savePath = fullfile(saveDir, fileName);

% Save all data
save(savePath, 'expInfo', 'blockDataTable', 'blockTimesTable', ...
    'experimentDataTable', '-v7.3');

% Create and save summary Excel file
summaryTable = experimentDataTable;

%summaryTable = experimentDataTable(:, {'blockNum', 'trialNum', 'emotionType', ...
%    'sentenceNum', 'audioDuration', 'trialStartTime', 'trialEndTime'});

% You could also add a summary row to your Excel output
summaryTable.Properties.UserData.experimentStartTime = expInfo.experimentStartTime;
summaryTable.Properties.UserData.experimentEndTime = expInfo.experimentEndTime;
summaryTable.Properties.UserData.experimentDuration = expInfo.experimentDuration;

excelPath = fullfile(saveDir, sprintf('%s_PassiveEmoVoice_%s_summary.xlsx', ...
    participant, date_str));
writetable(summaryTable, excelPath);

fprintf('Experiment complete\n');
fprintf('Data saved to: %s\n', savePath);

% Clean up
Datapixx('Close');

%% Helper Functions
function deliverTTL(ttlNum)
% Deliver TTL pulse based on number
switch ttlNum
    case 1
        doutValue = bin2dec('000000000000000000010000');
    case 2
        doutValue = bin2dec('000000000000000000001000');
    case 3
        doutValue = bin2dec('000000000000000000000100');
    case 4
        doutValue = bin2dec('000000000000000000100000');
    case 5
        doutValue = bin2dec('000000000000000001000000');
    case 6
        doutValue = bin2dec('000000000000000010000000');
end

bitMask = hex2dec('ffffff');
Datapixx('SetDoutValues', doutValue, bitMask);
Datapixx('RegWrRd');

WaitSecs(0.01);
% Reset to zero
Datapixx('SetDoutValues', 0, bitMask);
Datapixx('RegWrRd');
end

function [emotionSequence, trialOrder, sentenceNumbers] = trialRandomiser(stimInfo)
% Randomize trials with constraints
maxAttempts = 2000;

for attempt = 1:maxAttempts
    % Generate basic emotion sequence (1=angry, 2=neutral, 3=happy)
    emotionSequence = repelem([1 2 3], 20);
    emotionSequence = emotionSequence(randperm(60));
    
    % Check for three consecutive same emotions
    if any(emotionSequence(3:end) == emotionSequence(2:end-1) & ...
            emotionSequence(2:end-1) == emotionSequence(1:end-2))
        continue
    end
    
    % Set up trial indices for each emotion
    angryTrials = 1:20;
    neutralTrials = 21:40;
    happyTrials = 41:60;
    
    % Shuffle within each emotion group
    angryTrials = angryTrials(randperm(20));
    neutralTrials = neutralTrials(randperm(20));
    happyTrials = happyTrials(randperm(20));
    
    % Build trial order and sentence sequence
    trialOrder = zeros(1, 60);
    sentenceNumbers = zeros(1, 60);
    emotionCounters = containers.Map({1,2,3}, {1,1,1});
    
    for i = 1:60
        emotion = emotionSequence(i);
        counter = emotionCounters(emotion);
        
        if emotion == 1
            trialOrder(i) = angryTrials(counter);
        elseif emotion == 2
            trialOrder(i) = neutralTrials(counter);
        else % emotion == 3
            trialOrder(i) = happyTrials(counter);
        end
        
        sentenceNumbers(i) = stimInfo{trialOrder(i), 3};
        emotionCounters(emotion) = counter + 1;
    end
    
    % Check for three consecutive same sentences
    if ~any(sentenceNumbers(3:end) == sentenceNumbers(2:end-1) & ...
            sentenceNumbers(2:end-1) == sentenceNumbers(1:end-2))
        return
    end
end

error('Could not generate balanced trial sequence after %d attempts', maxAttempts);
end
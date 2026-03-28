%% AW Passive Voice Task for Infants

%% Instructions
%- make sure opm recording has started first
%- then on experiment script, make sure cursor is in command window, then run
%- put in subject number into the prompt (e.g. S4_1953)
%- then click ok and task will start
%- after each block, you have the option to press 'r' to resume, or 'q' to quit. 
%- to pause in the middle of the block, hold down Esc then get same options of r or q
%- if baby needs to stop recording, pause first rather than quitting in case we can continue the blocks.
%- at the end of experiment or when quitting, it has saved correctly when it shows the excel path saved.

clear all
close all

% We use PTB-3
AssertOpenGL;

% Open Datapixx, and stop any schedules which might already be running
Datapixx('Open');
Datapixx('StopAllSchedules');
Datapixx('InitAudio');
Datapixx('SetAudioVolume', 0.3);    % Set volume once at start
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

nBlocks = 12;
nTrialsPerBlock = 20;

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
expInfo.completed = false;  % Track if experiment completed normally

% Create backup directory
saveDir = fullfile(pwd, 'data');
backupDir = fullfile(saveDir, 'backups');
if ~exist(backupDir, 'dir')
    mkdir(backupDir)
end

% Initialize timing and pause tracking
pauseData = [];  % Track all pauses
blockBreakData = [];  % Track all block breaks
experimentData = [];
blockTimeData = [];
allBlockData = [];% Initialize empty array to store all blocks

% Initial TTL
deliverTTL(7);

%% Run experiment blocks
try
    overallStart = GetSecs;
    expInfo.experimentStartTime = overallStart;
    
    for blockNum = 1:nBlocks
        fprintf('Starting Block %d\n', blockNum);
        deliverTTL(6);
        % Generate trial sequence for this block
        [emotions, trials, sentences] = trialRandomiser(stimInfo);
        
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
        
        blockStart = GetSecs;
        blockTimes.blockNum = blockNum;
        blockTimes.blockStart=blockStart;
        
        % Run trials in this block
        for trialNum = 1:nTrialsPerBlock
            fprintf('Starting trial %d\n', trialNum);
            
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
            [y, fs] = psychwavread(filepathStim);  % Use psychwavread for better compatibility
            % Ensure stereo output (2 channels)
            if size(y, 2) == 1
                y = repmat(y, 1, 2);  % Duplicate mono to both channels
            end
            
            % Apply dB-based volume control (more acoustically appropriate)
            dB_increase = 6;  % Increase by 6 dB (adjustable)
            dB_gain = db2mag(dB_increase);
            y = y * dB_gain;
            
            % Clip to prevent distortion
            y = max(-1, min(1, y));
            
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
            Datapixx('SetAudioSchedule', 0, fs, maxScheduleFrames, 3, bufferAddress);
            
            % Start the audio
            Datapixx('StartAudioSchedule');
            audioStart = GetSecs;
            
            % Set emotion type TTL at start of audio
            switch trialData.emotionType
                case 1
                    doutValue = bin2dec('000000000000000000010000'); % 16
                case 3
                    doutValue = bin2dec('000000000000000000000100'); % 4
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
            postaudtriggerValue = bin2dec('000000000000000000100001'); % 33
            Datapixx('SetDoutValues', postaudtriggerValue, bitMask);
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
            if keyIsDown && keyCode(escapeKey)
                fprintf('Paused experiment - Press R to resume or Q to quit\n');
                deliverTTL(2); %pause trigger
                
                pauseStart = GetSecs;
                pauseHandled = false;
                
                while true
                    [keyIsDown, ~, keyCode] = KbCheck;
                    if keyIsDown
                        if keyCode(KbName('r'))
                            fprintf('Resuming experiment...\n');
                            deliverTTL(9); %resume trigger
                            pauseEnd = GetSecs;
                            
                            % Record pause information
                            pauseInfo.blockNum = blockNum;
                            pauseInfo.trialNum = trialNum;
                            pauseInfo.pauseStart = pauseStart;
                            pauseInfo.pauseEnd = pauseEnd;
                            pauseInfo.pauseDuration = pauseEnd - pauseStart;
                            pauseInfo.pauseType = 'trial_pause';
                            
                            pauseData = [pauseData; pauseInfo];
                            pauseHandled = true;
                            break
                        elseif keyCode(KbName('q'))
                            fprintf('Quitting experiment...\n');
                            deliverTTL(10); %exit trigger
                            
                            % Record pause information before quitting
                            pauseEnd = GetSecs;
                            pauseInfo.blockNum = blockNum;
                            pauseInfo.trialNum = trialNum;
                            pauseInfo.pauseStart = pauseStart;
                            pauseInfo.pauseEnd = pauseEnd;
                            pauseInfo.pauseDuration = pauseEnd - pauseStart;
                            pauseInfo.pauseType = 'trial_quit';
                            
                            pauseData = [pauseData; pauseInfo];
                            
                            % Save data before quitting
                            saveExperimentData(expInfo, participant, date_str, saveDir, ...
                                experimentData, blockTimeData, allBlockData, pauseData, blockBreakData);
                            Datapixx('Close');
                            return
                        end
                    end
                    pause(0.01);
                end
            end
            
            % Save trial data
            if isempty(experimentData)
                experimentData = trialData;
            else
                experimentData(end+1) = trialData;
            end
            
            % Convert to table and save backup
            experimentDataTable = struct2table(experimentData);
            save(backupPath, 'experimentDataTable', '-append');
        end
        
        % Show break screen if not last block
        if blockNum < nBlocks
            blockEnd = GetSecs;
            blockTimes.blockEnd = blockEnd;
            blockTimes.blockDuration = blockEnd-blockStart;
            
            % After collecting block times
            if isempty(blockTimeData)
                blockTimeData = blockTimes;
            else
                blockTimeData(end+1) = blockTimes;
            end
            % Convert to table and save with backup
            blockTimesTable = struct2table(blockTimeData);
            save(backupPath, 'blockTimesTable', '-append');
            fprintf('Block break - Press R to resume or Q to quit\n');
            
            breakStart = GetSecs;
            
            while true
                [keyIsDown, ~, keyCode] = KbCheck;
                if keyIsDown
                    if keyCode(KbName('r'))
                        fprintf('Resuming experiment...\n');
                        blockBreakEnd = GetSecs;
                        blockBreakDuration = blockBreakEnd-blockEnd;
                        blockTimes.blockBreakDuration = blockBreakDuration;
                        
                        % Record detailed break information
                        breakInfo.blockNum = blockNum;
                        breakInfo.breakStart = breakStart;
                        breakInfo.breakEnd = blockBreakEnd;
                        breakInfo.breakDuration = blockBreakDuration;
                        breakInfo.breakType = 'block_break';
                        
                        blockBreakData = [blockBreakData; breakInfo];
                        
                        % Update the break duration in the existing entry
                        blockTimeData(end).blockBreakDuration = blockBreakDuration;
                        % Save updated timing info
                        blockTimesTable = struct2table(blockTimeData);
                        save(backupPath, 'blockTimesTable', '-append');
                        break
                        
                    elseif keyCode(KbName('q')) || keyCode(KbName('ESCAPE'))
                        fprintf('Quitting experiment...\n');
                        deliverTTL(10); %exit trigger
                        
                        % Record break information before quitting
                        blockBreakEnd = GetSecs;
                        breakInfo.blockNum = blockNum;
                        breakInfo.breakStart = breakStart;
                        breakInfo.breakEnd = blockBreakEnd;
                        breakInfo.breakDuration = blockBreakEnd - breakStart;
                        breakInfo.breakType = 'block_quit';
                        
                        blockBreakData = [blockBreakData; breakInfo];
                        
                        % Save data before quitting
                        saveExperimentData(expInfo, participant, date_str, saveDir, ...
                            experimentData, blockTimeData, allBlockData, pauseData, blockBreakData);
                        Datapixx('Close');
                        return
                    end
                end
                pause(0.01);
            end
        end
    end
    
    % If we reach here, experiment completed normally
    expInfo.completed = true;
    
    % Save last block data
    blockTimes.blockNum = nBlocks;
    blockTimes.blockStart = blockStart;
    blockEnd = GetSecs;
    blockTimes.blockEnd = blockEnd;
    blockTimes.blockDuration = blockEnd - blockStart;
    blockTimes.blockBreakDuration = NaN;
    
    % Add to blockTimeData
    if isempty(blockTimeData)
        blockTimeData = blockTimes;
    else
        blockTimeData(end+1) = blockTimes;
    end
    
    % Convert to table with the added information
    blockTimesTable = struct2table(blockTimeData);
    
    % Final TTL
    deliverTTL(8);
    expEnd = GetSecs;
    expInfo.experimentEndTime = expEnd;
    expInfo.experimentDuration = expEnd - overallStart;
    
    % Save final data
    saveExperimentData(expInfo, participant, date_str, saveDir, ...
        experimentData, blockTimeData, allBlockData, pauseData, blockBreakData);
    
    fprintf('Experiment complete\n');
    
catch ME
    % If any error occurs, save what we have
    fprintf('Error occurred: %s\n', ME.message);
    saveExperimentData(expInfo, participant, date_str, saveDir, ...
        experimentData, blockTimeData, allBlockData, pauseData, blockBreakData);
    rethrow(ME);
end

% Clean up
Datapixx('Close');

%% Nested function to save experiment data
function saveExperimentData(expInfo, participant, date_str, saveDir, ...
    experimentData, blockTimeData, allBlockData, pauseData, blockBreakData)
% Save experiment data - now as a separate function with explicit parameters

% Set end time if not already set
if ~isfield(expInfo, 'experimentEndTime')
    expInfo.experimentEndTime = GetSecs;
    expInfo.experimentDuration = expInfo.experimentEndTime - expInfo.experimentStartTime;
end

% Ensure we have tables for saving
if ~isempty(experimentData)
    experimentDataTable = struct2table(experimentData);
else
    experimentDataTable = table(); % Empty table if no trials completed
end

if ~isempty(blockTimeData)
    blockTimesTable = struct2table(blockTimeData);
else
    blockTimesTable = table(); % Empty table if no blocks completed
end
f
if ~isempty(allBlockData)
    blockDataTable = array2table(allBlockData, ...
        'VariableNames', {'BlockNum', 'TrialNumWithinBlock', 'EmotionType', ...
        'SentenceNum', 'GlobalTrialNum'});
else
    blockDataTable = table(); % Empty table if no trials completed
end

% Create tables for pause and break data
if ~isempty(pauseData)
    pauseDataTable = struct2table(pauseData);
else
    pauseDataTable = table();
end

if ~isempty(blockBreakData)
    blockBreakDataTable = struct2table(blockBreakData);
else
    blockBreakDataTable = table();
end

% Save all information in a single MAT file
fileName = sprintf('%s_PassiveEmoVoice_%s.mat', participant, date_str);
savePath = fullfile(saveDir, fileName);

% Save all data including pause and break information
save(savePath, 'expInfo', 'blockDataTable', 'blockTimesTable', ...
    'experimentDataTable', 'pauseDataTable', 'blockBreakDataTable', '-v7.3');

% Create and save summary Excel file
if ~isempty(experimentDataTable)
    summaryTable = experimentDataTable;
    
    % Add experiment summary information
    summaryTable.Properties.UserData.experimentStartTime = expInfo.experimentStartTime;
    summaryTable.Properties.UserData.experimentEndTime = expInfo.experimentEndTime;
    summaryTable.Properties.UserData.experimentDuration = expInfo.experimentDuration;
    summaryTable.Properties.UserData.completed = expInfo.completed;
    
    excelPath = fullfile(saveDir, sprintf('%s_PassiveEmoVoice_%s_summary.xlsx', ...
        participant, date_str));
    
    % Write main data
    writetable(summaryTable, excelPath, 'Sheet', 'TrialData');
    
    % Write pause data if available
    if ~isempty(pauseDataTable)
        writetable(pauseDataTable, excelPath, 'Sheet', 'PauseData');
    end
    
    % Write break data if available
    if ~isempty(blockBreakDataTable)
        writetable(blockBreakDataTable, excelPath, 'Sheet', 'BreakData');
    end
    
    % Write block timing data if available
    if ~isempty(blockTimesTable)
        writetable(blockTimesTable, excelPath, 'Sheet', 'BlockTiming');
    end
end

fprintf('Data saved to: %s\n', savePath);
if exist('excelPath', 'var')
    fprintf('Excel summary saved to: %s\n', excelPath);
end
end

%% Helper Functions
function deliverTTL(ttlNum)
% Deliver TTL pulse based on number
switch ttlNum
    case 1
        doutValue = bin2dec('000000000000000000010000'); % 16
    case 2
        doutValue = bin2dec('000000000000000000001001'); % 9
    case 3
        doutValue = bin2dec('000000000000000000000100'); % 4
    case 4
        doutValue = bin2dec('000000000000000000100000'); % 32
    case 5
        doutValue = bin2dec('000000000000000001000000'); % 64
    case 6
        doutValue = bin2dec('000000000000000010000000'); % 128
    case 7
        doutValue = bin2dec('000000000000000010000001'); % 129
    case 8
        doutValue = bin2dec('000000000000000010000010'); % 130
    case 9
        doutValue = bin2dec('000000000000000000001010'); % 10
    case 10
        doutValue = bin2dec('000000000000000000001011'); % 11
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
    % Generate basic emotion sequence (1=angry, 3=happy)
    emotionSequence = repelem([1 3], 10);
    emotionSequence = emotionSequence(randperm(20));
    
    % Check for three consecutive same emotions
    if any(emotionSequence(3:end) == emotionSequence(2:end-1) & ...
            emotionSequence(2:end-1) == emotionSequence(1:end-2))
        continue
    end
    
    % Set up trial indices for each emotion
    angryTrials = 1:10;
    happyTrials = 11:20;
    
    % Shuffle within each emotion group
    angryTrials = angryTrials(randperm(10));
    happyTrials = happyTrials(randperm(10));
    
    % Build trial order and sentence sequence
    trialOrder = zeros(1, 20);
    sentenceNumbers = zeros(1, 20);
    emotionCounters = containers.Map({1,3}, {1,1});
    
    for i = 1:20
        emotion = emotionSequence(i);
        counter = emotionCounters(emotion);
        
        if emotion == 1
            trialOrder(i) = angryTrials(counter);
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
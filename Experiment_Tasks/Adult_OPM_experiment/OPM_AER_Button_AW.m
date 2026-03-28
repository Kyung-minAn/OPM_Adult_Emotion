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

%button setup
timeOutSeconds = 5;

%% create the sound
sample_freq = 48000;
NormalStl(1,:) =  MakeBeep(1000, 0.1, sample_freq); % normal beep
NormalStl(2,:) = NormalStl(1,:);
nChannels_N = size(NormalStl, 1);
nTotalFrames_N = size(NormalStl, 2);

%trial
Ntrial = 120; 
%trlstart=zeros(length(Ntrial),1);
%jitter=zeros(length(Ntrial),1);
%endtrl=zeros(length(Ntrial),1);

for i=1:Ntrial
    expdata.trlnum = i;
    trlstarttime=GetSecs;
    expdata.trlstart=trlstarttime;

    aud = NormalStl;
    Datapixx('WriteAudioBuffer', aud, 0);
    Datapixx('SetAudioSchedule', 0, sample_freq, nTotalFrames_N, 3, 0, nTotalFrames_N);

    % Start the playback
    Datapixx('SetDoutValues', 4);
    Datapixx('StartAudioSchedule');
    Datapixx('RegWrRd');    % Synchronize Datapixx registers to local register cache
    WaitSecs(0.1);%was 0.5  <-----#
    Datapixx('SetDoutValues', 0);
    Datapixx('RegWrRd');
        
    %Button response
    % Initialize return values
    buttonNum = 0;    % Default return for timeout or no press
    pressTime = -1;   % Default time for timeout or no press
    RT = -1; %if no button press
    try
        % Set up button logging
        Datapixx('EnableDinDebounce');
        Datapixx('SetDinLog');
        Datapixx('StartDinLog');

        % Main loop
        fprintf('Ready to detect button press. Will timeout after %.1f seconds.\n', timeOutSeconds);

        buttonPressed = false;  % Flag to track if a button was pressed
        startTime = GetSecs();  % Get start time
        while ~buttonPressed && (GetSecs() - startTime) < timeOutSeconds

            % Check for button presses
            Datapixx('RegWrRd');
            status = Datapixx('GetDinStatus');

            if status.newLogFrames > 0
                logData = Datapixx('ReadDinLog', status.newLogFrames);
                currentTime = GetSecs();  % Get time of button press

                % Debug - let's see what we're getting
                %fprintf('Log Data Structure:\n');
                %disp(logData);

                % Process button press
                buttonCode = logData;  % Assuming direct value
                % Map button codes to actions and set return values
                if buttonCode == 64513  % Red button
                    fprintf('Red button pressed\n');
                    buttonNum = 1;
                    pressTime = currentTime - startTime;
                    %fprintf(pressTime);
                    RT = currentTime-trlstarttime;
                    % Send TTL pulse
                    Datapixx('SetDoutValues', bin2dec('000000000000000000010000'));
                    Datapixx('RegWrRd');
                    WaitSecs(0.001);  % Brief pulse
                    Datapixx('SetDoutValues', 0);
                    Datapixx('RegWrRd');
                    buttonPressed = true;  % Set flag to exit loop
                end
            end
            % Small delay to prevent CPU hogging
            WaitSecs(0.001);
        end

        % Check if we timed out
        if ~buttonPressed
            fprintf('Timeout occurred - no button pressed within %.1f seconds\n', timeOutSeconds);
        end

        % Clean up just the button logging
        Datapixx('StopDinLog');

    catch err
        % Error handling
        fprintf('Error: %s\n', err.message);
        Datapixx('StopDinLog');
    end

    fprintf('Trial finished.\n');

    expdata.pressTime=pressTime;
    expdata.RT = RT;
    jitter=3.5 + 1*rand; %jitter between 3.5-4.5s
    expdata.jitter = jitter;
    WaitSecs(jitter);  %jitter 1.8-2.3s
    expdata.trlend =GetSecs;

    % Add to blockTimeData
    if ~exist('AER_expdata', 'var')
        AER_expdata = expdata;
    else
        AER_expdata(end+1) = expdata;
    end

end

Datapixx('StopAudioSchedule');
Datapixx('RegWrRd');
Datapixx('Close');

fprintf('\nSavingData\n\n');
AER_expdata_Table = struct2table(AER_expdata);
saveDir='<enter_save_directory>';
participant = '<enter_participant_id>';
date_str = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
fileName = sprintf('%s_AER_Button_%s.mat', participant, date_str);
savePath = fullfile(saveDir, fileName);

save(savePath, 'AER_expdata_Table', '-v7.3');
fprintf('\nAER completed\n\n');
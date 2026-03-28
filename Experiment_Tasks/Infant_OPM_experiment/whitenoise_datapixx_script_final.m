%% White noise datapixx script

%% Instructions
%- make sure new opm recording has started first
%- cursor in command window
%- run script
%- click ok on script prompt and it will start
%- it will automatically run for 3 mins.

clear all
close all

% using PTB-3
AssertOpenGL;

%% Get participant info using input dialog
prompt = {'Participant ID:'};
dlgtitle = 'White Noise Experiment Info';
dims = [1 35];
definput = {sprintf('%06d', randi([0 999999]))};
answer = inputdlg(prompt, dlgtitle, dims, definput);

% Check if user cancelled dialog
if isempty(answer)
    fprintf('Experiment cancelled by user\n');
    return;
end

participant_id = answer{1};
fprintf('Participant ID: %s\n', participant_id);

% Setup keyboard checking
KbName('UnifyKeyNames');
escapeKey = KbName('ESCAPE');

% Open Datapixx, and stop any schedules which might already be running
Datapixx('Open');
Datapixx('StopAllSchedules');
Datapixx('InitAudio');
Datapixx('SetAudioVolume', 0.15); % between 0-1
Datapixx('RegWrRd'); % Synchronize Datapixx registers to local register cache

% Show how many TTL output bits are in the Datapixx
nBits = Datapixx('GetDoutNumBits');
fprintf('\nDATAPixx has %d TTL output bits\n\n', nBits);

%% Generate infant-safe white noise
sample_freq = 48000;  % DataPixx typically uses 48kHz
duration = 3 * 60;    % 3 minutes in seconds
amplitude = 0.8;      % between 0-1

% Generate time vector
n_samples = duration * sample_freq;

% Generate white noise
white_noise_mono = amplitude * randn(n_samples, 1);

% Apply infant-friendly low-pass filtering
% Design gentle low-pass filter (cutoff around 8 kHz)
cutoff_freq = 8000;  % Hz
nyquist_freq = sample_freq/2;
normalized_cutoff = cutoff_freq / nyquist_freq;

% Create Butterworth filter
[b, a] = butter(4, normalized_cutoff, 'low');

% Apply filter
filtered_noise = filter(b, a, white_noise_mono);

% Normalize to prevent clipping
filtered_noise = filtered_noise / max(abs(filtered_noise)) * amplitude;

% Apply gentle fade-in and fade-out to not cause startling
fade_duration = 2; % seconds
fade_samples = round(fade_duration * sample_freq);

% Fade-in
fade_in = linspace(0, 1, fade_samples)';
filtered_noise(1:fade_samples) = filtered_noise(1:fade_samples) .* fade_in;

% Fade-out
fade_out = linspace(1, 0, fade_samples)';
filtered_noise(end-fade_samples+1:end) = filtered_noise(end-fade_samples+1:end) .* fade_out;

% Create stereo version (duplicate mono to both channels)
WhiteNoiseStimulus = [filtered_noise'; filtered_noise'];

nChannels = size(WhiteNoiseStimulus, 1);
nTotalFrames = size(WhiteNoiseStimulus, 2);

fprintf('Generated white noise: %.1f minutes, %d Hz, %.1f amplitude\n', ...
        duration/60, sample_freq, amplitude);
fprintf('Channels: %d, Total frames: %d\n', nChannels, nTotalFrames);

%% Experiment data structure
expdata.participant_id = participant_id;
expdata.duration_minutes = duration/60;
expdata.sample_freq = sample_freq;
expdata.amplitude = amplitude;
expdata.cutoff_freq = cutoff_freq;
expdata.paused_times = NaN;  % Track any pause events
expdata.resume_times = NaN;% Track resume events
expdata.early_termination = false;
expdata.termination_time=NaN;

%% Play white noise with triggers
fprintf('\n=== Starting White Noise Playback ===\n');
fprintf('Duration: %.1f minutes\n', duration/60);
fprintf('SAFETY: Monitor infant response throughout playback\n\n');

% Record start time
expdata.start_time = GetSecs;

% Load audio buffer
Datapixx('WriteAudioBuffer', WhiteNoiseStimulus, 0);

% Set up audio schedule
% Parameters: bufferBaseAddr, sampleRate, nFrames, lrMode, bufferBaseAddr, nFrames
Datapixx('SetAudioSchedule', 0, sample_freq, nTotalFrames, 3, 0, nTotalFrames);

% Send START trigger (TTL bit 4, same as your beeps)
fprintf('Sending START trigger...\n');
Datapixx('SetDoutValues', 4);  % Set bit 4 high
Datapixx('StartAudioSchedule');
Datapixx('RegWrRd');

% Brief trigger pulse
WaitSecs(0.1);
Datapixx('SetDoutValues', 0);  % Set all bits low
Datapixx('RegWrRd');

fprintf('White noise started at %.3f seconds\n', GetSecs - expdata.start_time);
fprintf('Playing for %.1f minutes...\n', duration/60);
fprintf('Press ESCAPE to pause, R to resume, Q to quit during playback\n\n');

% Wait for the full duration of the white noise WITH escape key monitoring
start_wait_time = GetSecs;
total_wait_time = duration + 0.5;  % Duration plus buffer

while (GetSecs - start_wait_time) < total_wait_time
    % Check for escape key
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown && keyCode(escapeKey)
        fprintf('Paused experiment - Press R to resume or Q to quit\n');
        
        % Send pause trigger
        Datapixx('SetDoutValues', 2);  % Pause trigger
        Datapixx('RegWrRd');
        WaitSecs(0.1);
        Datapixx('SetDoutValues', 0);
        Datapixx('RegWrRd');
        
        % Record pause time
        pause_time = GetSecs;
        if isnan(expdata.paused_times)
            expdata.paused_times = pause_time - expdata.start_time;
        else
            exdata.paused_times = [expdata.paused_times, pause_time - expdata.start_time];
        end
        
        % Pause the audio
        Datapixx('StopAudioSchedule');
        Datapixx('RegWrRd');
        
        % Wait for resume or quit command
        while true
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown
                if keyCode(KbName('r'))
                    fprintf('Resuming experiment...\n');
                    
                    % Send resume trigger
                    Datapixx('SetDoutValues', 9);  % Resume trigger
                    Datapixx('RegWrRd');
                    WaitSecs(0.1);
                    Datapixx('SetDoutValues', 0);
                    Datapixx('RegWrRd');
                    
                    % Record resume time
                    resume_time = GetSecs;
                    if isnan(expdata.resume_times)
                        expdata.resume_times = resume_time - expdata.start_time;
                    else
                        expdata.resume_times = [expdata.resume_times, resume_time - expdata.start_time];
                    end
                    
                    % Calculate remaining audio and restart
                    elapsed_time = pause_time - expdata.start_time;
                    remaining_frames = max(0, nTotalFrames - round(elapsed_time * sample_freq));
                    
                    if remaining_frames > 0
                        % Restart audio from where we left off (simplified - plays full audio again)
                        % Note: For exact resume, you'd need to calculate buffer offset
                        Datapixx('WriteAudioBuffer', WhiteNoiseStimulus, 0);
                        Datapixx('SetAudioSchedule', 0, sample_freq, nTotalFrames, 3, 0, nTotalFrames);
                        Datapixx('StartAudioSchedule');
                        Datapixx('RegWrRd');
                        
                        % Adjust wait time for pause duration
                        pause_duration = resume_time - pause_time;
                        start_wait_time = start_wait_time + pause_duration;
                    end
                    break;
                    
                elseif keyCode(KbName('q'))
                    fprintf('Quitting experiment...\n');
                    
                    % Send exit trigger
                    Datapixx('SetDoutValues', 10);  % Exit trigger
                    Datapixx('RegWrRd');
                    WaitSecs(0.1);
                    Datapixx('SetDoutValues', 0);
                    Datapixx('RegWrRd');
                    
                    % Record early termination
                    expdata.early_termination = true;
                    expdata.termination_time = GetSecs - expdata.start_time;
                    
                    % Clean up and exit
                    Datapixx('StopAudioSchedule');
                    Datapixx('RegWrRd');
                    Datapixx('Close');
                    
                    return;
                end
            end
            pause(0.01);  % Small pause to prevent excessive CPU usage
        end
    end
    
    pause(0.01);  % Small pause to prevent excessive CPU usage during monitoring
end

% Send END trigger
fprintf('Sending END trigger...\n');
expdata.end_time = GetSecs;
Datapixx('SetDoutValues', 8);  % Use bit 8 for end trigger (different from start)
Datapixx('RegWrRd');

% Brief trigger pulse
WaitSecs(0.1);
Datapixx('SetDoutValues', 0);  % Set all bits low
Datapixx('RegWrRd');

% Stop audio schedule
Datapixx('StopAudioSchedule');
Datapixx('RegWrRd');

% Calculate actual duration
actual_duration = expdata.end_time - expdata.start_time;
expdata.actual_duration = actual_duration;

fprintf('White noise ended at %.3f seconds\n', actual_duration);
fprintf('Actual playback duration: %.2f minutes\n', actual_duration/60);

%% Clean up and save data
Datapixx('Close');
fprintf('\n=== White Noise Session Completed ===\n');

%% Display session summary
fprintf('\n=== SESSION SUMMARY ===\n');
fprintf('Participant: %s\n', participant_id);
fprintf('Start trigger: TTL bit 4 (value 4)\n');
fprintf('End trigger: TTL bit 8 (value 8)\n');
fprintf('Pause trigger: TTL bit 2 (value 2)\n');
fprintf('Resume trigger: TTL bit 9 (value 9)\n');
fprintf('Quit trigger: TTL bit 10 (value 10)\n');
fprintf('Planned duration: %.2f minutes\n', duration/60);
fprintf('Actual duration: %.2f minutes\n', actual_duration/60);

if ~isnan(expdata.paused_times)
    fprintf('Number of pauses: %d\n', length(expdata.paused_times));
    fprintf('Pause times: %s\n', mat2str(expdata.paused_times, 2));
    fprintf('Resume times: %s\n', mat2str(expdata.resume_times, 2));
end
fprintf('Sample rate: %d Hz\n', sample_freq);
fprintf('Amplitude: %.1f (%.1f%%)\n', amplitude, amplitude*100);
fprintf('Low-pass cutoff: %d Hz\n', cutoff_freq);
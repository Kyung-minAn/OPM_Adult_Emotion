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

%create the sound
sample_freq = 48000;
NormalStl(1,:) =  MakeBeep(1000, 0.1, sample_freq); % normal beep
NormalStl(2,:) = NormalStl(1,:);
nChannels_N = size(NormalStl, 1);
nTotalFrames_N = size(NormalStl, 2);

%trial
Ntrial = 120; %10; 
trlstart=zeros(length(Ntrial),1);
jitter=zeros(length(Ntrial),1);
endtrl=zeros(length(Ntrial),1);

for i=1:Ntrial
        expdata.trlnum = i;
        expdata.trlstart=GetSecs;
        
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
        
        jitter=1.8 + 0.5*rand;
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
fileName = sprintf('%s_AER_%s.mat', participant, date_str);
savePath = fullfile(saveDir, fileName);

save(savePath, 'AER_expdata_Table', '-v7.3');
fprintf('\nAER completed\n\n');


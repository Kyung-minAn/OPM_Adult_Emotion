"""
===============================================
Preprocessing OPM-MEG Task Data - Filter, Annotate and ICA (specifically designed for FieldLine data)

Sections 1-13 = Identify bad channels, annotate breaks and notch/bandpass filtering data.
                Save as '_ann.fif' file and save MNE reports.
    
    -1 = Import relevant modules
    -2 = Define custom helper functions
        ## Function to match sensor numbers to FieldLine formatted channel names
        ## Function to only keep bad data segments lasting the specified minimum artefact duration
        ## Function to check if new annotated segments overlap with any existing annotations
    -3 = Define subject and task information, then load in BIDs formatted data
        ## Change to your subject and task information, your file paths and directories
        ## Create derivatives path with task subfolder, where you want preprocessed files outputs saved
        ## Read in raw BIDs formatted data
        ## Events information
    -4 = Bad and noisy channels from acqusition notes to exclude
    -5 = MNE report
        ## Set-up MNE report files directories and file names
        ## Open report generated from BIDs conversion script
    -6 = PSD of pre-filtered raw data
    -7 = Plot histogram of sensor power to identify sensors/channels with high noise levels
        ## Additional check to identify sensors with high noise that have higher average power vs distribution of sensors
        ## High power sensors above histogram threshold
    -8 = Identify and mark bad channels/sensors
        ## Find FieldLine formatted sensor/channel names from list of sensors to exclude from dataset
        ## Option to manually add more channels to drop from dataset
        ## Removing bad sensors/channels from dataset (manual lists & sensors exceeding high power threshold from histogram check)
    -9 = Plot power amplitude of data in pT without bad channels, prior to notch and bandpass filtering
    -10 = Mark 'bad' annotations in dataset to ignore block breaks and pre/post experiment parts in recording
        ## Change to your event/trigger markers for this to work with your data
        ## Pre and post experiment annotations
        ## Block break annotations
        ## For each subsequent block, find period from last trial end to first trial end of next block
        ## Combine and apply annotations for pre-exp, post-exp, and block breaks
    -11 = Additional bad channel check in task data (excluding block breaks or pre/post experiment segments)
        ## Define thresholds
        ## Extract task data 'good segments' excluding block breaks ore pre/post experiment segments
        ## Concatenate all good segments for bad channel detection
        ## Identify bad channels exceeding threshold percentage and threshold above median stdev
        ## Plot signal from all sensors/channels
        ## Plot of bad channels with persistent noise overlayed on top of good channels to be removed from dataset
        ## Remove additional bad channels identified from the dataset
    -12 = Apply notch and bandpass filters to data after bad channels have been removed
        ## Notch and bandpass filter
        ## Amplitude plot of filtered data
        ## PSD plot of filtered data
    -13 = Save filtered and annotated _ann.fif data and save MNE reports (hdf5 & html)

Sections 14-19 = Applying ICA correction, identifying any other bad segments.
                 Save as '_ica.fif' file and MNE reports.

    -14 = ICA for artifact removal (fastICA with 20 components), plot components and timecourses
        ## Make a copy of data filtered between 1-30Hz - which works better for ICA
        ## Set up fastICA with 20 components and fit ICA to data
        ## Interactive plots for ICA inspection
    -15 = Optional section if you want to see all ICA component breakdowns and sensor layout
    -16 = Manual selection of bad ICA components
        ## Manually list bad components to exclude after visual inspection
        ## Plot overlay of bad ICA components with data
    -17 = Apply ICA and visualising PSD and signal amplitude afterwards
        ## Applying ICA to data and excluding bad components
        ## Plot PSDs to inspect signal noise post-ICA at low and high frequency ranges
        ## Plot signal amplitude to inspect noise levels after ICA
    -18 = Additional method to identify time points with extreme values (transient noise) to exclude when epoching data
        ## Find time points exceeding the extreme threshold set (excluding already identified bad channels)
        ## Process new bad/extreme data segments that do not overlap with existing ones
        ## Add new non-overlapping extreme data segments as 'bad' annotations, so epochs containing bad segments are dropped from analysis
        ## Plot these identified extreme time periods of transient noise
        ## Final plot to visually check data signal per channel after preprocessing
    -19 = Save data with ICA applied as _ica.fif and save MNE reports (hdf5 & html)

@author: Alice Waitt (aewaitt), 2025
==============================================  

"""
#%% 1. Import relevant modules
import os
import os.path as op
import numpy as np
import mne
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg for interactive plots
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
plt.ion()
from mne_bids import (BIDSPath, read_raw_bids)

#%% 2. Define custom helper functions
## Function to match sensor numbers to FieldLine formatted channel names
def find_exact_sensor_channels(ch_names, sensor_numbers):
    """Find FieldLine formatted channel names containing specified sensor numbers"""
    matches = []
    for ch in ch_names:
        if '_bz-s' in ch:
            try:
                num = int(ch.split('s')[-1])
                if num in sensor_numbers:
                    matches.append(ch)
            except ValueError:
                continue
    return matches

## Function to only keep bad data segments lasting the specified minimum artefact duration
def find_segments(mask, min_duration=30): # Duration of bad segment in milliseconds
    """Goes through true segements of boolean mask, corresponding to bad data segments exceeding extreme value thresholds. 
    Only keeps bad segments lasting for specificed minimum duration.
    Returns final list of bad segment in tuples format (start_idx, end_idx)."""
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, val in enumerate(mask):
        if val and not in_segment:
            # Start of a new segment
            in_segment = True
            start_idx = i
        elif not val and in_segment:
            # End of segment
            if i - start_idx >= min_duration:  # Only keep segments of minimum length
                segments.append((start_idx, i))
            in_segment = False
    
    # Handle case where segment extends to the end
    if in_segment and len(mask) - start_idx >= min_duration:
        segments.append((start_idx, len(mask)))
        
    return segments

## Function to check if new annotated segments overlap with any existing annotations
def overlaps_with_existing(new_start, new_end, existing_segments):
    for start, end in existing_segments:
        # Check for overlap
        if (new_start <= end and new_end >= start):
            return True
    return False

#%% 3. Define subject and task information, then load in BIDs formatted data
## Change to your subject and task information, your file paths and directories
subject = '01' #Change subject number here
session = '01' #Change session number if necessary
task = 'PassiveEmoVoice' #Name of your task here
run = '01'
multiruns = 0 #If using multiple runs for subject or using run 02 (or greater) make this 1, otherwise 0
meg_suffix = 'meg'
meg_extension = '.fif'
events_suffix = 'events'
events_extension = '.tsv'
bids_folder = 'Z:\BIDS' #Change to where your BIDS directory is
ann_suffix = 'ann'
ica_suffix='ica'
notch_width = 1.5
l_freq_hist = 60 # Defining filtered freqs for Section 7 histogram check
h_freq_hist = 80
l_freq = 1 # 1Hz minimum high pass filter for system, any lower and there are too many artefacts remaining in data
h_freq= 80
l_h_suffix = f"{l_freq}-{h_freq}Hz_"
hfc_status = "HFC not applied" #If operating in single axis mode, advisable not to use HFC
amp_scale = 1e12  #Converting to pico Tesla(pT)

bids_path = BIDSPath(
    subject=subject, 
    session=session, 
    task=task, 
    run=run, 
    suffix=meg_suffix,
    extension=meg_extension, 
    root=bids_folder
)

## Create derivatives path with task subfolder, where you want preprocessed files outputs saved
deriv_root = op.join(bids_folder, 'derivatives/preprocessing')
deriv_path = BIDSPath(subject=subject, session=session, datatype='meg',
            task=task, run=run, suffix=meg_suffix, root=deriv_root).mkdir()
task_subfolder = op.join(deriv_path.directory, task) # Creating task subfolder path
if not op.exists(task_subfolder):
    os.makedirs(task_subfolder)

deriv_fname_fif = bids_path.basename.replace(meg_suffix, l_h_suffix + ann_suffix)
deriv_fname_fif_1 = op.join(task_subfolder, deriv_fname_fif)
deriv_fname_fif_ica = deriv_fname_fif.replace(l_h_suffix+ann_suffix, l_h_suffix+ica_suffix)
deriv_fname_fif_1_ica = op.join(task_subfolder, deriv_fname_fif_ica)
deriv_fname_csv_1 = deriv_fname_fif_1.replace('fif', 'csv') # CSV output filename

print(bids_path)
print(deriv_fname_fif)
print(deriv_fname_fif_1)
print(deriv_fname_fif_ica)
print(deriv_fname_fif_1_ica)

## Read in raw BIDs formatted data
raw = read_raw_bids(bids_path=bids_path, verbose=False,extra_params={'preload':True})

## Events information
# Should use 'mne.events_from_annotations' instead of 'mne.find_events'.
# This finds events tsv/json files from bids conversion, which have handled the downsampling with more accuracy.
events, event_id = mne.events_from_annotations(raw) 

#%% 4. Bad and noisy channels from acqusition notes to exclude
badchanlist = [] # Channels that were noted as being turned off during acquisition
noisychans = []  # Channels that were on but looked noisy during data acqusition
excludechans = badchanlist + noisychans # Ensures these channels are excluded from data analysis

#%% 5. MNE report
## Set-up MNE report files directories and file names
# Finds MNE report files generated from BIDs converstion script to add to.
# html and hdf5 formats, html for viewing report and hdf5 to add to report later.
report_root = op.join(bids_folder, 'derivatives/mne-reports')  # RDS folder for reports
report_folder = op.join(report_root , 'sub-' + subject, 'task-' + task)

if multiruns==1:# Adds run number if storing multiple task run reports
    report_fname = op.join(report_folder, f'report_sub-{subject}_{task}_run-{run}_raw.hdf5')
    report_ann_fname = op.join(report_folder, f'report_sub-{subject}_{task}_{l_h_suffix}{ann_suffix}_run-{run}.hdf5') 
    html_report_ann_fname = op.join(report_folder, f'report_ann_sub-{subject}_{task}_{l_h_suffix}{ann_suffix}_run-{run}.html')
    report_ica_fname = op.join(report_folder, f'report_sub-{subject}_{task}_{l_h_suffix}{ica_suffix}_run-{run}.hdf5')
    html_report_ica_fname = op.join(report_folder, f'report_sub-{subject}_{task}_{l_h_suffix}{ica_suffix}_run-{run}.html')
else:   
    report_fname = op.join(report_folder, f'report_sub-{subject}_{task}_raw.hdf5')
    report_ann_fname = op.join(report_folder, f'report_sub-{subject}_{task}_{l_h_suffix}{ann_suffix}.hdf5') 
    html_report_ann_fname = op.join(report_folder, f'report_ann_sub-{subject}_{task}_{l_h_suffix}{ann_suffix}.html')
    report_ica_fname = op.join(report_folder, f'report_sub-{subject}_{task}_{l_h_suffix}{ica_suffix}.hdf5')
    html_report_ica_fname = op.join(report_folder, f'report_sub-{subject}_{task}_{l_h_suffix}{ica_suffix}.html')

## Open report generated from BIDs conversion script
report = mne.open_report(report_fname)

#%% 6. PSD of pre-filtered raw data
n_fft = 2000
raw_PSD = raw.compute_psd(method="welch", fmin=0.1, fmax=120, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2))
psds = raw_PSD.get_data() # Units are in T^2/Hz
freqs = raw_PSD.freqs 
psd_db = [10 * np.log10(psd * 1e30) for psd in psds] # Ref power: 1fT^2=1e-30T^2
average_power = np.mean(psd_db, axis=1)
fig_psd_raw = raw_PSD.plot(show=True)
report.add_figure(fig=fig_psd_raw, 
                  title='Raw PSD before filtering',
                  caption='Power spectral density of raw MEG data')

#%% 7. Plot histogram of sensor power to identify sensors/channels with high noise levels
## Additional check to identify sensors with high noise that have higher average power vs distribution of sensors
rawfilt=raw.copy()
iir_params = dict(order=5, ftype="butter")
rawfilt=rawfilt.filter(l_freq=l_freq_hist, h_freq=h_freq_hist, method='iir', iir_params= iir_params) # Filter data in med-high freq range
raw_PSD_flt = rawfilt.compute_psd(method="welch", fmin=55, fmax=85, picks="mag", n_fft=n_fft, n_overlap=int(n_fft/2))
psds_flt = raw_PSD_flt.get_data() # Units are in T^2/Hz
freqs_flt = raw_PSD_flt.freqs 
psd_db_flt = [10 * np.log10(psd * 1e30) for psd in psds] # Ref power: 1fT^2=1e-30T^2
average_power_flt = np.mean(psd_db_flt, axis=1)
figfltpsd=raw_PSD_flt.plot(show=True)

report.add_figure(fig=figfltpsd,
                  title='Raw PSD for high freqs, data filtered 60-80',
                  caption='Power spectral density for high frequency range')

fig_hist_sensors=plt.figure(figsize=(10, 6))
plt.hist(average_power_flt, bins=100, color='skyblue', edgecolor='black')
plt.xlabel('Average Power (dB) [fT^2/Hz]')
plt.ylabel('Number of Sensors')
plt.title('Histogram of Average Power Across Sensors')
plt.show()
report.add_figure(fig=fig_hist_sensors,
                  title='Sensor power histogram (raw)',
                  caption='Distribution of power across all MEG sensors')

## High power sensors above histogram threshold
sensor_names = raw_PSD.ch_names
avg_threshold = 40 #Adjusted between 38-44 depending on subject histogram distribution
sensors_above_threshold = [sensor_names[i] for i, avg in enumerate(average_power) if avg > avg_threshold]
print(f"Sensors above threshold: {sensors_above_threshold}")

#%% 8. Identify and mark bad channels/sensors
## Find FieldLine formatted sensor/channel names from list of sensors to exclude from dataset
bad_channels = find_exact_sensor_channels(raw.info['ch_names'], excludechans)

## Option to manually add more channels to drop from dataset
# e.g. UoB-specific data uses s16_bz as the EOG channel, which has a different channel name format
bad_channels = bad_channels + [] 
print(f"Bad channels from list: {bad_channels}")

## Removing bad sensors/channels from dataset (manual lists & sensors exceeding high power threshold from histogram check)
allbads = bad_channels + sensors_above_threshold
allbads = list(dict.fromkeys(allbads))  # Remove duplicates in bad channel list
raw.drop_channels(allbads)
print(f"All bad channels: {allbads}")

#%% 9. Plot power amplitude of data in pT without bad channels, prior to notch and bandpass filtering
picks = mne.pick_types(raw.info, meg=True, exclude='bads')
data_ds, time_ds = raw[picks, :]
data_ds = data_ds * amp_scale #Scaled to pT in plot
fig_raw_amp, ax = plt.subplots(figsize=(12, 6), layout="constrained")
plot_kwargs = dict(lw=1, alpha=1)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True)
set_kwargs = dict(
    ylim=(-2000, 2000), xlim=time_ds[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)"
)
ax.set(title="Before Filters", **set_kwargs)
plt.show()
report.add_figure(fig=fig_raw_amp,
                  title='Signal amplitude before filtering',
                  caption=f'Unfiltered MEG data for all channels, except noisy channels {allbads}')

#%% 10. Mark 'bad' annotations in dataset to ignore block breaks and pre/post experiment parts in recording
## Change to your event/trigger markers for this to work with your data
block_start = event_id[np.str_('BlockStart')] # Change to the name of your block start trigger
trl_end = event_id[np.str_('TwoSecPostVoice')] # Change to the name trigger marking the end of a trial
pre_exp_mark = 0.5 # Set time before the first trigger in data to include
post_exp_mark = 2.3 # Set time after the last trigger in the data to include
last_trl_buffer = 1 # Change how long (s) after last trial in block to mark when block break starts
pre_trl_buffer = 0.5 # Change how much (s) before the start of the next block to mark end of block break

## Pre and post experiment annotations
first_stim = events[0,0]/raw.info["sfreq"] # First trigger in data for start of experiment
last_stim = events[-1,0]/raw.info["sfreq"] # Last trigger in data for end of experiment
recording_start = raw.first_time # First time point
recording_end = raw.first_time + raw.times[-1]  # Last time point
epsilon = 1e-6  # Add tiny buffer (1 microsecond) to avoid floating-point precision issues
recording_end -= epsilon  # Slightly reduce the end time
start_time = first_stim - pre_exp_mark  # Time pre-experiment before first trigger to include
last_time = last_stim + post_exp_mark # Time after the last trigger in the data to include

# Pre-experiment annotation (from start of recording to just before first event/trigger)
onset_pre = recording_start
duration_pre = start_time - recording_start

# Post-experiment annotation (from after last stimulus/event to end of recording)
onset_post = recording_start + last_time
duration_post = max(0, recording_end - onset_post)  # Ensure not exceeding recording length or non-negative duration

## Block break annotations
# Find block break periods
block_times = mne.pick_events(events, include=block_start)
trl_end_times = mne.pick_events(events, include=trl_end)
block_start_timestamps = block_times[:, 0]
trial_end_timestamps = trl_end_times[:, 0]
block_break_onsets = [] # Initialise block break onset list
block_break_durations = [] # Initialise block break duration list

# Handle first block specially
if len(block_start_timestamps) > 0:
    mask_first_post_trl = trial_end_timestamps > block_start_timestamps[0] # Find first trial end marker after the first block start
    if np.any(mask_first_post_trl):
        first_trlend_time = trial_end_timestamps[np.where(mask_first_post_trl)[0][0]]
        first_block_start = recording_start # From recording start
        first_block_end = (first_trlend_time / raw.info["sfreq"]) - pre_exp_mark
        
        # Add first block break (if it doesn't overlap with pre-experiment annotation)
        if first_block_end > onset_pre + duration_pre:
            first_block_start = max(first_block_start, onset_pre + duration_pre)
            block_break_onsets.append(first_block_start)
            block_break_durations.append(first_block_end - first_block_start)

## For each subsequent block, find period from last trial end to first trial end of next block
# Note, depending on your data triggers you may want to change the markers for how you are
# working out the start and end of the blocks
for i in range(1, len(block_start_timestamps)):
   
    mask_end = trial_end_timestamps < block_start_timestamps[i] # Find the last trial end before this block start
    if np.any(mask_end):
        last_trial_end_time = trial_end_timestamps[np.where(mask_end)[0][-1]]
        
        # Find the first trial end time after this block start
        mask_post_trial = trial_end_timestamps > block_start_timestamps[i]
        if np.any(mask_post_trial):
            first_post_trial_time = trial_end_timestamps[np.where(mask_post_trial)[0][0]]
            
            # Convert to seconds from recording start
            block_break_start = (last_trial_end_time / raw.info["sfreq"]) + last_trl_buffer # How long (s) after last trial in block to mark when block break starts
            block_break_end = (first_post_trial_time / raw.info["sfreq"]) - pre_trl_buffer # How much (s) before the start of the next block to mark end of block break
            
            # Ensure positive duration
            if block_break_end > block_break_start:
                block_break_onsets.append(block_break_start)
                block_break_durations.append(block_break_end - block_break_start)

## Combine and apply annotations for pre-exp, post-exp, and block breaks
# Combine 'bad' annotations
onsets = [onset_pre] + block_break_onsets + [onset_post]
durations = [duration_pre] + block_break_durations + [duration_post]
descriptions = ["bad_pre_exp"] + ["bad_block_break"] * len(block_break_onsets) + ["bad_post_exp"]

# Apply 'bad' annotations to dataset
all_annots = mne.Annotations(
    onset=onsets,
    duration=durations,
    description=descriptions,
    orig_time=raw.info["meas_date"],
)
raw.set_annotations(raw.annotations + all_annots)  # add to any existing annotations

# Print out timings to check
print(f"Recording duration: {raw.duration} s")
print(f"Recording time range: {recording_start} to {recording_end} s")
print(f"Pre-exp annotation: {onset_pre} to {onset_pre + duration_pre} s")
for i, (onset, duration) in enumerate(zip(block_break_onsets, block_break_durations)):
    print(f"Block break {i+1}: {onset} to {onset + duration} s")
print(f"Post-exp annotation: {onset_post} to {onset_post + duration_post} s")

#%% 11. Additional bad channel check in task data (excluding block breaks or pre/post experiment segments)
## Define thresholds
channel_threshold = 5   # For persistent bad channels (pT)
percent_threshold = 5.0 # Percentage of data exceeding threshold to mark as bad
stddev = 2 # Threshold above median stdev to mark as bad

## Extract task data 'good segments' excluding block breaks ore pre/post experiment segments
# Find all existing bad segments from annotations
raw_check = raw.copy()
bad_segments = []
for annot in raw.annotations:
    if annot['description'].startswith('bad'):
        bad_segments.append((annot['onset'], annot['onset'] + annot['duration']))

# Sort and merge overlapping bad segments
bad_segments.sort()
merged_bad = []
if bad_segments:
    current_start, current_end = bad_segments[0]
    for start, end in bad_segments[1:]:
        if start <= current_end:  # Overlapping segments
            current_end = max(current_end, end)
        else:  # Non-overlapping segment
            merged_bad.append((current_start, current_end))
            current_start, current_end = start, end
    merged_bad.append((current_start, current_end))

# Find good segments between the bad segments
good_segments = []
last_end = raw.first_time
for start, end in merged_bad:
    if start > last_end:
        good_segments.append((last_end, start))
    last_end = max(last_end, end)

# Add segment from last bad segment to end of recording if needed
if last_end < recording_end:
    good_segments.append((last_end, recording_end))

## Concatenate all good segments for bad channel detection
if good_segments:
    raw_segments = [] # Define variable to store the good segments
    
    # Crop and collect each good segment
    for i, (tmin, tmax) in enumerate(good_segments):
        if tmax - tmin > 1.0:  # Only keep segments longer than 1 second
            print(f"Good segment {i+1}: {tmin:.2f} to {tmax:.2f} s (duration: {tmax-tmin:.2f}s)")
            raw_segments.append(raw.copy().crop(tmin=tmin, tmax=tmax))
    if raw_segments:
        raw_check = mne.concatenate_raws(raw_segments) # Concatenate all good segments
        print(f"Created concatenated data from {len(raw_segments)} good segments, total duration: {raw_check.times[-1]:.2f}s")
    else:
        print("No good segments longer than 1 second found!")
else:
    print("No good segments found!")

# Plot trace of concatenated good data for inspection
raw_check.plot()
raw_check.notch_filter(np.arange(50, 251, 50), notch_widths=notch_width)
iir_params = dict(order=5, ftype="butter")
raw_checkfilt = raw_check.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params= iir_params) 
raw_checkfilt.plot()

## Identify bad channels exceeding threshold percentage and threshold above median stdev
# Get all MEG data
picks = mne.pick_types(raw_checkfilt.info, meg=True, exclude='bads')
data, times = raw_checkfilt[picks, :]
data = data * amp_scale # Convert to pT
data_normalized = data - np.mean(data, axis=1)[:, np.newaxis] # Normalise each channel

# Identify bad channels based on percentage of points exceeding threshold
exceed_points = np.sum((data_normalized > channel_threshold) | 
                       (data_normalized < -channel_threshold), axis=1)
total_points = data_normalized.shape[1]
percentage_exceed = (exceed_points / total_points) * 100
bad_channel_indices = np.where(percentage_exceed > percent_threshold)[0]
bad_channel_names = [raw_checkfilt.ch_names[picks[i]] for i in bad_channel_indices]
print(f"Found {len(bad_channel_names)} sensors exceeding ±{channel_threshold} pT for >{percent_threshold}% of time:")
print(bad_channel_names)

# Channels with excessive noise will have higher standard deviation
std_devs = np.std(data, axis=1)
std_threshold = np.median(std_devs) * stddev # Threshold above median stdev to mark as bad
bad_std_indices = np.where(std_devs > std_threshold)[0]
bad_std_channels = [raw_checkfilt.ch_names[picks[i]] for i in bad_std_indices]
print(f"Found {len(bad_std_channels)} noisy channels based on standard deviation:")
print(bad_std_channels)

# Print channels with highest percentage of threshold violations
ch_percentages = [(raw_checkfilt.ch_names[picks[i]], percentage_exceed[i]) for i in range(len(picks))]
ch_percentages.sort(key=lambda x: x[1], reverse=True)
print("\nTop channels with highest percentage of threshold violations:")
for ch_name, percent in ch_percentages[:10]:
    print(f"{ch_name}: {percent:.2f}%")

# Generate full list of bad channels after converting indices to sets to avoid duplicates
bad_indices_set = set(bad_channel_indices)
bad_std_indices_set = set(bad_std_indices)
combined_indices_set = bad_indices_set.union(bad_std_indices_set) # Combine the sets
bad_channel_indices = sorted(list(combined_indices_set)) # Convert back to a list and sort for consistency
bad_channel_names = [raw_checkfilt.ch_names[picks[i]] for i in bad_channel_indices] # Rebuild channel names list based on combined indices
print(f"Found {len(bad_channel_names)} total bad channels after combining both methods:")
print(bad_channel_names)

## Plot signal from all sensors/channels
fig_all_channels, ax_all = plt.subplots(figsize=(12, 6), layout="constrained")
plot_kwargs = dict(lw=0.5, alpha=0.3)
ax_all.plot(times, data_normalized.T, **plot_kwargs)
y_limit = channel_threshold * 2 # Set plot limits based on the data
ax_all.set(
    title=f"All channels pre-removal (break periods removed)",
    ylim=(-y_limit, y_limit), 
    xlim=(times[0], times[-1]), 
    xlabel="Time (s)", 
    ylabel="Amplitude (pT)"
)
ax_all.grid(True)
plt.show()

report.add_figure(fig=fig_all_channels,
                    title=f'All channels plot pre-removal',
                    caption=f'No channels highlighted')

## Plot of bad channels with persistent noise overlayed on top of good channels to be removed from dataset
fig_bad_channels, ax_bad = plt.subplots(figsize=(12, 6), layout="constrained")

# First plot good channels in light blue
good_channel_mask = np.ones(len(picks), dtype=bool)
good_channel_mask[bad_channel_indices] = False
good_indices = np.where(good_channel_mask)[0]
ax_bad.plot(times, data_normalized[good_indices].T, color='lightblue', 
            lw=0.5, alpha=0.2, zorder=1)

# Plot each bad channel in a different color with higher opacity
colors = plt.cm.rainbow(np.linspace(0, 1, len(bad_channel_indices)))
for idx, i in enumerate(bad_channel_indices):
    ax_bad.plot(times, data_normalized[i], color=colors[idx], 
                linewidth=1.0, alpha=0.7, zorder=2,
                label=f"{raw_checkfilt.ch_names[picks[i]]}")

# Set plot limits
if len(bad_channel_indices) > 0:
    max_val = np.max(np.abs(data_normalized[bad_channel_indices])) 
    y_limit = max(channel_threshold * 1.5, min(max_val * 1.2, 10))
else:
    y_limit = channel_threshold * 2
    
ax_bad.set(
    title=f"Persistently noisy channels exceeding ±{channel_threshold} pT for >{percent_threshold}% of time or {stddev} Stdevs (break periods removed)",
    ylim=(-y_limit, y_limit), 
    xlim=(times[0], times[-1]), 
    xlabel="Time (s)", 
    ylabel="Amplitude (pT)"
)

# Add legend for bad channels - show only first few in legend to avoid overcrowding
if len(bad_channel_indices) > 15:
    handles, labels = ax_bad.get_legend_handles_labels()
    ax_bad.legend(handles[:10], labels[:10], loc='upper right', 
                 title=f"Bad channels (showing 10/{len(bad_channel_indices)})")
    plt.figtext(0.02, 0.02, f"Total bad channels: {len(bad_channel_indices)}", 
                ha="left", fontsize=9)
else:
    ax_bad.legend(loc='upper right', title="Bad channels")

ax_bad.grid(True)
plt.show()

## Remove additional bad channels identified from the dataset
removedchans = allbads + bad_channel_names
print(removedchans)
report.add_figure(fig=fig_bad_channels,
                    title=f'Persistently noisy channels (±{channel_threshold} pT threshold)',
                    caption=f'Bad channels highlighted in red ({len(bad_channel_names)} channels) will be removed')
raw.drop_channels(bad_channel_names) 

#%% 12. Apply notch and bandpass filters to data after bad channels have been removed
## Notch and bandpass filter
print("Applying filters...")
raw.notch_filter(np.arange(50, 251, 50), notch_widths=notch_width) # Notch filter
raw_proc = raw.copy()
raw_proc.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params= iir_params) # Bandpass filter

## Amplitude plot of filtered data
picks = mne.pick_types(raw_proc.info, meg=True, exclude='bads')
data_ds, _ = raw_proc[picks, :]
data_ds = data_ds * amp_scale
fig_filt_amp, ax = plt.subplots(figsize=(12,6),layout="constrained")
plot_kwargs = dict(lw=1, alpha=0.5)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True)
set_kwargs = dict(
    ylim=(-10, 10), xlim=time_ds[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)"
)
ax.set(title=f"After notch (width={notch_width}) and bandpass({l_freq}-{h_freq}Hz) - {hfc_status}", **set_kwargs)
plt.show()
report.add_figure(fig=fig_filt_amp,
                  title=f'Signal amplitude after notch (width={notch_width} and bandpass filtering ({l_freq}-{h_freq}Hz) - {hfc_status}',
                  caption=f'Filtered MEG data for all clean channels, excluding: {removedchans}')

## PSD plot of filtered data
n_fft = 2000
psdcopy = raw_proc.copy()
raw_PSD_proc40 = psdcopy.compute_psd(method="welch", fmin=0.1, fmax=40, picks="mag", exclude='bads')
fig_psd_filtered40 = raw_PSD_proc40.plot()
report.add_figure(fig=fig_psd_filtered40,
                  title='PSD of low freqs after notch and bandpass filtering',
                  caption='Power spectral density after applying notch and bandpass filters')

raw_PSD_proc120 = psdcopy.compute_psd(method="welch", fmin=0.1, fmax=120, picks="mag", exclude='bads')
fig_psd_filtered120 = raw_PSD_proc120.plot()
report.add_figure(fig=fig_psd_filtered120,
                  title='PSD of full freq range after notch and bandpass filtering',
                  caption='Power spectral density after applying notch and bandpass filters')

#%% 13. Save filtered and annotated _ann.fif data and save MNE reports (hdf5 & html)
raw_proc.save(deriv_fname_fif_1, overwrite=True)
raw_proc.annotations.save(deriv_fname_csv_1, overwrite=True) # Save CSV file of annotations/events in data

report.save(report_ann_fname, overwrite=True)
report.save(html_report_ann_fname, overwrite=True, open_browser=True)  # Opens after save to view report

#%% 14. ICA for artifact removal (fastICA with 20 components), plot components and timecourses
## Make a copy of data filtered between 1-30Hz - which works better for ICA
raw_resmpl = raw_proc.copy()
raw_resmpl.filter(1, 30, method='iir', iir_params= iir_params)

## Set up fastICA with 20 components and fit ICA to data
# These settings work well with OPM data
print("Running ICA...")
ica = ICA(
    n_components= 20,
    max_iter='auto',
    random_state=96,
    method='fastica'
)

print("Fitting ICA...")
picks = mne.pick_types(raw_proc.info, meg=True, exclude='bads')
ica.fit(raw_resmpl, picks=picks, reject_by_annotation=True) # Fit ICA to data
print("ICA fit complete")

## Interactive plots for ICA inspection
print("Displaying ICA components for visual inspection...")
fig_ica_components = ica.plot_components()
icasourcesfig=ica.plot_sources(raw_resmpl)

report.add_figure(fig=fig_ica_components,
                  title='ICA components',
                  caption='Spatial patterns of identified ICA components')
report.add_figure(fig=icasourcesfig,
                  title='ICA components timecourses',
                  caption='Timecourses of identified ICA components')

#%% 15. Optional section if you want to see all ICA component breakdowns and sensor layout
for ica_comp in list(range(0,20)):
    fig_ica_comp = ica.plot_properties(raw_proc, picks=ica_comp, show=False)

#Plot sensor layout to see if any components match specific sensor locations
mne.viz.plot_sensors(raw_proc.info, show_names = True)

#%% 16. Manual selection of bad ICA components
## Manually list bad components to exclude after visual inspection
manual_ica_picks = []

## Plot overlay of bad ICA components with data
fig_overlay=ica.plot_overlay(raw_resmpl, exclude=manual_ica_picks, picks="mag")
report.add_figure(fig=fig_overlay,
                  title='ICA components overlay with data',
                  caption='comparing before and after removal of ICA components')

bad_ica_components = list(set(manual_ica_picks))
print(f"Final bad ICA components: {bad_ica_components}")

#%% 17. Apply ICA and visualising PSD and signal amplitude afterwards
## Applying ICA to data and excluding bad components
print("Applying ICA and final filtering...")
ica.exclude = bad_ica_components

for component in ica.exclude:
    fig_comp_props = ica.plot_properties(raw_proc, picks=component, show=False)
    report.add_figure(fig=fig_comp_props,
                     title=f'Properties of ICA component {component}',
                     caption=f'Component {component} identified as artifact and removed')

raw_cleaned = raw_proc.copy()
ica.apply(raw_cleaned)

## Plot PSDs to inspect signal noise post-ICA at low and high frequency ranges
raw_cleancopy = raw_cleaned.copy()
# Low frequency visualisation between 1-40Hz
post_ICA_PSD= raw_cleancopy.compute_psd(method="welch", fmin=0.1, fmax=40, picks="mag", exclude='bads')
post_ICA_PSD_fig = post_ICA_PSD.plot()
report.add_figure(fig=post_ICA_PSD_fig,
                  title='PSD after notch and bandpass filtering',
                  caption='Power spectral density after applying notch and bandpass filters')
# Broad frequency visualisation between 1-120Hz
post_ICA_PSD_120 = raw_cleancopy.compute_psd(method="welch", fmin=0.1, fmax=120, picks="mag", exclude='bads')
post_ICA_PSD_fig_120 = post_ICA_PSD_120.plot()
report.add_figure(fig=post_ICA_PSD_fig_120,
                  title='PSD after notch and bandpass filtering 1-120',
                  caption='Power spectral density after applying notch and bandpass filters')

## Plot signal amplitude to inspect noise levels after ICA
# Set-up plot
picks = mne.pick_types(raw_cleaned.info, meg=True, exclude='bads')
data, times = raw_cleaned[picks, :]
data = data * amp_scale  # Scale to pT
data_normalized = data - np.mean(data, axis=1)[:, np.newaxis] # Normalise each channel
fig_raw_cleaned, ax = plt.subplots(figsize=(12, 6), layout="constrained")
plot_kwargs = dict(lw=1, alpha=1)
set_kwargs = dict(
    ylim=(-10, 10), xlim=times[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)"
)

# Plot data
for start, end in merged_bad: # Merged_bad = all 'bad' periods identified previously in the script
    ax.axvspan(start, end, color='lightblue', alpha=0.5) # Highlight existing annotations (e.g. breaks) in light blue

ax.plot(times, data_normalized.T, **plot_kwargs)
ax.grid(True)
ax.set(title="After Filters and ICA (with existing annotations)", **set_kwargs)
plt.show()
report.add_figure(fig=fig_raw_cleaned,
                 title='Signal amplitude after filtering',
                 caption=f'Filtered MEG data for all channels (except {removedchans}), with existing annotations in light blue')

#%% 18. Additional method to identify time points with extreme values (transient noise) to exclude
## Find time points exceeding the extreme threshold set (excluding already identified bad channels)
extreme_threshold = 8   # Can change if there are too many/too few segments
extreme_mask = np.any((data_normalized  > extreme_threshold) | 
                         (data_normalized  < -extreme_threshold), axis=0)

# Find noisy data segments with extreme values
extreme_segments = find_segments(extreme_mask, min_duration=30) # 30ms min duration of bad segment
print(f"Identified {len(extreme_segments)} segments with extreme values")

## Process new bad/extreme data segments that do not overlap with existing ones
onsets = []
durations = []
descriptions = []
existing_segs_bad = merged_bad.copy()
new_segs_bad = []

for start_idx, end_idx in extreme_segments:
    onset = times[start_idx]
    duration = times[end_idx-1] - onset if end_idx < len(times) else times[-1] - onset
    
    # Add a small buffer around the bad segment
    buffer = 0.05  # 50ms buffer
    onset_buffered = max(0, onset - buffer)
    duration_buffered = duration + 2 * buffer
    end_buffered = onset_buffered + duration_buffered
    
    # Only add if this segment doesn't overlap with existing annotations
    if not overlaps_with_existing(onset_buffered, end_buffered, existing_segs_bad):
        onsets.append(onset_buffered)
        durations.append(duration_buffered)
        descriptions.append('BAD_transient')
        
        # Add this segment to existing_segments to prevent future overlaps within this loop
        existing_segs_bad.append((onset_buffered, end_buffered))
        new_segs_bad.append((onset_buffered, end_buffered))

## Add new non-overlapping extreme data segments as 'bad' annotations, so epochs containing bad segments are dropped from analysis
# Create new Annotations object with non-overlapping bad data segments
if onsets:
    bad_annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=raw_cleaned.info["meas_date"]
    )
    
    # Add new annotations to the existing ones
    raw_cleaned.set_annotations(raw_cleaned.annotations + bad_annotations) 
    print(f"Added {len(onsets)} new non-overlapping BAD_transient annotations")
else:
    print("No new non-overlapping segments to annotate")

## Plot these identified extreme time periods of transient noise
fig_extreme, ax_extreme = plt.subplots(figsize=(12, 6), layout="constrained")

# First highlight existing annotations (breaks) in light blue
for start, end in merged_bad:
    ax_extreme.axvspan(start, end, color='lightblue', alpha=0.5, label='Existing annotations')

ax_extreme.plot(times, data_normalized.T, color='gray', **plot_kwargs)

# Add shaded regions for extreme periods in red
for start_bad, end_bad in new_segs_bad:
    ax_extreme.axvspan(start_bad, end_bad, color='red', alpha=0.5, label='Bad segments')

# Set plot limits based on data
ax_extreme.set(
    title=f"Additional periods with extreme values (>{extreme_threshold} pT) highlighted in red", # Outside break times
    ylim=(-10, 10), 
    xlim=(times[0], times[-1]), 
    xlabel="Time (s)", 
    ylabel="Amplitude (pT)"
)
ax_extreme.grid(True)
plt.show()
report.add_figure(fig=fig_extreme,
                    title=f'Transient extreme noise identification (±{extreme_threshold} pT threshold)',
                    caption=f'Red shaded areas indicate {len(onsets)} time periods with extreme values, blue shaded areas are exisiting annotations')

## Final plot to visually check data signal per channel after preprocessing
raw_cleaned.plot()

#%% 19. Save data with ICA applied as _ica.fif and save MNE reports (hdf5 & html)
raw_cleaned.save(deriv_fname_fif_1_ica, overwrite=True)

report.save(report_ica_fname, overwrite=True)
report.save(html_report_ica_fname, overwrite=True, open_browser=True)  # Opens after save to view report
# %%

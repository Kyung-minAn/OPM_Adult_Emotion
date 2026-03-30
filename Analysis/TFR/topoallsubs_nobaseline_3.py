"""
TFR group analysis

@aewaitt 2025
"""
#%%
import os.path as op
import numpy as np
import mne
import matplotlib
matplotlib.use('QtAgg')  # Keep QtAgg for interactive plots
import matplotlib.pyplot as plt
plt.ion()
from mne_bids import BIDSPath
import time
from functools import wraps
from mne.report import Report

# Defining custom functions
def retry_on_network_error(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (OSError, IOError) as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
            return None
        return wrapper
    return decorator

@retry_on_network_error(max_retries=3, delay=3)
def load_epochs_safely(file_path):
    return mne.read_epochs(file_path, preload=True)

def standardize_event_ids(epochs_list, target_event_id):
    """Standardize event_id across all epochs objects"""
    standardized_epochs = []
    
    for epochs in epochs_list:
        # Create a copy to avoid modifying original
        epochs_copy = epochs.copy()
        
        # Update the event_id to match target
        epochs_copy.event_id = target_event_id.copy()
        
        # Update the events array to match new event_id
        for event_name, new_code in target_event_id.items():
            if event_name in epochs.event_id:
                old_code = epochs.event_id[event_name]
                # Update events array where old code appears
                mask = epochs_copy.events[:, 2] == old_code
                epochs_copy.events[mask, 2] = new_code
        
        standardized_epochs.append(epochs_copy)
    
    return standardized_epochs

def quick_topo_allsensors(tfr, name, figsize=(15, 12), baseline=None, mode="percent", 
                          subplot_size=0.06, spacing_factor=1.2, vmin=None, vmax=None,
                          avg_plot_size=0.12, avg_position=(0.02, 0.02)):
    """
    Plot TFRs for every channel at the location of the channel in a topographic layout.
    
    Parameters:
    -----------
    tfr : mne.time_frequency.AverageTFR
        The TFR object to plot
    name : str
        Name for the plot
    figsize : tuple
        Figure size (width, height)
    baseline : tuple or None
        Baseline period (start, end) in seconds, e.g., (-1.1, -0.1)
    mode : str
        Baseline correction mode ('percent', 'ratio', 'logratio', 'mean', 'zscore')
    subplot_size : float
        Size of individual subplots
    spacing_factor : float
        Factor to increase spacing between subplots (>1 = more spread out)
    vmin, vmax : float or None
        Custom color scale limits. If None, will be calculated from data
    avg_plot_size : float
        Size of the average plot in the corner (default: 0.12)
    avg_position : tuple
        Position of average plot (x, y) in figure coordinates
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Apply baseline correction if specified
    if baseline is not None:
        tfr_corrected = tfr.copy()
        tfr_corrected.apply_baseline(baseline, mode=mode)
    else:
        tfr_corrected = tfr
    
    # Get channel positions and other info
    picks, pos, merge_channels, names, _, sphere, clip_origin = mne.viz.topomap._prepare_topomap_plot(
        tfr_corrected, 'mag', sphere=None
    )
    
    # Normalize positions to [0, 1] range
    posx = pos.copy()
    posx = posx - pos.min(axis=0)
    posx = posx / posx.max(axis=0)
    
    # Apply spacing factor to spread out the positions more
    # Center around 0.5 and scale
    posx_centered = posx - 0.5
    posx_scaled = posx_centered * spacing_factor
    posx = posx_scaled + 0.5
    
    # Ensure positions stay within reasonable bounds
    margin = subplot_size
    posx = np.clip(posx, margin, 1 - margin)
    
    # Create figure with specified size
    fig = plt.figure(figsize=figsize)
    baseline_str = f" (baseline: {baseline} {mode})" if baseline else ""
    fig.suptitle(f'Topo Plot: {name}{baseline_str}', fontsize=16)
    
    # Plot each channel's TFR at its topographic position
    extent = (tfr_corrected.times[0], tfr_corrected.times[-1], 
              tfr_corrected.freqs[0], tfr_corrected.freqs[-1])
    
    # Calculate vmin/vmax from baseline-corrected data for better contrast
    if vmin is not None and vmax is not None:
        # Use custom scaling
        data_min, data_max = vmin, vmax
        print(f"Using custom scaling: {vmin:.2f} to {vmax:.2f}")
    elif mode == "percent":
        # For percent change, use symmetric range around 0
        data_abs_max = np.percentile(np.abs(tfr_corrected.data), 95)
        data_min, data_max = -data_abs_max, data_abs_max
        print(f"Auto percent scaling: {data_min:.3f} to {data_max:.3f}")
    else:
        data_min, data_max = np.percentile(tfr_corrected.data, [5, 95])
        print(f"Auto scaling: {data_min:.3f} to {data_max:.3f}")
    
    for ii in range(pos.shape[0]):
        # Create subplot at normalized position
        ax = fig.add_axes([posx[ii, 0] - subplot_size/2, 
                          posx[ii, 1] - subplot_size/2, 
                          subplot_size, subplot_size])
        
        # Plot the TFR for this channel
        im = ax.imshow(tfr_corrected.data[ii, :, :], 
                      aspect="auto", 
                      cmap='RdBu_r',
                      origin='lower', 
                      extent=extent, 
                      vmin=data_min, 
                      vmax=data_max)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add channel name as title (small font)
        if hasattr(tfr_corrected, 'ch_names'):
            ax.set_title(tfr_corrected.ch_names[ii], fontsize=7)
    
    # Add average plot in a corner (with controllable size and position)
    ax_avg = fig.add_axes([avg_position[0], avg_position[1], avg_plot_size, avg_plot_size])
    avg_data = tfr_corrected.data.mean(axis=0)
    
    # Use same scaling for average plot
    if vmin is not None and vmax is not None:
        avg_min, avg_max = vmin, vmax
    elif mode == "percent":
        avg_abs_max = np.percentile(np.abs(avg_data), 95)
        avg_min, avg_max = -avg_abs_max, avg_abs_max
    else:
        avg_min, avg_max = np.percentile(avg_data, [5, 95])
    
    im_avg = ax_avg.imshow(avg_data, 
                          aspect="auto", 
                          cmap='RdBu_r',
                          origin='lower', 
                          extent=extent, 
                          vmin=avg_min, 
                          vmax=avg_max)
    
    ax_avg.set_title('Average', fontsize=10)
    ax_avg.set_xlabel('Time (s)', fontsize=8)
    ax_avg.set_ylabel('Frequency (Hz)', fontsize=8)
    ax_avg.tick_params(labelsize=7)
    
    # Add colorbar (positioned relative to average plot)
    cbar_ax = fig.add_axes([avg_position[0] + avg_plot_size + 0.01, avg_position[1], 
                           0.01, avg_plot_size])
    cbar = plt.colorbar(im_avg, cax=cbar_ax)
    if mode == "percent":
        cbar.set_label('% Change', fontsize=8)
    else:
        cbar.set_label('Power', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    plt.show()
    return fig

# Alternative simpler version using subplots
def quick_topo_subplots(tfr, name, figsize=(15, 12)):
    """
    Alternative version using regular subplots in a grid layout.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_channels = tfr.data.shape[0]
    
    # Calculate grid size
    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f'TFR Grid Plot: {name}', fontsize=16)
    
    # Flatten axes array for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    extent = (tfr.times[0], tfr.times[-1], tfr.freqs[0], tfr.freqs[-1])
    data_min, data_max = np.percentile(tfr.data, [5, 95])
    
    for ii in range(n_channels):
        ax = axes[ii]
        
        im = ax.imshow(tfr.data[ii, :, :], 
                      aspect="auto", 
                      cmap='RdBu_r',
                      origin='lower', 
                      extent=extent, 
                      vmin=data_min, 
                      vmax=data_max)
        
        if hasattr(tfr, 'ch_names'):
            ax.set_title(tfr.ch_names[ii], fontsize=8)
        
        ax.set_xlabel('Time (s)', fontsize=6)
        ax.set_ylabel('Freq (Hz)', fontsize=6)
        ax.tick_params(labelsize=6)
    
    # Hide unused subplots
    for ii in range(n_channels, len(axes)):
        axes[ii].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    return fig

# File directory prep
#% 1. Load in resampled BIDs data
session = '01'
task = 'PassiveEmoVoice'
meg_suffix = 'meg'
meg_extension = '.fif'
bids_folder = '<enter path>'
epoch_suffix='epo_tfr'
l_freq = 1
h_freq= 80
l_h_suffix = f"{l_freq}-{h_freq}Hz_"
deriv_root = op.join(bids_folder, 'derivatives/preprocessing')  # output path
report_root = op.join(bids_folder, 'derivatives/mne-reports/group_level')  # RDS folder for reports
report_fname = op.join(report_root, f'group_report_nobaseline_{task}_{epoch_suffix}.hdf5') 
html_report_fname = op.join(report_root, f'group_report_nobaseline_{task}_{epoch_suffix}.html')
topo_group_fname = op.join(bids_folder, 'derivatives/grouplevelanalysis/topoplots_allsubs/NoBaseline')

# List of subject IDs
excsubs = [10,19]
s = list(range(1, 23))
for ex in excsubs:
    s.remove(ex)
subjects = [f'sub-{i:02d}' for i in s]  # puts into sub-01 format.
subnum = [f'{i:02d}' for i in s]  # puts into 01 format. 

# Conditions
conditions = ['AngryVoice', 'HappyVoice', 'NeutralVoice']

# Initialize dictionaries to store evoked data
all_epochs = {cond: [] for cond in conditions}
subject_counts = {cond: [] for cond in conditions}

# Load and process data for each subject
for subject in subnum:
    print(f"Processing subject {subject}...")

    try:
        # Path to the subject's epoched data
        #load data
        if subject =='21':
            run='02'
        else:
            run='01'

        bids_path = BIDSPath(subject=subject, session=session, datatype='meg',
                    task=task, run=run, suffix=l_h_suffix+epoch_suffix, 
                    root=deriv_root, extension=meg_extension, check=False)
        # Get the standard directory from BIDSPath
        standard_directory = bids_path.directory
        # Create the task subfolder path
        task_subfolder = op.join(standard_directory, task)
        # Full path to file
        file_path = op.join(task_subfolder, bids_path.basename)
        # Read the file directly
        epochs = load_epochs_safely(file_path)
        if epochs is None:
            print(f"Failed to load data for subject {subject} after multiple attempts")
            continue
        
        # Count trials per condition for this subject and store epochs
        for cond in conditions:
            cond_epochs = epochs[cond]
            if len(cond_epochs) > 0:  # Only add if there are trials
                all_epochs[cond].append(cond_epochs)
                print(f"  {subject}, {cond}: {len(cond_epochs)} trials")
            
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        continue

#%
# Concatenate epochs across subjects
angry_epochs_list = all_epochs['AngryVoice']
happy_epochs_list = all_epochs['HappyVoice'] 
neutral_epochs_list = all_epochs['NeutralVoice']

# 1. Find common channels across all subjects and both conditions
all_epochs_groups = angry_epochs_list + happy_epochs_list + neutral_epochs_list
print(f"Total epoch objects: {len(all_epochs_groups)}")
print(f"Expected: {len(subjects) * 3} (20 subjects × 3 conditions)")

common_channels = set(all_epochs_groups[0].ch_names)
for epochs in all_epochs_groups:
    common_channels = common_channels.intersection(set(epochs.ch_names))

common_channels = list(common_channels)
print(f"Found {len(common_channels)} channels common to all subjects")

#% Create a mapping from epoch index to subject
# Assuming order: all angry, then all happy, then all neutral
n_subjects = len(subjects)
epoch_to_subject = {}
for i in range(len(all_epochs_groups)):
    if i < n_subjects:  # angry epochs
        epoch_to_subject[i] = subjects[i]
    elif i < 2 * n_subjects:  # happy epochs
        epoch_to_subject[i] = subjects[i - n_subjects]
    else:  # neutral epochs
        epoch_to_subject[i] = subjects[i - 2 * n_subjects]

# Find all unique channels across all subjects
all_channels = set()
for epochs in all_epochs_groups:
    all_channels.update(epochs.ch_names)

# Find excluded channels
excluded_channels = all_channels - set(common_channels)
print(f"\nFound {len(excluded_channels)} channels that are excluded:")
print(f"Excluded channels: {sorted(list(excluded_channels))}")

# Track which subjects are missing each excluded channel
# Need to check per subject, not per epoch
missing_channel_info = {}
for channel in excluded_channels:
    missing_subjects = set()
    
    # Check each epoch and map back to subject
    for i, epochs in enumerate(all_epochs_groups):
        if channel not in epochs.ch_names:
            subject_id = epoch_to_subject[i]
            missing_subjects.add(subject_id)
    
    missing_channel_info[channel] = sorted(list(missing_subjects))

# Display detailed information about missing channels
print("\n" + "="*60)
print("DETAILED MISSING CHANNEL ANALYSIS")
print("="*60)

for channel in sorted(excluded_channels):
    missing_subjects = missing_channel_info[channel]
    n_missing = len(missing_subjects)
    n_total = len(subjects)
    n_present = n_total - n_missing
    
    print(f"\nChannel: {channel}")
    print(f"  Missing in {n_missing}/{n_total} subjects ({n_missing/n_total*100:.1f}%)")
    print(f"  Present in {n_present}/{n_total} subjects ({n_present/n_total*100:.1f}%)")
    print(f"  Missing in subjects: {missing_subjects}")

# Summary statistics
print(f"\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total subjects: {len(subjects)}")
print(f"Total epoch objects: {len(all_epochs_groups)}")
print(f"Total unique channels: {len(all_channels)}")
print(f"Common channels: {len(common_channels)}")
print(f"Excluded channels: {len(excluded_channels)}")

#%% Optional: Check for consistency across conditions within subjects
print(f"\n" + "="*60)
print("CONDITION CONSISTENCY CHECK")
print("="*60)

inconsistent_subjects = []
for i, subject in enumerate(subjects):
    # Get channels for each condition for this subject
    angry_channels = set(angry_epochs_list[i].ch_names)
    happy_channels = set(happy_epochs_list[i].ch_names)
    neutral_channels = set(neutral_epochs_list[i].ch_names)
    
    # Check if all conditions have the same channels
    if not (angry_channels == happy_channels == neutral_channels):
        inconsistent_subjects.append(subject)
        print(f"\nSubject {subject} has inconsistent channels across conditions:")
        print(f"  Angry: {len(angry_channels)} channels")
        print(f"  Happy: {len(happy_channels)} channels")  
        print(f"  Neutral: {len(neutral_channels)} channels")

if not inconsistent_subjects:
    print("All subjects have consistent channels across conditions ✓")


# 2. Pick only these common channels from each evoked object
angry_epochs_common = []
happy_epochs_common = []
neutral_epochs_common=[]

for epochs in angry_epochs_list:
    angry_epochs_common.append(epochs.copy().pick_channels(common_channels))

for epochs in happy_epochs_list:
    happy_epochs_common.append(epochs.copy().pick_channels(common_channels))

for epochs in neutral_epochs_list:
    neutral_epochs_common.append(epochs.copy().pick_channels(common_channels))

#% make sure event ID mappings are consitent (can have different ID for multirun files)
standard_event_id = {
    'AngryVoice': 1,
    'HappyVoice': 2, 
    'NeutralVoice': 3
}

# Standardize each condition
angry_epochs_std = standardize_event_ids(angry_epochs_common, standard_event_id)
happy_epochs_std = standardize_event_ids(happy_epochs_common, standard_event_id)
neutral_epochs_std = standardize_event_ids(neutral_epochs_common, standard_event_id)

# Concatenate epochs from all subjects for each condition
angry_epochs_all = mne.concatenate_epochs(angry_epochs_std)
happy_epochs_all = mne.concatenate_epochs(happy_epochs_std)
neutral_epochs_all = mne.concatenate_epochs(neutral_epochs_std)

#%% COMPUTE TFRs WITH BASELINE CORRECTION
print("="*60)
print("COMPUTING TFRs")
print("="*60)

# Test different baseline correction methods
baseline_methods = ['logratio', 'mean', None]
baseline_method = None #'percent' #'logratio' #'mean' #'logratio'  # Start with dB conversion - most common for power

print(f"Using baseline method: {baseline_method}")

# Recompute TFRs with better baseline correction
tfr_angry_subjects = []
tfr_happy_subjects = []
tfr_neutral_subjects = []

#% Time-frequency analysis of slower frequencies (<30 hz)
freqs = np.arange(2, 31, 1)
n_cycles = freqs #/ 2
time_bandwidth = 1#2.0 #compare to see how it looks! temporal smoothing here

matplotlib.use('Agg') # Switch to non-interactive backend to prevent showing plots

for i, subject in enumerate(subnum):
    print(f"Recomputing TFRs for subject {subject}...")
    
    try:
        # Get the standardized epochs for this subject
        angry_subj = angry_epochs_std[i]
        happy_subj = happy_epochs_std[i]
        neutral_subj = neutral_epochs_std[i]
        
        #Number of epochs per condition
        angry_trls = angry_subj.events.shape[0]
        happy_trls = happy_subj.events.shape[0]
        neutral_trls = neutral_subj.events.shape[0]

        # Compute TFRs (same parameters as before)
        tfr_angry_subj = angry_subj.compute_tfr(
            method="multitaper",
            freqs=freqs,
            picks='mag',
            return_itc=False,
            average=True,
            decim=2,
            n_jobs=-1,
            verbose=False,  
            n_cycles=n_cycles,
            use_fft=True,
            time_bandwidth=time_bandwidth)
        
        tfr_happy_subj = happy_subj.compute_tfr(
            method="multitaper",
            freqs=freqs,
            picks='mag',
            return_itc=False,
            average=True,
            decim=2,
            n_jobs=-1,
            verbose=False,
            n_cycles=n_cycles,
            use_fft=True,
            time_bandwidth=time_bandwidth)
        
        tfr_neutral_subj = neutral_subj.compute_tfr(
            method="multitaper",
            freqs=freqs,
            picks='mag',
            return_itc=False,
            average=True,
            decim=2,
            n_jobs=-1,
            verbose=False,
            n_cycles=n_cycles,
            use_fft=True,
            time_bandwidth=time_bandwidth)
        
        fig_ang = quick_topo_allsensors(tfr_angry_subj, rf'Angry TFRs Sub-{subject}, Trials={angry_trls}', 
                                baseline=None, 
                                figsize=(18, 15),
                                subplot_size=0.05,
                                spacing_factor=0.8,
                                vmin=-0.2, vmax=0.2,
                                avg_plot_size=0.08,
                                avg_position=(0.07, 0.07)) #(0.01,0.01) # Bottom-left corner
        ang_fname = op.join(topo_group_fname, f'Sub-{subject}_AngryTopo.png' )
        fig_ang.savefig(ang_fname)

        fig_happy = quick_topo_allsensors(tfr_happy_subj, rf'Happy TFRs Sub-{subject}, Trials={happy_trls}', 
                                baseline=None, 
                                figsize=(18, 15),
                                subplot_size=0.05,
                                spacing_factor=0.8,
                                vmin=-0.2, vmax=0.2,
                                avg_plot_size=0.08,
                                avg_position=(0.07, 0.07))
        happy_fname = op.join(topo_group_fname, f'Sub-{subject}_HappyTopo.png' )
        fig_happy.savefig(happy_fname)

        fig_neutral = quick_topo_allsensors(tfr_neutral_subj, rf'Neutral TFRs Sub-{subject}, Trials={neutral_trls}', 
                                baseline=None, 
                                figsize=(18, 15),
                                subplot_size=0.05,
                                spacing_factor=0.8,
                                vmin=-0.2, vmax=0.2,
                                avg_plot_size=0.08,
                                avg_position=(0.07, 0.07))
        neutral_fname = op.join(topo_group_fname, f'Sub-{subject}_NeutralTopo.png' )
        fig_neutral.savefig(neutral_fname)

        # Apply improved baseline correction
        if baseline_method is not None:
            tfr_angry_subj.apply_baseline(baseline=(-1.1, -0.1), mode=baseline_method)
            tfr_happy_subj.apply_baseline(baseline=(-1.1, -0.1), mode=baseline_method)
            tfr_neutral_subj.apply_baseline(baseline=(-1.1, -0.1), mode=baseline_method)
        
        # Store in lists
        tfr_angry_subjects.append(tfr_angry_subj)
        tfr_happy_subjects.append(tfr_happy_subj)
        tfr_neutral_subjects.append(tfr_neutral_subj)
        
    except Exception as e:
        print(f"Error computing TFRs for subject {subject}: {e}")
        continue

print(f"Successfully computed TFRs for {len(tfr_angry_subjects)} subjects")


#%% Generate MNE report
report = Report(title=f'group_report_{task}_{epoch_suffix}')

# plot sensor layout
info = angry_epochs_all.info
adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type='mag')  # Adjust ch_type if needed
adjplot = mne.viz.plot_ch_adjacency(epochs.info, adjacency, ch_names)
report.add_figure(fig=adjplot, title='Adjacency plot of sensor arrangement')

# %% all sensor topo plots
# fig_ang = quick_topo_allsensors(tfr_angry, 'Angry TFRs', 
#                                 baseline=None,
#                                 figsize=(18, 15),
#                                 subplot_size=0.05,
#                                 spacing_factor=0.8,
#                                 avg_plot_size=0.08,
#                                 avg_position=(0.07, 0.07)) #(0.01,0.01) # Bottom-left corner

# fig_happy = quick_topo_allsensors(tfr_happy, 'Happy TFRs', 
#                                 baseline=None,
#                                 figsize=(18, 15),
#                                 subplot_size=0.05,
#                                 spacing_factor=0.8,
#                                 avg_plot_size=0.08,
#                                 avg_position=(0.07, 0.07)) #(0.01,0.01) # Bottom-left corner

# fig_neutral = quick_topo_allsensors(tfr_neutral, 'Neutral TFRs', 
#                                 baseline=None,
#                                 figsize=(18, 15),
#                                 subplot_size=0.05,
#                                 spacing_factor=0.8,
#                                 avg_plot_size=0.08,
#                                 avg_position=(0.07, 0.07)) #(0.01,0.01) # Bottom-left corner
##Add figs to report
report.add_figure(fig=fig_ang, title='Angry - Topoplot arrangement of TFRs per sensor')
report.add_figure(fig=fig_happy, title='Happy - Topoplot arrangement of TFRs per sensor')
report.add_figure(fig=fig_neutral, title='Neutral - Topoplot arrangement of TFRs per sensor')

# %%  Diff data for diff plots
tfr_diff_a_h = tfr_angry_subjects.copy()
tfr_diff_a_h.data = (tfr_angry_subjects.data - tfr_happy_subjects.data)/(tfr_angry_subjects.data + tfr_happy_subjects.data)

tfr_diff_a_n = tfr_angry_subjects.copy()
tfr_diff_a_n.data = (tfr_angry_subjects.data - tfr_neutral_subjects.data)/(tfr_angry_subjects.data + tfr_neutral_subjects.data)

tfr_diff_h_n = tfr_happy_subjects.copy()
tfr_diff_h_n.data = (tfr_happy_subjects.data - tfr_neutral_subjects.data)/(tfr_happy_subjects.data + tfr_neutral_subjects.data)

#%% Diff plots
fig_ah = quick_topo_allsensors(tfr_diff_a_h, 'Angry-Happy',
                            baseline=None, 
                            mode=None,
                            figsize=(18, 15),
                            subplot_size=0.05,
                            spacing_factor=0.8,
                            vmin=-0.2, vmax=0.2,
                            avg_plot_size=0.08,
                            avg_position=(0.07, 0.07)) #(0.01,0.01) # Bottom-left corner

fig_an = quick_topo_allsensors(tfr_diff_a_n, 'Angry-Neutral', 
                            baseline=None, 
                            mode=None,
                            figsize=(18, 15),
                            subplot_size=0.05,
                            spacing_factor=0.8,
                            vmin=-0.2, vmax=0.2,
                            avg_plot_size=0.08,
                            avg_position=(0.07, 0.07))

fig_hn = quick_topo_allsensors(tfr_diff_h_n, 'Happy-Neutral', 
                            baseline=None, 
                            mode=None,
                            figsize=(18, 15),
                            subplot_size=0.05,
                            spacing_factor=0.8,
                            vmin=-0.2, vmax=0.2,
                            avg_plot_size=0.08,
                            avg_position=(0.07, 0.07))
##Add diff figs to report
report.add_figure(fig=fig_ah, title='Angry-Happy - Topoplot arrangement of diff TFRs per sensor')
report.add_figure(fig=fig_an, title='Angry-Neutral - Topoplot arrangement of diff TFRs per sensor')
report.add_figure(fig=fig_hn, title='Happy-Neutral - Topoplot arrangement of diff TFRs per sensor')

# %% Strong sensors to focus on
#right temporal strong sensors
right_strong = ['R407_bz-s26','R505_bz-s24','R507_bz-s22','R503_bz-s17']
#left temporal strong sensors
left_strong = ['L407_bz-s41','L505_bz-s74','L603_bz-s78','L503_bz-s79']

# Save current backend
original_backend = matplotlib.get_backend()

#%% Right strong sensors plot
for sensors_r in right_strong:
    # Switch to non-interactive backend to prevent showing plots
    matplotlib.use('Agg')
    
    # Create a figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot each condition in its own subplot
    tfr_angry_subjects.plot(
        picks=[sensors_r], 
        baseline=None, 
        tmin=-1.5, tmax=3,
        axes=axes[0],
        show=False)
    axes[0].set_title(f"ANGRY\n{sensors_r}", fontsize=12, fontweight='bold')
    
    tfr_happy_subjects.plot(
        picks=[sensors_r],
        baseline=None, 
        tmin=-1.5, tmax=3,
        axes=axes[1],
        show=False)
    axes[1].set_title(f"HAPPY\n{sensors_r}", fontsize=12, fontweight='bold')
    
    tfr_neutral_subjects.plot(
        picks=[sensors_r],
        baseline=None, 
        tmin=-1.5, tmax=3,
        axes=axes[2],
        show=False)
    axes[2].set_title(f"NEUTRAL\n{sensors_r}", fontsize=12, fontweight='bold')
    
    # Set main title for the entire figure
    fig.suptitle(f'Right Strong Sensor TFR Comparison: {sensors_r}', fontsize=16)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Add to report
    report.add_figure(fig=fig, title=f'TFR Comparison for {sensors_r} (Right)')
    
    # Switch back to interactive backend if want to see plots later
    # matplotlib.use(original_backend)

#%% Left strong sensors plot
for sensors_l in left_strong:
    # Switch to non-interactive backend to prevent showing plots
    matplotlib.use('Agg')
    
    # Create a figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot each condition in its own subplot
    tfr_angry_subjects.plot(
        picks=[sensors_l],
        baseline=None,
        tmin=-1.5, tmax=3,
        axes=axes[0],
        show=False)
    axes[0].set_title(f"ANGRY\n{sensors_l}", fontsize=12, fontweight='bold')
    
    tfr_happy_subjects.plot(
        picks=[sensors_l],
        baseline=None, 
        tmin=-1.5, tmax=3,
        axes=axes[1],
        show=False)
    axes[1].set_title(f"HAPPY\n{sensors_l}", fontsize=12, fontweight='bold')
    
    tfr_neutral_subjects.plot(
        picks=[sensors_l],
        baseline=None, 
        tmin=-1.5, tmax=3,
        axes=axes[2],
        show=False)
    axes[2].set_title(f"NEUTRAL\n{sensors_l}", fontsize=12, fontweight='bold')
    
    # Set main title for the entire figure
    fig.suptitle(f'Left Strong Sensor TFR Comparison: {sensors_l}', fontsize=16)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Add to report
    report.add_figure(fig=fig, title=f'TFR Comparison for {sensors_l} (Left)')

#%%
report.save(report_fname, overwrite=True)
report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
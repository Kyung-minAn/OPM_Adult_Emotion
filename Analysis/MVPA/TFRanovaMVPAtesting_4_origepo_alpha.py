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
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer, LinearModel
from mne.stats import permutation_cluster_test

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

# File directory prep
#%% 1. Load in resampled BIDs data
session = '01'
task = 'PassiveEmoVoice'
meg_suffix = 'meg'
meg_extension = '.fif'
bids_folder = '<enter path>'
epoch_suffix='epo_orig_tfr'
l_freq = 1
h_freq= 80
l_h_suffix = f"{l_freq}-{h_freq}Hz_"
deriv_root = op.join(bids_folder, 'derivatives/preprocessing')  # output path

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
        # Full path to your file
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

## 2. Create a mapping from epoch index to subject
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

# Summary info
print(f"\n" + "="*60)
print("SUMMARY INFO")
print("="*60)
print(f"Total subjects: {len(subjects)}")
print(f"Total epoch objects: {len(all_epochs_groups)}")
print(f"Total unique channels: {len(all_channels)}")
print(f"Common channels: {len(common_channels)}")
print(f"Excluded channels: {len(excluded_channels)}")

## 3. Consistency check across conditions within subjects
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

## 4. Pick only these common channels from each evoked object
angry_epochs_common = []
happy_epochs_common = []
neutral_epochs_common=[]

for epochs in angry_epochs_list:
    angry_epochs_common.append(epochs.copy().pick_channels(common_channels))
for epochs in happy_epochs_list:
    happy_epochs_common.append(epochs.copy().pick_channels(common_channels))
for epochs in neutral_epochs_list:
    neutral_epochs_common.append(epochs.copy().pick_channels(common_channels))

## 5. Make sure event ID mappings are consitent (can have different ID for multirun files)
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

#%% 4. MVPA setup
print("\n" + "="*60)
print("SUBJECT-LEVEL MVPA APPROACH")
print("="*60)

freq_range = (8, 12) # Alpha band can change to any freqs
baseline_period = (-0.1, 0.0)  # Mean subtraction baseline window (s)
analysis_window = (-0.1, 1.5)  # Analysis window (s)
n_splits = 5 #4 # CV folds
plot_dir='Z:\EmotionVoiceAdultOPM\Analysis\MVPA_plots'
 
cond_pairs = [
    ('AngryVoice', 'HappyVoice'),
    ('AngryVoice', 'NeutralVoice'),
    ('HappyVoice', 'NeutralVoice'),
]
 
# Map condition name to lists
epochs_lists = {
    'AngryVoice'  : angry_epochs_std, # angry_epochs_list,
    'HappyVoice'  : happy_epochs_std, #happy_epochs_list,
    'NeutralVoice': neutral_epochs_std, #neutral_epochs_list,
}
# %% 5. Run MVPA
all_scores = {f"{c1}_vs_{c2}": [] for c1, c2 in cond_pairs}
times      = None

for subj_idx, subj_id in enumerate(subnum):
    print(f"\nSubject {subj_id}")
    subj_scores_this = {}
    for cond1, cond2 in cond_pairs:
        try:
            ep1 = epochs_lists[cond1][subj_idx].copy()
            ep2 = epochs_lists[cond2][subj_idx].copy()
  
            ep1 = ep1.filter(*freq_range, method='fir', phase='zero', verbose=False)
            ep2 = ep2.filter(*freq_range, method='fir', phase='zero', verbose=False)
 
            ep1.resample(500)
            ep2.resample(500)

            ep1.apply_baseline(baseline_period, verbose=False)
            ep2.apply_baseline(baseline_period, verbose=False)
 
            ep1.crop(*analysis_window)
            ep2.crop(*analysis_window)
 
            if times is None:
                times = ep1.times
 
            X1 = ep1.get_data(picks='mag')
            X2 = ep2.get_data(picks='mag')
            X  = np.concatenate([X1, X2], axis=0)
            print(f"X shape: {X.shape}")  # Expect 3D (n_trials, n_channels, n_times)
            y  = np.array([0] * len(X1) + [1] * len(X2))
 
            print(f"  {cond1} vs {cond2}: {len(X1)} / {len(X2)} trials")
 
            #Sliding Estimator = linear SVM at each timepoint
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel='linear', C=1.0, class_weight='balanced')
            )
 
            time_decod = SlidingEstimator(clf, scoring='roc_auc', n_jobs=-1, verbose=False)
            
            # Shuffle X and y together before the classifier otherwise all in order
            rng = np.random.default_rng(42)
            idx = rng.permutation(len(y))
            X   = X[idx]
            y   = y[idx]

            scores = cross_val_multiscore(time_decod, X, y,
                                          cv=n_splits, n_jobs=-1)
            scores = scores.mean(axis=0)   # average over folds -> (n_times,)
 
            all_scores[f"{cond1}_vs_{cond2}"].append(scores)
            subj_scores_this[f"{cond1}_vs_{cond2}"] = scores
            print(f"    Peak AUC = {scores.max():.3f} "
                  f"at {times[scores.argmax()]*1000:.0f} ms")
            
        except Exception as e:
            print(f"  {cond1} vs {cond2}: skipped — {e}")
            continue   

    if subj_scores_this:
        fig, axes = plt.subplots(1, len(cond_pairs),
                                figsize=(6 * len(cond_pairs), 4),
                                sharey=True)

        for ax, (cond1, cond2) in zip(axes, cond_pairs):
            key = f"{cond1}_vs_{cond2}"

            if key not in subj_scores_this:
                ax.set_title(f'{cond1} vs {cond2}\n(no data)', fontsize=10)
                continue
            
            times_ms  = times * 1000
            s         = subj_scores_this[key]
            peak_auc  = s.max()
            peak_t_ms = times_ms[s.argmax()]

            ax.axhline(0.5, color='k', linestyle='--', label='Chance')
            ax.axvline(0.0, color='k', linestyle='-', alpha=0.4)
            ax.plot(times_ms, s, linewidth=1.5, label=f'Sub {subj_id}')

            # Mark and label peak
            ax.axvline(peak_t_ms, color='red', linestyle=':', alpha=0.7)
            ax.annotate(f'{peak_auc:.2f}\n{peak_t_ms:.0f}ms',
                        xy=(peak_t_ms, peak_auc),
                        xytext=(peak_t_ms + 40, peak_auc - 0.03),
                        fontsize=7, color='red',
                        arrowprops=dict(arrowstyle='->', color='red',
                                        lw=0.8))
            ax.set_title(f'{cond1} vs {cond2}', fontsize=10)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('AUC')
            ax.set_ylim(0.35, 0.80)
            ax.legend(fontsize=8)

        plt.suptitle(
            f'Subject {subj_id} — MVPA orig emo categories'
            f'({freq_range[0]}-{freq_range[1]} Hz)',
            fontsize=12)
        plt.tight_layout()
        plt.savefig(op.join(plot_dir, f'sub-{subj_id}_mvpa.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_dir}/sub-{subj_id}_mvpa.png")

# %% Plots
fig, axes = plt.subplots(1, len(cond_pairs),
                         figsize=(6 * len(cond_pairs), 4),
                         sharey=True)

for ax, (cond1, cond2) in zip(axes, cond_pairs):
 
    key          = f"{cond1}_vs_{cond2}"
    group_matrix = np.array(all_scores[key])   # (n_subjects, n_times)
    mean_scores  = group_matrix.mean(axis=0)
    sem_scores   = group_matrix.std(axis=0) / np.sqrt(len(group_matrix))
    times_ms     = times * 1000
 
    # Using pointwise t-test vs chance for exploratory analysis
    # Quicker to run, otherwise mne.stats.permutation_cluster_1samp_test for final analysis
    _, p_vals = stats.ttest_1samp(group_matrix, 0.5, axis=0)
 
    ax.axhline(0.5, color='k', linestyle='--', label='Chance')
    ax.axvline(0.0, color='k', linestyle='-', alpha=0.4)
    ax.fill_between(times_ms,
                    mean_scores - sem_scores,
                    mean_scores + sem_scores, alpha=0.3)
    ax.plot(times_ms, mean_scores, linewidth=2,
            label=f'n={len(group_matrix)}')
    
    # Peak annotation
    peak_idx  = mean_scores.argmax()
    peak_t_ms = times_ms[peak_idx]
    peak_auc  = mean_scores[peak_idx]
    ax.axvline(peak_t_ms, color='red', linestyle=':', alpha=0.7)
    ax.annotate(f'{peak_auc:.2f}\n{peak_t_ms:.0f}ms',
                xy=(peak_t_ms, peak_auc),
                xytext=(peak_t_ms + 40, peak_auc - 0.03),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    
    # Red bar marking uncorrected significant timepoints
    sig = p_vals < 0.05
    if sig.any():
        ax.fill_between(times_ms, 0.495, 0.505,
                        where=sig, color='red', alpha=0.7,
                        label='p<0.05 uncorrected')
 
    ax.set_title(f'{cond1} vs {cond2}', fontsize=10)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('AUC')
    ax.set_ylim(0.35, 0.80)
    ax.legend(fontsize=8)
 
plt.suptitle(
    f'MVPA: Sliding Estimator orig emo categories'
    f'({freq_range[0]}-{freq_range[1]} Hz)',
    fontsize=12)
plt.tight_layout()
plt.savefig(op.join(plot_dir, 'group_mvpa_results.png'),
                    dpi=150, bbox_inches='tight')
plt.show()
 
# Summary
print("\n" + "=" * 50)
for key, scores_list in all_scores.items():
    m = np.array(scores_list).mean(axis=0)
    print(f"{key}: peak AUC = {m.max():.3f} "
          f"at {times[m.argmax()]*1000:.0f} ms")
# %%

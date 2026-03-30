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

#%% 6. Time-frequency analysis of slower frequencies (<30 hz)
freqs = np.arange(2, 31, 1)
n_cycles = freqs / 2
time_bandwidth = 2.0

#% RECOMPUTE TFRs WITH BETTER BASELINE CORRECTION
print("="*60)
print("RECOMPUTING TFRs WITH IMPROVED BASELINE CORRECTION")
print("="*60)

# Test different baseline correction methods
baseline_methods = ['logratio', 'mean', None]
baseline_method = 'logratio'  # Start with dB conversion - most common for power

print(f"Using baseline method: {baseline_method}")

# Recompute TFRs with better baseline correction
tfr_angry_subjects_new = []
tfr_happy_subjects_new = []
tfr_neutral_subjects_new = []

for i, subject in enumerate(subnum):
    print(f"Recomputing TFRs for subject {subject}...")
    
    try:
        # Get the standardized epochs for this subject
        angry_subj = angry_epochs_std[i]
        happy_subj = happy_epochs_std[i]
        neutral_subj = neutral_epochs_std[i]
        
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
        
        # Apply improved baseline correction
        if baseline_method is not None:
            tfr_angry_subj.apply_baseline(baseline=(-1.1, -0.1), mode=baseline_method)
            tfr_happy_subj.apply_baseline(baseline=(-1.1, -0.1), mode=baseline_method)
            tfr_neutral_subj.apply_baseline(baseline=(-1.1, -0.1), mode=baseline_method)
        
        # Store in lists
        tfr_angry_subjects_new.append(tfr_angry_subj)
        tfr_happy_subjects_new.append(tfr_happy_subj)
        tfr_neutral_subjects_new.append(tfr_neutral_subj)
        
        # Print some statistics to check values
        if i == 0:  # Just for first subject
            print(f"  Power value range - Angry: {tfr_angry_subj.data.min():.3f} to {tfr_angry_subj.data.max():.3f}")
            print(f"  Power value range - Happy: {tfr_happy_subj.data.min():.3f} to {tfr_happy_subj.data.max():.3f}")
            print(f"  Power value range - Neutral: {tfr_neutral_subj.data.min():.3f} to {tfr_neutral_subj.data.max():.3f}")
        
    except Exception as e:
        print(f"Error computing TFRs for subject {subject}: {e}")
        continue

print(f"Successfully computed TFRs for {len(tfr_angry_subjects_new)} subjects")

# Update the tfr lists to use the new baseline-corrected data
tfr_angry_subjects = tfr_angry_subjects_new
tfr_happy_subjects = tfr_happy_subjects_new
tfr_neutral_subjects = tfr_neutral_subjects_new

#%% IMPROVED FEATURE EXTRACTION
print("\n" + "="*60)
print("IMPROVED FEATURE EXTRACTION")
print("="*60)

# Define frequency bands and time windows
freq_bands = {
    'theta': (4, 8),
    'alpha': (8, 12), 
    'beta': (13, 30)
}

time_windows = {
    'theta': (0.0, 0.3),      # Early processing
    'alpha': (0.3, 1.6),     # Sustained processing  
    'beta': (1.6, 1.9)       # Offset processing
}

def extract_improved_features(tfr_subjects_dict, freq_bands, time_windows):
    """
    Extract features with better handling of power values.
    """
    features_list = []
    
    for condition, tfr_list in tfr_subjects_dict.items():
        for subj_idx, tfr in enumerate(tfr_list):
            subject_id = subnum[subj_idx]
            
            for band_name, (low_freq, high_freq) in freq_bands.items():
                time_start, time_end = time_windows[band_name]
                
                # Select frequency and time ranges
                tfr_crop = tfr.copy().crop(tmin=time_start, tmax=time_end, 
                                          fmin=low_freq, fmax=high_freq)
                
                # Extract mean power across time-frequency window
                mean_power = tfr_crop.data.mean(axis=(1, 2))  # Mean across freqs and times
                
                # Store features
                features_list.append({
                    'subject': subject_id,
                    'condition': condition,
                    'freq_band': band_name,
                    'mean_power': mean_power.mean(),  # Average across all channels
                    'channel_powers': mean_power,     # Individual channel powers
                    'raw_power_stats': {
                        'min': tfr_crop.data.min(),
                        'max': tfr_crop.data.max(),
                        'mean': tfr_crop.data.mean(),
                        'std': tfr_crop.data.std()
                    }
                })
    
    return pd.DataFrame(features_list)

# Extract features with improved baseline correction
tfr_subjects_dict = {
    'AngryVoice': tfr_angry_subjects,
    'HappyVoice': tfr_happy_subjects,
    'NeutralVoice': tfr_neutral_subjects
}

features_df = extract_improved_features(tfr_subjects_dict, freq_bands, time_windows)

print(f"Extracted features shape: {features_df.shape}")
print("\nPower value statistics:")
print(features_df.groupby(['condition', 'freq_band'])['mean_power'].describe())

#%% FIXED STATISTICAL ANALYSIS WITH PROPER P-VALUES AND PLOTS

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, f_oneway
import warnings
warnings.filterwarnings('ignore')

#%% 1. PROPER STATISTICAL ANALYSIS WITH P-VALUES
print("="*60)
print("STATISTICAL ANALYSIS WITH P-VALUES")
print("="*60)

# Ensure correct data structure
print(f"Features dataframe shape: {features_df.shape}")
print(f"Columns: {features_df.columns.tolist()}")
print(f"Unique subjects: {features_df['subject'].nunique()}")
print(f"Unique conditions: {features_df['condition'].unique()}")
print(f"Unique freq bands: {features_df['freq_band'].unique()}")

# Create a more detailed statistical summary
def perform_statistical_tests(features_df):
    """
    Perform comprehensive statistical tests on the feature data.
    """
    results = {}
    
    # Get unique conditions and frequency bands
    conditions = features_df['condition'].unique()
    freq_bands = features_df['freq_band'].unique()
    
    print("\nDETAILED STATISTICAL RESULTS:")
    print("="*50)
    
    for band in freq_bands:
        print(f"\n{band.upper()} BAND ANALYSIS:")
        print("-" * 40)
        
        # Get data for this frequency band
        band_data = features_df[features_df['freq_band'] == band]
        
        # Descriptive statistics
        desc_stats = band_data.groupby('condition')['mean_power'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        print("Descriptive Statistics:")
        print(desc_stats)
        
        # Prepare data for pairwise comparisons
        condition_data = {}
        for condition in conditions:
            condition_data[condition] = band_data[
                band_data['condition'] == condition
            ]['mean_power'].values
        
        # Pairwise t-tests
        print(f"\nPairwise T-tests for {band} band:")
        comparison_results = {}
        
        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions):
                if i < j:  # Avoid duplicate comparisons
                    data1 = condition_data[cond1]
                    data2 = condition_data[cond2]
                    
                    if len(data1) == len(data2) and len(data1) > 0:
                        # Paired t-test
                        t_stat, p_val = ttest_rel(data1, data2)
                        
                        # Effect size (Cohen's d for paired samples)
                        diff = data1 - data2
                        d = diff.mean() / diff.std()
                        
                        # Significance stars
                        if p_val < 0.001:
                            sig = "***"
                        elif p_val < 0.01:
                            sig = "**"
                        elif p_val < 0.05:
                            sig = "*"
                        else:
                            sig = "ns"
                        
                        comparison_results[f"{cond1}_vs_{cond2}"] = {
                            't_stat': t_stat,
                            'p_val': p_val,
                            'effect_size': d,
                            'significance': sig
                        }
                        
                        print(f"  {cond1} vs {cond2}: t={t_stat:.3f}, p={p_val:.4f} {sig}, d={d:.3f}")
        
        # One-way ANOVA for overall effect
        anova_data = [condition_data[cond] for cond in conditions]
        f_stat, p_val_anova = f_oneway(*anova_data)
        
        if p_val_anova < 0.001:
            sig_anova = "***"
        elif p_val_anova < 0.01:
            sig_anova = "**"
        elif p_val_anova < 0.05:
            sig_anova = "*"
        else:
            sig_anova = "ns"
        
        print(f"  Overall ANOVA: F={f_stat:.3f}, p={p_val_anova:.4f} {sig_anova}")
        
        results[band] = {
            'descriptive': desc_stats,
            'pairwise': comparison_results,
            'anova': {'f_stat': f_stat, 'p_val': p_val_anova, 'significance': sig_anova}
        }
    
    return results

# Run statistical analysis
statistical_results = perform_statistical_tests(features_df)

#%% 2. COMPREHENSIVE VISUALIZATION WITH STATISTICAL ANNOTATIONS
print("\n" + "="*60)
print("COMPREHENSIVE VISUALIZATION")
print("="*60)

def create_comprehensive_plots(features_df, statistical_results):
    """
    Create comprehensive plots with statistical annotations.
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Boxplot with significance annotations
    plt.subplot(3, 3, 1)
    ax1 = sns.boxplot(data=features_df, x='freq_band', y='mean_power', hue='condition')
    plt.title('Mean Power by Frequency Band and Condition\n(with individual data points)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add individual points
    sns.stripplot(data=features_df, x='freq_band', y='mean_power', hue='condition', 
                 dodge=True, alpha=0.7, size=3, ax=ax1)
    
    # Add significance annotations
    y_max = features_df['mean_power'].max()
    for i, band in enumerate(features_df['freq_band'].unique()):
        if band in statistical_results:
            pairwise_results = statistical_results[band]['pairwise']
            y_pos = y_max * 1.1
            
            # Add significance stars for most significant comparison
            if pairwise_results:
                min_p = min([res['p_val'] for res in pairwise_results.values()])
                if min_p < 0.05:
                    sig_symbol = "***" if min_p < 0.001 else "**" if min_p < 0.01 else "*"
                    plt.text(i, y_pos, sig_symbol, ha='center', va='bottom', fontsize=12)
    
    # Plot 2: Statistical summary table
    plt.subplot(3, 3, 2)
    plt.axis('off')
    
    # Create summary table
    summary_text = "STATISTICAL SUMMARY\n" + "="*25 + "\n\n"
    for band, results in statistical_results.items():
        summary_text += f"{band.upper()} BAND:\n"
        summary_text += f"ANOVA: F={results['anova']['f_stat']:.3f}, "
        summary_text += f"p={results['anova']['p_val']:.4f} {results['anova']['significance']}\n"
        
        for comparison, res in results['pairwise'].items():
            summary_text += f"{comparison}: p={res['p_val']:.4f} {res['significance']}\n"
        summary_text += "\n"
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Plot 3: Effect sizes
    plt.subplot(3, 3, 3)
    effect_sizes = []
    comparisons = []
    bands = []
    
    for band, results in statistical_results.items():
        for comparison, res in results['pairwise'].items():
            effect_sizes.append(res['effect_size'])
            comparisons.append(comparison.replace('_vs_', ' vs '))
            bands.append(band)
    
    if effect_sizes:
        effect_df = pd.DataFrame({
            'effect_size': effect_sizes,
            'comparison': comparisons,
            'band': bands
        })
        
        sns.barplot(data=effect_df, x='band', y='effect_size', hue='comparison')
        plt.title('Effect Sizes (Cohen\'s d)')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Small effect')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Medium effect')
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.9, label='Large effect')
        plt.legend()
        plt.xticks(rotation=45)
    
    # Plot 4: Individual subject trajectories with proper statistics
    plt.subplot(3, 3, 4)
    pivot_data = features_df.pivot_table(
        index=['subject', 'freq_band'], 
        columns='condition', 
        values='mean_power'
    ).reset_index()
    
    colors = ['red', 'blue', 'green']
    for i, band in enumerate(features_df['freq_band'].unique()):
        band_data = pivot_data[pivot_data['freq_band'] == band]
        if len(band_data) > 0:
            plt.scatter(band_data['AngryVoice'], band_data['HappyVoice'], 
                       c=colors[i], label=band, alpha=0.7, s=50)
    
    plt.xlabel('Angry Voice Power')
    plt.ylabel('Happy Voice Power')
    plt.title('Subject-wise Angry vs Happy\n(each point = one subject)')
    plt.legend()
    
    # Add diagonal line
    min_val = min(plt.xlim()[0], plt.ylim()[0])
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Plot 5: Correlation matrix
    plt.subplot(3, 3, 5)
    correlation_data = features_df.pivot_table(
        index='subject',
        columns=['condition', 'freq_band'],
        values='mean_power'
    )
    
    if not correlation_data.empty:
        corr_matrix = correlation_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', mask=mask)
        plt.title('Correlation Matrix')
    
    # Plot 6: P-value heatmap
    plt.subplot(3, 3, 6)
    p_values = []
    band_names = []
    comparison_names = []
    
    for band, results in statistical_results.items():
        for comparison, res in results['pairwise'].items():
            p_values.append(res['p_val'])
            band_names.append(band)
            comparison_names.append(comparison.replace('_vs_', '\nvs '))
    
    if p_values:
        p_df = pd.DataFrame({
            'p_value': p_values,
            'band': band_names,
            'comparison': comparison_names
        })
        
        p_matrix = p_df.pivot(index='comparison', columns='band', values='p_value')
        sns.heatmap(p_matrix, annot=True, cmap='Reds_r', fmt='.3f', 
                   cbar_kws={'label': 'p-value'})
        plt.title('P-values Heatmap\n(darker = more significant)')
    
    # Plot 7-9: Individual frequency band distributions
    for i, band in enumerate(features_df['freq_band'].unique()):
        plt.subplot(3, 3, 7 + i)
        band_data = features_df[features_df['freq_band'] == band]
        
        # Violin plot with individual points
        sns.violinplot(data=band_data, x='condition', y='mean_power', alpha=0.7)
        sns.stripplot(data=band_data, x='condition', y='mean_power', 
                     color='black', alpha=0.6, size=3)
        
        plt.title(f'{band.title()} Band Power Distribution')
        plt.xticks(rotation=45)
        
        # Add statistical annotation
        if band in statistical_results:
            anova_result = statistical_results[band]['anova']
            plt.text(0.5, 0.95, f"F={anova_result['f_stat']:.3f}, p={anova_result['p_val']:.4f} {anova_result['significance']}", 
                    transform=plt.gca().transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Create comprehensive plots
fig = create_comprehensive_plots(features_df, statistical_results)

#%% 3. SUMMARY REPORT
print("\n" + "="*60)
print("FINAL STATISTICAL REPORT")
print("="*60)

def generate_summary_report(statistical_results):
    """
    Generate a comprehensive summary report.
    """
    print("SUMMARY OF SIGNIFICANT EFFECTS:")
    print("-" * 40)
    
    significant_effects = []
    
    for band, results in statistical_results.items():
        # Check ANOVA
        if results['anova']['p_val'] < 0.05:
            significant_effects.append(f"{band.upper()} band shows overall significant differences (p={results['anova']['p_val']:.4f})")
        
        # Check pairwise comparisons
        for comparison, res in results['pairwise'].items():
            if res['p_val'] < 0.05:
                significant_effects.append(
                    f"{band.upper()} band: {comparison.replace('_vs_', ' vs ')} "
                    f"(p={res['p_val']:.4f}, d={res['effect_size']:.3f})"
                )
    
    if significant_effects:
        for effect in significant_effects:
            print(f"✓ {effect}")
    else:
        print("No significant effects found.")
    
    print(f"\nTOTAL COMPARISONS TESTED: {sum(len(results['pairwise']) for results in statistical_results.values())}")
    print("NOTE: Consider multiple comparison correction if needed (e.g., Bonferroni)")

generate_summary_report(statistical_results)

#%% 4. SUBJECT-LEVEL MVPA APPROACH (experimenting)
print("\n" + "="*60)
print("SUBJECT-LEVEL MVPA APPROACH")
print("="*60)

def run_subject_level_mvpa(epochs_dict, freq_range=(4, 8), time_range=(-0.2, 2.0)):
    """
    Run MVPA separately for each subject, then aggregate results.
    This addresses the concern about individual variability.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    
    print("Running subject-level MVPA analysis...")
    
    # Get subject IDs
    n_subjects = len(subnum)
    
    # Initialize results storage
    subject_results = {}
    
    # Define condition pairs
    condition_pairs = [
        ('AngryVoice', 'HappyVoice'),
        ('AngryVoice', 'NeutralVoice'),
        ('HappyVoice', 'NeutralVoice')
    ]
    
    for comparison_idx, (cond1, cond2) in enumerate(condition_pairs):
        print(f"\nAnalyzing {cond1} vs {cond2}:")
        
        subject_scores = []
        
        for subj_idx in range(n_subjects):
            try:
                # Get epochs for this subject and condition pair
                epochs1 = epochs_dict[cond1].copy()
                epochs2 = epochs_dict[cond2].copy()
                
                # Filter frequency range
                if freq_range:
                    epochs1 = epochs1.filter(l_freq=freq_range[0], h_freq=freq_range[1])
                    epochs2 = epochs2.filter(l_freq=freq_range[0], h_freq=freq_range[1])
                
                # Crop time range
                if time_range:
                    epochs1 = epochs1.crop(tmin=time_range[0], tmax=time_range[1])
                    epochs2 = epochs2.crop(tmin=time_range[0], tmax=time_range[1])
                
                # Get data for this subject (need to extract subject-specific trials)
                # Using a time-averaged approach per subject
                
                # Average across time for each trial, then use trials as samples
                X1 = epochs1.get_data(picks='mag').mean(axis=2)  # Shape: (n_trials, n_channels)
                X2 = epochs2.get_data(picks='mag').mean(axis=2)
                
                # Combine data
                X = np.vstack([X1, X2])
                y = np.hstack([np.zeros(len(X1)), np.ones(len(X2))])
                
                # Only proceed if have enough trials
                if len(X) >= 10:  # Minimum 10 trials total
                    # Create classifier
                    clf = Pipeline([
                        ('scaler', StandardScaler()),
                        ('svm', SVC(kernel='linear', C=1.0, probability=True))
                    ])
                    
                    # Cross-validation
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    
                    cv_scores = []
                    for train_idx, test_idx in cv.split(X, y):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        clf.fit(X_train, y_train)
                        y_pred_proba = clf.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, y_pred_proba)
                        cv_scores.append(auc)
                    
                    subject_score = np.mean(cv_scores)
                    subject_scores.append(subject_score)
                    
                    print(f"  Subject {subnum[subj_idx]}: AUC = {subject_score:.3f}")
                
            except Exception as e:
                print(f"  Subject {subnum[subj_idx]}: Error - {e}")
                continue
        
        # Store results for this comparison
        if subject_scores:
            subject_results[f"{cond1}_vs_{cond2}"] = {
                'scores': subject_scores,
                'mean_auc': np.mean(subject_scores),
                'std_auc': np.std(subject_scores),
                'n_subjects': len(subject_scores)
            }
            
            # Statistical test against chance
            t_stat, p_val = stats.ttest_1samp(subject_scores, 0.5)
            subject_results[f"{cond1}_vs_{cond2}"].update({
                't_stat': t_stat,
                'p_val': p_val
            })
            
            print(f"  Group result: {np.mean(subject_scores):.3f} ± {np.std(subject_scores):.3f}")
            print(f"  vs Chance: t={t_stat:.3f}, p={p_val:.3f}")
    
    return subject_results

# Run subject-level MVPA
print("Note: This approach treats each subject separately, then aggregates results.")
print("This controls for individual differences better than pooling all trials.")

epochs_dict = {
    'AngryVoice': angry_epochs_all,
    'HappyVoice': happy_epochs_all,
    'NeutralVoice': neutral_epochs_all
}
# Example with theta range
subject_mvpa_results = run_subject_level_mvpa(
    epochs_dict, 
    freq_range=(4, 8),  # Theta range
    time_range=(-0.2, 2.0)
)

print("\n" + "="*60)
print("SUBJECT-LEVEL MVPA RESULTS")
print("="*60)

for comparison, result in subject_mvpa_results.items():
    print(f"\n{comparison}:")
    print(f"  Mean AUC: {result['mean_auc']:.3f} ± {result['std_auc']:.3f}")
    print(f"  N subjects: {result['n_subjects']}")
    print(f"  vs Chance: t={result['t_stat']:.3f}, p={result['p_val']:.3f}")
    
    sig = "***" if result['p_val'] < 0.001 else "**" if result['p_val'] < 0.01 else "*" if result['p_val'] < 0.05 else "ns"
    print(f"  Significance: {sig}")
# %%

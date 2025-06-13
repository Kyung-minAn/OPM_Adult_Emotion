"""
===============================================
Convert OPM-MEG data into BIDs format (specifically designed for FieldLine data)

    -1 = Set-up - Import relevant modules
    -2 = Define subject and task information
        ## Change to your subject and task information, your file paths and dirs
    -3 = Defining functions for empty room bids conversion
        ## This section has all the custom functions to handle bids conversion of emptyroom recordings.
            # Function to create subject-specific empty room BIDS path
            # Function to find out if emptyroom data is already bids converted
            # Function to save empty room report (saved in hdf5 & html format)
            # Function to convert emptyroom data files into BIDs format
    -4 = Defining functions to generate MNE reports (saved in hdf5 & html format)
        ## This section outlines the different custom functions for generating report data for normal tasks with events, and just resting state data separately
            # Function for saving MNE report for any task data with events triggers
            # Function for saving MNE report for resting state data (without event triggers)
    -5 = Defining function for processing task data MNE BIDs conversion
    -6 = Main execution of running master functions for BIDs formatting emptyroom and task data
            # First empty room data is converted into BIDs format
            # Next all task data is BIDs formatted in order

@author: Alice Waitt (aewaitt), 2025
==============================================  

"""
#%% 1. Set-up - Import relevant modules
import os
import glob
import numpy as np
import mne
from mne_bids import BIDSPath, write_raw_bids
import os.path as op
from mne.report import Report
import re

#%% 2. Define subject and task information
## Change to your subject and task information, your file paths and dirs
subject = '01' #Change subject number here
session = '01' #Change session number if necessary
base_path = rf'Z:\P{subject}\TaskData' #Where all raw data is stored, can change to specify project folder within each subject folder like this example
bids_folder = r'Z:\BIDS' #Separate directory where you want your BIDs data all to be saved
sub_folders = glob.glob(op.join(base_path, 'sub-*')) #FieldLine saves subject data in a sub and date format, this allows us to find this without hardcoding it
data_path = sub_folders[0]
desired_sfreq = 1000  #From 5000Hz original sampling freq. Best to first downsample to 1000Hz prior to preprocessing OPM data
tasks = ['AuditoryMotor','restingstate'] #BIDs convert all files at once, list all your task files here (including resting state if recorded)

#%% 3. Defining functions for empty room bids conversion
## This section has all the custom functions to handle bids conversion of emptyroom recordings.

## Function to create subject-specific empty room BIDS path
def get_emptyroom_bids_path(subject, bids_folder):
    return BIDSPath(
        subject=subject,
        session='emptyroom',
        task='noise',
        datatype='meg',
        root=bids_folder
    )

## Function to find out if emptyroom data is already bids converted
#This can save time if you have already done it and just converting an additional task file.
def emptyroom_bids_exists(subject, bids_folder, er_files):
    meg_path = os.path.join(
        bids_folder, 
        f'sub-{subject}', 
        'ses-emptyroom', 
        'meg'
    )
    
    # If the directory doesn't exist, no files have been converted
    if not os.path.exists(meg_path):
        print(f"MEG path does not exist: {meg_path}")
        return False
    
    # Count expected number of runs
    expected_runs = len(er_files)
    
    # Special case for single file
    if expected_runs == 1:
        # Look for files without run numbers
        no_run_files = glob.glob(os.path.join(meg_path, f"sub-{subject}_ses-emptyroom_task-noise_meg.*"))
        if no_run_files:
            print(f"Found single empty room file without run number: {no_run_files}")
            return True
    
    # For multiple files, check each run number
    if expected_runs > 1:
        found_runs = set()
        for run_num in range(1, expected_runs + 1):
            run_files = glob.glob(os.path.join(meg_path, f"*_run-{run_num:02d}_*"))
            if run_files:
                found_runs.add(run_num)
        
        all_found = len(found_runs) == expected_runs
        print(f"Found {len(found_runs)}/{expected_runs} run-numbered files: {found_runs}")
        return all_found
    
    return False

## Function to save empty room report (saved in hdf5 & html format)
# Note, this function has a customised error message for UoB data using a specific EOG channel.
# Can change this error to another channel if it errors with trying to generate raw plots in report.
def save_emptyroom_report(subject, raw_er, bids_folder, run=None):
    # Create report directory structure
    report_root = op.join(bids_folder, 'derivatives/mne-reports')
    report_folder = op.join(report_root, f'sub-{subject}', 'ses-emptyroom')
    os.makedirs(report_folder, exist_ok=True)
    
    # Define report filenames with run information only if provided
    run_suffix = f"_run-{run:02d}" if run is not None else ""
    report_fname = op.join(report_folder, f'report_sub-{subject}_ses-emptyroom{run_suffix}_raw.hdf5')
    html_report_fname = op.join(report_folder, f'report_sub-{subject}_ses-emptyroom{run_suffix}_raw.html')
    
    # Create and populate the report
    report_title = f'sub-{subject} Empty Room Recording' + (f' Run-{run:02d}' if run is not None else '')
    report = Report(title=report_title)
    
    # Add raw data visualization
    report.add_raw(raw=raw_er, title=f"Empty Room{' Run-' + f'{run:02d}' if run is not None else ''}", psd=True)
    
    # Add power spectrum density with detailed view
    fig_psd = raw_er.plot_psd(fmax=100, tmax=np.inf, picks='meg', show=False)
    report.add_figure(fig=fig_psd, title="Empty Room Power Spectrum" + (f" Run-{run:02d}" if run is not None else ""))
    
    # Add sensor-wise spectral content
    try:
        fig_psd_topomap = raw_er.plot_psd_topomap(show=False)
    except:
        print('s16 localisation error') # Error for UoB-specific data using s16_bz as the EOG channel
        meg_picks = mne.pick_types(raw_er.info, meg=True)
        meg_picks = [pick for pick in meg_picks if raw_er.ch_names[pick] != 's16_bz']
        fig_psd_topomap = raw_er.plot_psd_topomap(picks=meg_picks, show=False)
    
    report.add_figure(fig=fig_psd_topomap, title="Spatial Distribution of Spectral Content")
    
    # Noise metrics
    noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk')
    fig_cov = mne.viz.plot_cov(noise_cov, raw_er.info, show=False)
    report.add_figure(fig=fig_cov, title="Noise Covariance Matrix")
    
    # Save the report
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True)
    
    print(f"Empty room{' Run-' + str(run) if run is not None else ''} report saved to {html_report_fname}")

## Function to convert emptyroom data files into BIDs format
def process_emptyroom(subject, data_path, bids_folder, desired_sfreq):
    # Find empty room files first
    er_files = glob.glob(os.path.join(data_path, f'*_file-emptyroom_raw.fif'))
    if not er_files:
        print("No empty room files found")
        return None
    
    print(f"Found {len(er_files)} empty room files: {[os.path.basename(f) for f in er_files]}")
    
    # Create the base empty room BIDS path (for checking and returning)
    emptyroom_bids_path = get_emptyroom_bids_path(subject, bids_folder)
    
    # Check if all runs already exist in BIDS format
    if emptyroom_bids_exists(subject, bids_folder, er_files):
        print(f"All empty room data runs for subject {subject} already exist in BIDS format")
        return emptyroom_bids_path
    
    # Make sure the main session directory exists
    os.makedirs(os.path.join(
        bids_folder, 
        f'sub-{subject}', 
        'ses-emptyroom',
        'meg'
    ), exist_ok=True)
    
    # Process each empty room file - handle single vs multiple files differently
    single_file = len(er_files) == 1
    
    for i, er_fname in enumerate(er_files):
        run_num = i + 1  # Run numbers start from 1
        print(f"Processing empty room file {i+1}/{len(er_files)}: {os.path.basename(er_fname)}")
        
        try:
            raw_er = mne.io.read_raw_fif(er_fname, preload=True)

            # Check if channel positions exist
            has_positions = False
            try:
                _ = raw_er.plot_sensors(show=False)
                has_positions = True
            except RuntimeError:
                if run_num ==2:
                    print("already converted backup emptyroom, continuing to task mne conversion")
                    break
                else:
                    print("No channel positions found in FIF file, will use different emptyroom file from same day")
                    del(raw_er)
                    single_file = 1
                    er_sub='01' #Change to subject number of empty room file to use
                    er_base_path = rf'Z:\P{er_sub}\TaskData'#Change to base path where file is
                    er_sub_folders = glob.glob(os.path.join(er_base_path, 'sub-*'))
                    er_data_path = er_sub_folders[0]
                    er_file_to_use = glob.glob(os.path.join(er_data_path, f'*_file-emptyroom_raw.fif'))
                    er_fname_to_use = er_file_to_use[0]
                    raw_er = mne.io.read_raw_fif(er_fname_to_use, preload=True)

            # Process empty room data - downsampling the data
            lowpass_freq = desired_sfreq / 4.0
            raw_resampled_er = raw_er.copy().filter(l_freq=None, h_freq=lowpass_freq)
            raw_resampled_er.resample(sfreq=desired_sfreq)
            raw_resampled_er.info["line_freq"] = 50
            
            # Create BIDS path for this file - omit run number if single file
            current_emptyroom_bids_path = BIDSPath(
                subject=subject,
                session='emptyroom',
                task='noise',
                run=None if single_file else run_num,  # Only add run number for multiple files
                datatype='meg',
                root=bids_folder
            )
            
            # Save empty room report - omit run number if single file
            if single_file:
                save_emptyroom_report(subject, raw_resampled_er, bids_folder)
            else:
                save_emptyroom_report(subject, raw_resampled_er, bids_folder, run=run_num)
            
            # Write empty room data to BIDS
            write_raw_bids(
                raw=raw_resampled_er,
                bids_path=current_emptyroom_bids_path,
                allow_preload=True,
                format='FIF',
                overwrite=True
            )
            
            success_msg = "Successfully wrote empty room data to BIDS format" if single_file else f"Successfully wrote empty room run-{run_num:02d} data to BIDS format"
            print(success_msg)
            
            # Free up memory
            del raw_er, raw_resampled_er
        
        except Exception as e:
            error_msg = f"Error processing empty room file: {e}" if single_file else f"Error processing empty room run-{run_num:02d}: {e}"
            print(error_msg)
    
    # Return the base emptyroom_bids_path
    return emptyroom_bids_path

#%% 4. Defining functions to generate MNE reports (saved in hdf5 & html format)
## This section outlines the different custom functions for generating report data for normal tasks with events, and just resting state data separately

## Function for saving MNE report for any task data with events triggers
# Note, this summarises original and final trigger numbers and categories if they are changed
# Can remove duplicate summaries of events if you do not recategorise/use original event IDs and categories
def save_event_report(subject, task, raw, orig_events, events, event_dict, orig_dict, bids_folder, run=None):
    # Create report directory structure
    report_root = op.join(bids_folder, 'derivatives/mne-reports')
    if not op.exists(op.join(report_root , f'sub-{subject}', f'task-{task}')):
        os.makedirs(op.join(report_root , f'sub-{subject}', f'task-{task}'))
    report_folder = op.join(report_root, f'sub-{subject}', f'task-{task}')
    os.makedirs(report_folder, exist_ok=True)
    
    # Define report filenames with run information only if provided
    run_suffix = f"_run-{run}" if run is not None else ""
    report_fname = op.join(report_folder, f'report_sub-{subject}_{task}{run_suffix}_raw.hdf5')
    html_report_fname = op.join(report_folder, f'report_sub-{subject}_{task}{run_suffix}_raw.html')
    
    # Create and populate the report
    report = Report(title=f'sub-{subject}_{task}'+ (f' Run-{run}' if run is not None else ''))
    
    # Add raw data visualization
    report.add_raw(raw=raw, title="Raw", psd=True)
    
    def print_event_summary(events, event_dict=None):
        """
        Print a summary of events by type.
        
        Parameters:
        -----------
        events : ndarray, shape (n_events, 3)
            Events array from mne.find_events
        event_dict : dict or None
            Dictionary mapping event IDs to descriptions
        """
        unique_ids, counts = np.unique(events[:, 2], return_counts=True)
        
        print("\nEvent Summary:")
        print("--------------")
        for id_val, count in zip(unique_ids, counts):
            name = event_dict.get(id_val, f"Unknown-{id_val}") if event_dict else f"ID-{id_val}"
            print(f"{name} (ID: {id_val}): {count} occurrences")

    # Print event summaries to console (optional)
    # This is more relevant if you had to separate out triggers to redefine categories etc, so can customise here.

    print("\nOriginal Events:")
    print_event_summary(orig_events, orig_dict)
    print("\nProcessed Events:")
    print_event_summary(events, event_dict)
    
    # Add event plot to report
    if len(events) > 0:
        fig_orig = mne.viz.plot_events(orig_events, sfreq=raw.info["sfreq"], 
                                    first_samp=raw.first_samp, 
                                    event_id=orig_dict)
        report.add_figure(fig=fig_orig, title=f"Original trigger events - {task}")
        
        fig_proc = mne.viz.plot_events(events, sfreq=raw.info["sfreq"], 
                                    first_samp=raw.first_samp, 
                                    event_id=event_dict)
        report.add_figure(fig=fig_proc, title=f"Processed trigger events - {task}")
    else:
        # Add a note if no events found
        report.add_html(f"<p>No events found for task {task}</p>")
    
    # Save the report
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True)
    
    print(f"Report saved to {html_report_fname}")

## Function for saving MNE report for resting state data (without event triggers)
# Can customise outputs for report, some example plots to use in report here
def save_resting_report(subject, raw, bids_folder, run=None):
    # Create report directory structure
    report_root = op.join(bids_folder, 'derivatives/mne-reports')
    report_folder = op.join(report_root, f'sub-{subject}', 'task-restingstate')
    os.makedirs(report_folder, exist_ok=True)
    
    # Define report filenames with run information only if provided
    run_suffix = f"_run-{run}" if run is not None else ""
    report_fname = op.join(report_folder, f'report_sub-{subject}_restingstate{run_suffix}_raw.hdf5')
    html_report_fname = op.join(report_folder, f'report_sub-{subject}_restingstate{run_suffix}_raw.html')
    
    # Create and populate the report
    report = Report(title=f'sub-{subject} Resting State'+ (f' Run-{run}' if run is not None else ''))
    
    # Add raw data visualization with more detailed PSD plots
    report.add_raw(raw=raw, title="Resting State Recording", psd=True)
    
    # Add specialized visualizations for resting data
    # For example, power spectrum density with more bands
    fig_psd = raw.plot_psd(fmax=100, tmax=np.inf, picks='meg', show=False)
    report.add_figure(fig=fig_psd, title="MEG Power Spectrum Density")

    # Add sensor-wise spectral content
    fig_psd_topomap = raw.plot_psd_topomap(show=False)
    report.add_figure(fig=fig_psd_topomap, title="Spatial Distribution of Spectral Content")
    
    # You might also want to add sensor plots
    fig_sensors = raw.plot_sensors(show=False)
    report.add_figure(fig=fig_sensors, title="Sensor Layout")
    
    # Save the report
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True)
    
    print(f"Resting state report saved to {html_report_fname}")

#%% 5. Defining function for processing task data MNE BIDs conversion
# Update task specific event and dictionary info for your data
# Update specific digital trigger channel in OPM file

def process_task(subject, session, task, data_path, bids_folder, desired_sfreq):
    print(f"\nProcessing task: {task}")
    
    # Find all task files with a more specific approach to avoid duplicates
    standard_files = glob.glob(os.path.join(data_path, f'*_file-{task}_raw.fif'))
    numbered_files = glob.glob(os.path.join(data_path, f'*{task}[0-9][0-9]*_raw.fif'))
    task_files = list(set(standard_files + numbered_files))
    
    if not task_files:
        print(f"No files found for task: {task}")
        return
    
    # Sort the files to ensure correct order
    task_files = sorted(task_files)
    total_runs = len(task_files)

    print(f"Found {total_runs} files for task {task}: {[os.path.basename(f) for f in task_files]}")
    
    # Process each file
    for idx, raw_fname in enumerate(task_files):
        base_name = os.path.basename(raw_fname)
        is_first_run = (idx == 0)
        is_last_run = (idx == total_runs - 1)
        
        # Extract run number
        run_match = re.search(f'{task}(\d+)', base_name)
        if run_match:
            run = run_match.group(1)
            if len(run) == 1:
                run = f"0{run}"
        else:
            run = f"{idx+1:02d}"
        
        print(f"Loading task file: {base_name} (Run: {run}, First: {is_first_run}, Last: {is_last_run})")
        
        # Load and process the data
        try:
            # First try normal loading
            raw = mne.io.read_raw_fif(raw_fname, preload=True)
        except ValueError as e:
            error_msg = str(e)
            print(f"Error encountered loading in raw fif: {error_msg}")
                
        # Find events using error handling
        try:
            orig_events = mne.find_events(raw, stim_channel='di32') #Change to relevant digital trigger channel in system
            print(f"Found {len(orig_events)} events for task {task}, run {run}")
        except Exception as e:
            print(f"Error finding events: {e}")
            orig_events = np.array([]).reshape(0, 3)
        
        # Change to any task specific event trigger/dictionary information here
        if task == 'AuditoryMotor':
            events = orig_events
            event_dict = {"stimulus": 4, "button": 16}
            orig_dict = {"stimulus": 4, "button": 16}
        elif task == 'restingstate':
            events = orig_events if len(orig_events) > 0 else None
            event_dict = {} if events is None else {"event": 1}
            orig_dict = {}
        else:
            # Default case
            events = orig_events
            event_dict = {f"event_{val}": val for val in set(orig_events[:, 2])} if len(orig_events) > 0 else {}
            orig_dict = {}

         # Ensure events is always properly initialized
        if 'events' not in locals() or events is None or len(events) == 0:
            print(f"No valid events found for task {task}. Creating empty events array.")
            events = np.array([]).reshape(0, 3)  # Create empty events array with correct shape

        # After finding events but before resampling
        if task =='restingstate':
            if total_runs == 1:
                save_resting_report(subject, raw, bids_folder)
            else:
                save_resting_report(subject, raw, bids_folder, run=run)
        else:
            if total_runs == 1:
                save_event_report(subject, task, raw, orig_events, events, event_dict, orig_dict, bids_folder)
            else:
                save_event_report(subject, task, raw, orig_events, events, event_dict, orig_dict, bids_folder, run=run)

        # Resample data
        current_sfreq = raw.info['sfreq']
        lowpass_freq = desired_sfreq / 4.0
        
        # Downsample data before BIDs conversion
        raw_resampled = raw.copy().filter(l_freq=None, h_freq=lowpass_freq)
        raw_resampled.resample(sfreq=desired_sfreq)
        
        # Always create events_resampled, even if empty
        if len(events) > 0:
            events_resampled = events.copy()
            events_resampled[:, 0] = (events_resampled[:, 0] * desired_sfreq/current_sfreq).astype(int)
        else:
            events_resampled = np.array([]).reshape(0, 3)  # Empty array with correct shape
        
        # Set BIDS info
        raw_resampled.info["line_freq"] = 50
        
        er_files = glob.glob(os.path.join(data_path, f'*_file-emptyroom_raw.fif'))
        single_file = len(er_files) == 1
        emptyroom_bids_path_to_use = BIDSPath(
            subject=subject,
            session='emptyroom',
            task='noise',
            run=None if single_file else 1,  # Specify the run of emptyroom rec you want to use, otherwise default is run 1
            datatype="meg",
            root=bids_folder
        )
        
        # Make sure directory exists
        os.makedirs(os.path.join(bids_folder, f'sub-{subject}', f'ses-{session}'), exist_ok=True)
        
        # Create BIDSpath
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            task=task,
            run=run,
            datatype="meg",
            root=bids_folder
        )
        
        print(f"BIDS path: {bids_path}")
        
        # Write to BIDS format with careful error handling
        try:
            # Always pass events but handle empty case
            if len(events_resampled) > 0:
                # Case with events
                write_raw_bids(
                    raw=raw_resampled,
                    bids_path=bids_path,
                    events=events_resampled,
                    event_id=event_dict,
                    empty_room=emptyroom_bids_path_to_use,
                    format='FIF',
                    allow_preload=True,
                    overwrite=True,
                )
            else:
                # Empty events case - pass empty arrays instead of None
                write_raw_bids(
                    raw=raw_resampled,
                    bids_path=bids_path,
                    events=events_resampled,  # Pass empty array
                    event_id={},  # Empty dict
                    empty_room=emptyroom_bids_path_to_use,
                    format='FIF',
                    allow_preload=True,
                    overwrite=True,
                )
                
            print(f"Successfully wrote data for task: {task}, run: {run}")
            
        except Exception as e:
            print(f"Error during BIDS conversion: {e}")
            
            # Check if data was written despite the error
            bids_file = os.path.join(
                bids_folder, 
                f'sub-{subject}', 
                f'ses-{session}',
                'meg',
                f'sub-{subject}_ses-{session}_task-{task}_run-{run}_meg.fif'
            )
            
            if os.path.exists(bids_file):
                file_size = os.path.getsize(bids_file)
                print(f"Data file exists ({file_size/1024/1024:.2f} MB) despite error.")
                print("Considering this task successfully converted.")
            else:
                print(f"Data file was not created. Conversion failed.")

        # Free up memory
        del raw, raw_resampled
        if events is not None and len(events) > 0:
            del events, events_resampled

#%% 6. Main execution of running master functions for BIDs formatting emptyroom and task data
## 'process_emptyroom' and 'process_task' goes through all sub functions defined previously 
## (e.g.the file checks, making directories, events, mne reports etc.)
    
## First empty room data is converted into BIDs format
# This has contingency for checking if it has been done previously if BIDs formatting other task files at a later date
emptyroom_bids_path = process_emptyroom(subject, data_path, bids_folder, desired_sfreq)
    
## Next all task data is BIDs formatted in order
# Process each task specified in list in Section 2
for task in tasks:
    process_task(subject, session, task, data_path, bids_folder, desired_sfreq)

print("All tasks processed.")

# %%

# OPM_Adult_Emotion
This repository contains Python scripts designed as part of the analysis 
pipeline for data collected as part of the BBSRC Pioneer Project.

OPM-MEG data were recorded using a FieldLine system,
collected in a group of healthy adult participants.

These scripts are specifically designed for processing and analysing FieldLine data.

## Scripts
BIDs_formatting
- opm_bids_converter.py - Convert OPM-MEG data to BIDS format

Preprocessing
- preprocessing_ann_ica.py - Preprocessing OPM-MEG Task Data: Filter, Annotate and ICA

Experiment_Tasks
- Adult_OPM_experiment contains all the task scripts used for the adult experiment: Passive voice listening task (PassiveVoiceTaskAW_v004.m), Passive voice listening task with button press (PassiveVoiceTaskAW_v004_BUTTONS.m), Auditory evoked response (AER) task (OPM_AER_simple_sound_digitalTrigger_AudioPixx_AW.m) and AER with button press (OPM_AER_Button_AW.m).

- Infant_OPM_experiment contains all the task scripts used for infant data collection: Passive voice listening task based from the adult task (PassiveVoiceTaskInfantAW_final.m) , and white noise (whitenoise_datapixx_script_final.m).

## Requirements

- Python 3.8+
- MNE-Python
- MNE-BIDS
- NumPy, SciPy, Matplotlib
- MATLAB 2019b for running experiment task scripts

See requirements.txt for specific versions.

## Usage
These scripts would require customisation for your specific:

- File paths and directory structure
- OPM system configuration
- Task designs and event triggers
- Preprocessing parameters

## Notes

- Developed for a FieldLine OPM system, so would require adaptation 
if using other systems
- BIDS converter handles empty room recordings and multiple task types
- MNE reports are generated for quality control and easy reference

## Author

Alice Waitt (aewaitt), 2025

## Study PI
Kyung-min An


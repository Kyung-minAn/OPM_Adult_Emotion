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

## Requirements

- Python 3.8+
- MNE-Python
- MNE-BIDS
- NumPy, SciPy, Matplotlib

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


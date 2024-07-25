# EEGpykit

An MNE EEG Preprocessing and Analysis Library using Python

This project contains various scripts with the aim of preprocessing electroencephalography (EEG) data in a quick and consistent manner.
Many existing EEG preprocessing and analysis methods work using a graphical user interface and many manual steps. 
Researchers are often encouraged to make subjective decisions when rejecting noisy channels or periods of time in a time-consuming manual process.
This can lead to issues of reproducibility and depending on who preprocessed the data it can make the difference between significant or non-significant findings. 

To avoid these issues this project utilises the FASTER protocol for preprocessing EEG data,
in a way that is fully automated and reproducible.
In the spirit of automation and utilising available data to the fullest,
this project also provides tools for the analysis of EEG data using Multi Variate Pattern Analysis (MVPA),
allowing for the discrimination between activity in different experimental conditions over time.

This project has only been tested with our EEG setup and needs, and several of the analysis scripts are written for specific projects.
Therefore, your mileage may vary when using EEGpykit for your own uses!

## How to Use - Preprocessor
- See the EEG Analysis Workshop 2024 presentation in documents/ for a different set of usage instructions

### Ingredients
#### Types of files you should have already:
- **Raw EEG** data files (.fif or .cnt)
- **Trigger file** (.trg)
- **Behavioural data** (.csv)
- **Montage file** (.xyz)

#### Files you need to create:
- **Participant list** (.csv) file with the header: 

| ppt_num | data_path        | extra_events_path | pid  | EOG_channel | status | ref_channel | raw_format   |
|:--------|:-----------------|:------------------|------|:------------|:-------|:------------|:-------------|
| 1       | path/to/data/dir | path/to/.csv      | PPT1 | e.g VEOG    |        | e.g Cz      | .cnt or .fif |

- **trigger labels** (.csv)
	- This file describes all the triggers in the EEG data, what their numerical value refers to, and whether you want to keep them
	- Often you will want to create new labels and events based on the participant's behavioural data later, so this file will usually be superceded by that one
	- file with the header: 

| label          | value | active |
|:---------------|:------|:-------|
| event1/view    | 11    | 1      |
| event2/view    | 12    | 1      |
| event1/respond | 21    | 1      |
| event2/respond | 22    | 1      |

- A preprocessing **output directory**
	- e.g 'output_directory/'
- **extra_events** (.csv)
	- a headerless csv with two columns
		- event labels (column 1)
			- multiple labels can be assigned to a single event when separated by a '/'
		- event start time (column 2)
			- these need to match trigger times in the .trg file
		- a third column with event end time may also be allowable
	- this file allows you to create more events based on behavioural data than you could otherwise have with just the trigger file
	- make one for each data recording and give the path to it in the extra_events_path column of the Participant list .csv

### Starting the preprocessing gui:
- Assuming you already installed the modules listed in requirement.txt:
```sh
cd /path/to/EEGpykit/ 
python cli.py
```
- A window should open with the following entry fields:
  - Participant List File
    - path to **Participant list**
  - Output DB: 
    - path to **output directory**
  - Trigger labels:
    - path to **trigger labels**
  - tmin: 
    - negative start time of epoch baseline in seconds
  - bmax:
    - end time of epoch baseline (usually 0) in seconds
  - tmax: 
    - end time of epoch in seconds
  - additional_events_fname
    - leave this blank if using extra_events_path column in **Participant list** file
- Click submit to start preprocessing data

## How to use - MVPAnalysis
```python
from MVPA.MVPAnalysis import MVPAnalysis

MVPAnalysis(files, var1_events=[f'obvious'], var2_events=[f'prior'],
            excluded_events=['Rate', 'Missed'], scoring="roc_auc",
            output_dir=Path(output_dir, f'Naives/across_session/obvious_vs_priors'),
            indiv_plot=False, epochs_list=epochs_list, extra_event_labels=extra, jobs=-1)
```

## Main Scripts
### preprocessor.py
- Imports read_antcnt to read in data from eeg_data/
- Automatically preprocesses data to reduce noise and artifacts using MNE and the FASTER protocol

### cli.py
- Command line interface for preprocessor.py
 
> #### EEG PRE-PROCESSING STEPS
> 1. Filtering all EEG data with a low bandpass keep 0.5 to 40 hz
> 2. Resampling data and keeping samples up to 200 Hz
> 3. Splitting into epochs from -700 ms pre-target to 1000 post target
> 4. Marking bad channels (by variance, correlation, hurst, kurtosis and line noise)
> 5. Excluding dead channels 
> 6. Interpolating data across neighbouring channels to recover data in bad channels 
> 7. Marking bad epochs (using amplitude, variance & deviation)
> 8. Running an ICA- to remove eye-blink signals and any components that correlate with EOG or any signs of movement 
> 9. Filtering out channels per epoch if bad and interpolating across those found to be bad 
> 10. Filtering the signal for the 50 Hz noise (low pass)
> 11. Baseline correcting the signal in every channel - average of signal in single electrode in the range -700 to -200 ms before target onset [-200 to 0 ms- excluded since it contains a audio warning signal), then subtract this from every epoch range of -700 pre onset to 1000 ms post target onset [1000=500 mamogram +500 mask]

### analysis.py
- Reads epochs pickles from MNE_processing_db/ for analysing and plotting.
- Analyses data using MNE by condition, participant, and session.
- Three different analysis classes are defined:
  - Single participant analysis
  - Singe participant across session analysis
  - Group across session analysis
- Main analysis is Spatio Temporal Clustering

## MVPA
### MVPAnalysis.py
> #### MVPA Analysis Outline
> 1. Load preprocessed and epoched EEG data
>    * Each individual’s data is a 3d array of epochs x channels x times
>    * Filter data by matching epochs that belong to two different conditions
>    * Sort channels in row-major snakelike ordering for plotting purposes
>       * On 2d plots this groups data from neighbouring electrodes together
> 
> 2. Temporal decoding with MVPA
>    * Define a sliding estimator with a support vector machine (SVM)
>       * Weight classes by size so they are ‘balanced’
>          * n_samples / (n_classes * np.bincount(y))
>          * Otherwise, the SVM can tell the classes apart just from which is the more numerous one
>       * The sliding estimator fits a SVM classifier at every time point in the 2d epoch x channel data
>          * Sample rate was 200Hz, so each time point is equal to one measurement every 5 milliseconds
>       * Therefore, eeg data in every channel and epoch is used to try to tell the classes apart
>    * Cross validate scores with Leave One Group Out method (LOGO)
>       * Training and test data is split by participant (a participant can have multiple sessions worth of data)
>       * One participant is used to test the model and the others are used to train the sliding estimator SVMs
>       * This process is repeated so that each of the participants is the test data once
>    * Estimator performance for each split is scored using ROC AUCs for each time point
>       * Tells us how well it could correctly vs incorrectly discriminate between the classes at each time point
>       * E.g a value of 0.5 is as good as guessing while a value of 1.0 is a perfect classifier
>          * Lower than 0.5 would be a classifier that has learnt the classes the wrong way, but is still predicting a difference
>    * Average the AUC scores across all the splits to get an overall score for the whole dataset
>    * Plot data on line plot
>       * Calculate a moving average on the scores to smooth data points
> 
> 3. Create activity map plots
>    * Average data across evokeds (condition x evokeds x channels x times) to make 2d data for plotting
>       * Subtract the data of each condition
>    * 2 sample spatio-temporal cluster test on data (condition x evokeds x channels x times)
>    * Plot heatmap with 2d difference data
>       * Set values to zero if they are not in a significant cluster
>    * Make topographical plots over 4 timepoints
> 
> 4. Repeat process for each comparison

## utils
### read_antcnt.py
- Reads proprietary ANT Neuro .cnt files
  - ANT Neuro's ASALAB software outputs EEG data as .cnt files
  - This format is not natively supported by MNE (yet)
    - Keep an eye on this issue to see if native python support has been implemented:
    - https://github.com/mne-tools/mne-python/issues/3609
  - Requires pyeep module and pyeep.so (shared object file)
    - This is the compiled c bindings for the reader
    - Only works for the computer it was compiled on
  - After this step we store the MNE raw object as a .pickle file

### Libeep
- To read ANT cnt files You will need to replace the 'pyeep.so' with your own compiled version from here:
  - https://github.com/neuromti/tool-libeep/tree/master
  - or https://github.com/agricolab/tool-libeep
- Run these commands to (hopefully) compile libeep for python:

```bash
cd tool-libeep
mkdir build
cd build/
cmake .. -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))") -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
make
```
- Then replace the '.so' file in libeep/ with the new one from build/
### ANTcnt_to_fif.py
- Convert ANTcnt files to fif to make loading eeg quicker and easier for other people
- File size does increase a little though.

### dialogue_box.py
- Defines the dialog window used by preprocessor.py

## Other Important Files
- waveguard64.xyz
  - The montage file used by our EEG caps in xyz coordinates
  - Does not include coordinates for reference electrode
    - Those are added in preprocessor.py

# Contact
Lyndon Rakusen - [mail@lyndonrakusen.com](mailto:mail@lyndonrakusen.com)

Project Link: [github.com/llr510/EEGpykit](https://github.com/llr510/EEGpykit)

# Acknowledgments
- Nolan H., Whelan R. and Reilly RB. for inventing the FASTER protocol
- Marijn van Vliet for an example Python implementation of FASTER: https://github.com/wmvanvliet/mne-faster/
- Benedikt Ehinger for writing an ANT Neuro .cnt file reader: https://github.com/behinger/mne_tools

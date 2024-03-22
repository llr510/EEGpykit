# EEGpykit: An MNE EEG Preprocessing and Analysis Project
## Main Scripts
### preprocessor.py
- Imports read_antcnt to read in data from eeg_data/
- Automatically preprocesses data to reduce noise and artifacts using MNE and the FASTER protocol
### cli.py
- Command line interface for preprocessor.py
#### EEG PRE-PROCESSING STEPS
1. Filtering all EEG data with a low bandpass keep 0.5 to 40 hz
2. Resampling data and keeping samples up to 200 Hz
3. Splitting into epochs from -700 ms pre-target to 1000 post target
4. Marking bad channels (by variance, correlation, hurst, kurtosis and line noise)
5. Excluding dead channels 
6. Interpolating data across neighbouring channels to recover data in bad channels 
7. Marking bad epochs (using amplitude, variance & deviation)
8. Running an ICA- to remove eye-blink signals and any components that correlate with EOG or any signs of movement 
9. Filtering out channels per epoch if bad and interpolating across those found to be bad 
10. Filtering the signal for the 50 Hz noise (low pass)
11. Baseline correcting the signal in every channel - average of signal in single electrode in the range -700 to -200 ms before target onset [-200 to 0 ms- excluded since it contains a audio warning signal), then subtract this from every epoch range of -700 pre onset to 1000 ms post target onset [1000=500 mamogram +500 mask]
### analysis.py
- Reads epochs pickles from MNE_processing_db/ for analysing and plotting.
- Analyses data using MNE by condition, participant, and session.
- Three different analysis classes are defined:
  - Single participant analysis
  - Singe participant across session analysis
  - Group across session analysis
- Main analysis is Spatio Temporal Clustering
## MVPA
### MVPA_analysis.py
EEGpykit - MVPA Analysis Outline
1. Load preprocessed and epoched EEG data
   * Each individual’s data is a 3d array of epochs x channels x times
   * Filter data by matching epochs that belong to two different conditions
   * Sort channels in row-major snakelike ordering for plotting purposes
      * On 2d plots this groups data from neighbouring electrodes together

2. Temporal decoding with MVPA
   * Define a sliding estimator with a support vector machine (SVM)
      * Weight classes by size so they are ‘balanced’
         * n_samples / (n_classes * np.bincount(y))
         * Otherwise the SVM can tell the classes apart just from which is the more numerous one
      * The sliding estimator fits a SVM classifier at every time point in the 2d epoch x channel data
         * Sample rate was 200Hz, so each time point is equal to one measurement every 5 milliseconds
      * Therefore eeg data in every channel and epoch is used to try to tell the classes apart
   * Cross validate scores with Leave One Group Out method (LOGO)
      * Training and test data is split by participant (a participant can have multiple sessions worth of data)
      * One participant is used to test the model and the others are used to train the sliding estimator SVMs
      * This process is repeated so that each of the participants is the test data once
   * Estimator performance for each split is scored using ROC AUCs for each time point
      * Tells us how well it could correctly vs incorrectly discriminate between the classes at each time point
      * E.g a value of 0.5 is as good as guessing while a value of 1.0 is a perfect classifier
         * Lower than 0.5 would be a classifier that has learnt the classes the wrong way, but is still predicting a difference
   * Average the AUC scores across all the splits to get an overall score for the whole dataset
   * Plot data on line plot
      * Calculate a moving average on the scores to smooth data points

3. Create activity map plots
   * Average data across evokeds (condition x evokeds x channels x times) to make 2d data for plotting
      * Subtract the data of each condition
   * 2 sample spatio-temporal cluster test on data (condition x evokeds x channels x times)
   * Plot heatmap with 2d difference data
      * Set values to zero if they are not in a significant cluster
   * Make topographical plots over 4 timepoints

4. Repeat process for each comparison


## utils
### read_antcnt.py
- Reads proprietary ANT Neuro .cnt files.
  - ANT Neuro's ASALAB software outputs EEG data as .cnt files.
  - This format is not natively supported by MNE (yet).
  - Requires pyeep and pyeep.so.
    - This is the compiled c bindings for the reader.
    - Only works for the computer it was compiled on.
  - After this step we store the MNE raw object as a .pickle file.
### Libeep
- To read ANT cnt files You will need to replace the pyeep.so with your own compiled version from here:
  - https://github.com/neuromti/tool-libeep/tree/master
- Run these commands to compile libeep for python:
```bash
cd tool-libeep
mkdir build
cd build/
cmake .. -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
make
```
- Then replace the so file in libeep/ with the new one in build/
### ANTcnt_to_fif.py
- Convert ANTcnt files to fif to make loading eeg quicker and easier for other people. 
- File size does increase a little though.
### dialogue_box.py


## Other Important Files
- waveguard64.xyz
  - The montage file for our EEG caps in xyz coordinates.
  - Does not include coordinates for reference electrode.
  - Those are added in preprocessor.py


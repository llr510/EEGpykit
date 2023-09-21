# EEGpykit: An MNE EEG Preprocessing and Analysis Project
## Key Scripts:
### EEGTraining_testSessions.py
- Creates updated trigger files for data based on behavioural performance
### read_antcnt.py
- Reads proprietary ANT Neuro .cnt files.
  - ANT Neuro's ASALAB software outputs EEG data as .cnt files.
  - This format is not natively supported by MNE.
  - Requires pyeep and pyeep.so.
    - This is the compiled c bindings for the reader.
    - Only works for the computer it was compiled on.
    - **Todo: Compiling instructions coming soon**
  - After this step we store the MNE raw object as a .pickle file.
### preprocessor.py
- Imports read_antcnt to read in data from eeg_data/
- Automatically preprocesses data to reduce noise and artifacts using MNE and the FASTER protocol

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
## Other Important Files
- waveguard64.xyz
  - The montage file for our EEG caps in xyz coordinates.
  - Does not include coordinates for reference electrode.
  - Those are added in preprocessor.py


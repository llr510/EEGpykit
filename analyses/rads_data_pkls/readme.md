# README

- Contains pickled EEG data for use with machine learning
    - Data for 5 reading radiologists from York Hospital
- Data is stored in a python dictionary with the keys: {'data': X, 'label': y, 'key': {0: "condition1", 1: "condition2"}}
    - 'data' is a 3d array of epochs x channels x times
        - Epochs are trials with a couple of hundred values
        - Channels are individual electrodes
            - There are 64 channels
        - Times are EEG samples in microvolts
            - There should be 481 samples
            - Sample rate is 200 Hz 
    - 'label' is a 1d array of truth values for the epoch dimension of the data array
        - values are 1s and 0s
    - 'key' is the condition key for the label array

## normal_v_abnomal
- Contains trials where the stimulus presented was a normal or a abnormal mammogram

## normal_v_global
- Contains trials where the stimulus presented was a normal or a contralateral/prior mammogram
- Priors are mammograms from clients who went on to develop cancer in around 3 years time
- Contalaterals are left or right mammograms taken from abnormal cases, but don't have any abnormalities

## case_balanced
- Condition with most trials have been randomly subsambled so that both conditions have the same number of trials

## case_imbalanced
- Contains all trials for each condition

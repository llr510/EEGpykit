from pathlib import Path

import mne
from mne import io


def read_stephens_data(header_path, montage_path=None, ref_channel='Cz', verbose=False):
    """
    Custom data loading function for data collected with BrainVision and a custom montage.

    @param header_path: path to brainvision .vhdr file
    @param montage_path: optional path to custom EEG electrode mapping file
    @param ref_channel: name of online reference channel used during data collection
    @param verbose: whether to block and display raw data or not
    @return: raw data, events, channel information
    """
    # If an eeg data file doesn't load via the header file there's a chance that a file has been moved and renamed
    raw = io.read_raw_brainvision(header_path, preload=True, verbose=verbose)
    raw.set_channel_types({'VEOG': 'eog', 'Photo': 'misc'}, verbose=verbose)
    # In the custom montage the equivalent of PO2 was mysteriously named P20...
    mne.channels.rename_channels(raw.info, {'P20': 'PO2'}, verbose=verbose)
    # When recording one electrode is used as a reference.
    # Add that channel back with values of 0,
    # so that when we that change the reference later we get all that data back
    mne.add_reference_channels(raw, ref_channels=ref_channel, copy=False)
    # Get a suitable map of electrodes
    #TODO Find one that's more accurate to what Stephen used
    if not montage_path:
        montage = mne.channels.make_standard_montage('standard_1020')
    else:
        montage = mne.channels.read_custom_montage(montage_path)
    raw.set_montage(montage)
    # Get EEG and EOG channels
    picks = mne.pick_types(raw.info, eeg=True, eog=True)
    # Get reference electrode index
    ref = mne.pick_channels(raw.info['ch_names'], [ref_channel], verbose=verbose)[0]
    # Apply reference electrode coordinates to other electrodes ref coords
    # Useful for mne FASTER bad channel detection
    for ch in picks:
        raw.info['chs'][ch]['loc'][3:6] = raw.info['chs'][ref]['loc'][:3]
    # Get event information from annotations
    events = mne.events_from_annotations(raw, verbose=verbose)

    if verbose:
        print(raw.info)
        raw.plot(block=True)
        raw.plot_sensors(block=True, show_names=True)

    return raw, events, picks

if '__main__' in __name__:
    wd = Path('/Users/lyndonrakusen/Documents/Data/OSF Uploads for '
                       'Lyndon/EEG_Attentional_Template_SSM_NCI_Experiment1_K99_40/SSM_EEG/')
    header_paths = wd.glob('*.vhdr')
    for path in header_paths:
        read_stephens_data(path, verbose=True)
        quit()

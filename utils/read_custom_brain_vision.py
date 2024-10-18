import mne
from mne import io


def read_stephens_data(header_path, montage_path=None, verbose=False, ref_channel='Cz'):
    """
    Custom data loading function for data collected with BrainVision and a custom montage

    @param header_path:
    @param montage_path:
    @param verbose:
    @param ref_channel:
    @return:
    """
    raw = io.read_raw_brainvision(header_path, preload=True)
    raw.set_channel_types({'VEOG': 'eog', 'Photo': 'misc'})
    # In the custom montage the equivalent of PO2 was mysteriously named P20...
    mne.channels.rename_channels(raw.info, {'P20': 'PO2'})
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
    ref = mne.pick_channels(raw.info['ch_names'], [ref_channel])[0]
    # Apply reference electrode coordinates to other electrodes ref coords
    for ch in picks:
        raw.info['chs'][ch]['loc'][3:6] = raw.info['chs'][ref]['loc'][:3]
    # Get events from annotations
    events = mne.events_from_annotations(raw)

    if verbose:
        print(raw.info)
        raw.plot(block=True)
        raw.plot_sensors(block=True, show_names=True)

    return raw, events, picks

if '__main__' in __name__:

    read_stephens_data('/Users/lyndonrakusen/Documents/Data/OSF Uploads for Lyndon/EEG_Attentional_Template_SSM_NCI_Experiment1_K99_40/SSM_EEG/SID1.vhdr',
                   verbose=True)

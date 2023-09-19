"""
====================================
EEG artifact correction using FASTER
====================================
Modified script by Rakusen L.

References
----------
[1] Nolan H., Whelan R. and Reilly RB. FASTER: fully automated
    statistical thresholding for EEG artifact rejection. Journal of
    Neuroscience Methods, vol. 192, issue 1, pp. 152-162, 2010.
"""
from pathlib import Path, PosixPath
import mne
from mne import io
# pip install https://github.com/wmvanvliet/mne-faster/archive/refs/heads/main.zip
from mne_faster import (find_bad_channels, find_bad_epochs,
                        find_bad_components, find_bad_channels_in_epochs)
from utils.read_antcnt import read_raw_antcnt, read_events_trg
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import Union
from utils.dialogue_box import dialogue_window
import sys

sys.path.append('utils')


def resample_and_bandpass_raw(raw_fname, ref_channel, eog_channel, montage,
                              sfreq=200, hfreq=40, lfreq=0.5, plotting=False, report=None):
    """
    Loads raw EEG data from ANT CNT or other formats, bandpasses it, and saves a report with figures.
    Our data used EOG for blink detection, but had Vertical and Horizontal EOG mixed up sometimes.
    Loads the correct one and renames it VEOG.

    :param raw_fname: str or PosixPath
    :param ref_channel: str
    :param eog_channel:
    :param montage: DigMontage
    :param sfreq: int
    :param hfreq: float
    :param lfreq: float
    :param plotting: bool
    :param report:
    :return: RawANTCNT,
    """
    event_fname = Path(raw_fname).with_suffix('.trg')
    assert raw_fname.exists()
    if raw_fname.suffix != '.cnt':
        raw = io.read_raw(str(raw_fname), preload=True)
    else:
        raw = read_raw_antcnt(str(raw_fname), preload=True, eog=[eog_channel])
    events = read_events_trg(event_fname)

    raw.info['bads'] = []  # bads are going to be detected automatically

    if eog_channel == 'VEOG':
        drop_eog_channel = 'HEOG'
        raw.drop_channels([drop_eog_channel])
    else:
        drop_eog_channel = 'VEOG'
        raw.drop_channels([drop_eog_channel])
        mne.rename_channels(raw.info, {'HEOG': 'VEOG'})

    # In this example, we restrict analysis to EEG channels only to save memory and
    # time. However, these methods also work for MEG data.
    raw = raw.pick_types(meg=False, eeg=True, eog=True)

    # Set montage
    raw.set_montage(montage)

    # Keep whatever EEG reference the amplifier used for now. After the data is
    # cleaned, we will re-reference to an average reference.
    raw, _ = io.set_eeg_reference(raw, [ref_channel])

    report.add_raw(raw=raw, title='Raw before filtering')
    picks = mne.pick_types(raw.info, eeg=True, eog=True)
    raw.filter(l_freq=lfreq, h_freq=hfreq, method='iir', picks=picks)

    # resample to 200hz
    raw, events = raw.resample(sfreq=sfreq, events=events)

    if plotting:
        raw.plot_sensors(show_names=True, block=False)
        raw.plot(events=events, order=picks, n_channels=len(picks), block=False)

    # Get reference electrode index
    ref = mne.pick_channels(raw.info['ch_names'], [ref_channel])[0]
    # Apply reference electrode coordinates to other electrodes ref coords
    for ch in picks:
        raw.info['chs'][ch]['loc'][3:6] = raw.info['chs'][ref]['loc'][:3]

    # convert meas_date (a tuple of seconds, microseconds) into a float:
    meas_date = raw.info['meas_date']
    orig_time = raw.annotations.orig_time
    assert meas_date == orig_time

    report.add_raw(raw=raw, title='Raw after filtering')
    return raw, events, picks, report


def preprocess_with_faster(raw, events, event_ids, picks, tmin=-0.5, bmax=0, tmax=1, plotting=False,
                           report=None):
    """

    @param raw:
    @param events: array of events
    @param event_ids:
    @param picks:
    @param tmin:
    @param bmax:
    @param tmax:
    @param plotting:
    @param report:
    @return:
    """
    plt.close('all')
    logdict = {}
    event_repeated = 'merge'
    print(type(raw))
    # Construct epochs. Note that we also include EOG channels.
    epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, baseline=None,
                        preload=True, picks=picks, event_repeated=event_repeated, on_missing='warn')
    # Compute evoked before cleaning, using an average EEG reference
    epochs_before = epochs.copy()
    epochs_before.set_eeg_reference('average')

    # set baseline from start of epoch to before event happens
    baseline = (None, bmax)
    epochs_before.apply_baseline(baseline)
    report.add_epochs(epochs=epochs_before, title='Epochs before FASTER')

    evoked_before = epochs_before.average()
    if plotting:
        evoked_before.plot()

    ###############################################################################
    # Clean the data using FASTER
    # epochs = run_faster(epochs, thres=3, copy=False)
    print('# Clean the data using FASTER')
    epochs_before.set_eeg_reference('average')
    # Step 1: mark bad channels
    print('# Step 1: mark bad channels')
    epochs.info['bads'] = find_bad_channels(epochs, eeg_ref_corr=True)

    logdict['interpolated_channels'] = len(epochs.info['bads'])
    if len(epochs.info['bads']) > 0:
        epochs.interpolate_bads()

    try:
        print("Step 5: Find and interpolate dead channels that FASTER misses")
        epochs.set_eeg_reference('average')
        epochs, bads = find_dead_channels(epochs, plot=plotting)
        logdict['Extra Interpolated Channels'] = bads
        epochs.set_eeg_reference(['Cz'])
    except ValueError:
        print("Couldn't run psd_array_welch. Debugging needed.")

    # Step 2: mark bad epochs
    print('# Step 2: mark bad epochs')
    logdict['epochs_before'] = get_event_counts(epochs)
    bad_epochs = find_bad_epochs(epochs)

    if len(bad_epochs) > 0:
        epochs.drop(bad_epochs)
    logdict['epochs_after'] = get_event_counts(epochs)
    logdict['num_bad_epochs'] = len(bad_epochs)

    # Step 3: mark bad ICA components (using the build-in MNE functionality for this)
    print('# Step 3: mark bad ICA components (using the build-in MNE functionality for this)')
    ica = mne.preprocessing.ICA(0.99).fit(epochs)
    ica.exclude = find_bad_components(ica, epochs)

    logdict['rejected ica components'] = len(ica.exclude)
    ica.apply(epochs)
    # Need to re-baseline data after ICA transformation
    epochs.apply_baseline(epochs.baseline)

    # Step 4: mark bad channels for each epoch and interpolate them.
    print('# Step 4: mark bad channels for each epoch and interpolate them.')
    bad_channels_per_epoch = find_bad_channels_in_epochs(epochs, eeg_ref_corr=True)
    for i, b in enumerate(bad_channels_per_epoch):
        if len(b) > 0:
            ep = epochs[i]
            ep.info['bads'] = b
            ep.interpolate_bads()
            epochs._data[i, :, :] = ep._data[0, :, :]

    # Compute evoked after cleaning, using an average EEG reference
    # Second filtering to catch any high frequency electrical noise
    epochs.filter(l_freq=None, h_freq=40, method='fir', picks=picks)  # , h_trans_bandwidth=9)
    epochs.set_eeg_reference('average')
    epochs.apply_baseline(baseline)
    report.add_epochs(epochs=epochs, title='Epochs after FASTER')
    evoked_after = epochs.average()
    report.add_evokeds(evokeds=[evoked_before, evoked_after], titles=['Evoked before FASTER', 'Evoked after FASTER'])

    ##############################################################################
    if plotting:
        evoked_after.plot()
        epochs.plot_psd(fmin=2, fmax=100)

    return epochs, evoked_before, evoked_after, logdict, report


def get_event_counts(epochs):
    """
    Counts all events in an epoch object.
    Outputs a dictionary with event counts by event labels.
    """
    ids = {event: event_id for event_id, event in epochs.event_id.items()}
    events = [ids[e[2]] for e in epochs.events]
    event_counts = dict(Counter(events))
    return event_counts


def add_to_log_csv(logdict, output_path):
    """
    Takes log dictionary and saves it to a csv.
    If csv already exists append a row.
    Participant already in dict, overwrite their row with the new one.
    """
    log_path = Path(output_path, 'preprocessing_log.csv')
    logdict['data'] = str(datetime.now()).split('.')[0]
    df = pd.DataFrame([logdict])
    if not log_path.exists():
        df.to_csv(log_path, index=False)
    else:
        df = pd.read_csv(log_path)
        for idx, df_row in df.iterrows():
            if logdict['pid'] == df_row['pid']:
                df.loc[idx] = pd.Series(logdict)
                break
        else:
            df = df.append(pd.Series(logdict), ignore_index=True)
        df.to_csv(log_path, index=False)


def find_dead_channels(epochs, plot, deviations=3):
    """
    FASTER misses some dead channels as they look like the inverse of the reference.
    This function identifies them and interpolates them.
    """
    data = epochs.get_data()
    data = np.mean(data, axis=0)
    # Power Spectral Density using Welch's method
    psds, freqs = mne.time_frequency.psd_array_welch(x=data, sfreq=200)
    # Get average power for each channel
    ch_means = np.mean(psds, axis=1)
    # These values are really tiny. Change to log scale so we can see a difference
    ch_means = np.log(ch_means)
    # Assign power to channel names
    d = {ch: z for z, ch in zip(ch_means, epochs.ch_names)}
    if plot:
        plt.bar(*zip(*d.items()))
        plt.show()
    # Get grand mean and standard deviation
    M = np.mean(ch_means)
    std = np.std(ch_means)
    # If power is more than 3 deviations away from semi-uniform distribution: mark as bad
    bads = [key for key, val in d.items() if val < (M - (deviations * std)) or val > (M + (deviations * std))]
    if 'VEOG' in bads:
        bads.remove('VEOG')
    print(bads)
    # Add all bads to info and interpolate them.
    for bad in bads:
        epochs.info['bads'].append(bad)
    epochs.interpolate_bads()
    return epochs, bads


class EEG_Participant:
    """
    Defines participant object that contains all the information needed for preprocessing an individual's data.
    This information is then used to filter the raw recording and then preprocess with FASTER.
    The epochs object can then be used for further analysis.

    @param pid: The participants string identifier and name of recording data files.
    @param ppt_num: The participant's integer identifier e.g 1.
    @param data_path: The path to directory containing participant raw data (eeg, triggers, extra triggers)
    @param ref_channel: The channel acting as the reference electrode.
    @param EOG_channel: The name of the EOG channel used for blink detection.
    @param event_ids: The basic trigger values and their labels in the raw data.
    Any triggers and associated data not listed will be dropped at the preprocess_RAW stage.
    @param montage: The montage object used to map the electrodes in 3d space.
    @param status: meta info about the participant's progress through preprocessing e.g 'raw_filtered'
    @param output_path: The location of the preprocessing database where all the outputs end up.
    """

    def __init__(self, pid: str, ppt_num: int, data_path: Union[str, PosixPath], data_format: str, ref_channel: str,
                 EOG_channel: str, event_ids: dict, montage: mne.channels.DigMontage, status: str,
                 output_path: Union[str, PosixPath]):
        """
        Initialises participant object
        """
        self.status = status
        self.pid = pid
        self.raw_fname = Path(data_path, pid).with_suffix(data_format)
        self.filename = Path(output_path, pid).with_suffix('.pickle')
        self.ppt_num = ppt_num
        self.data_path = data_path
        self.montage = montage
        self.EOG_channel = EOG_channel
        self.ref_channel = ref_channel
        self.event_ids = event_ids
        self.report = mne.Report(title=pid)
        self.picks = None
        self.events = None
        self.RAW = None
        self.epochs = None
        self.evoked_before = None
        self.evoked_after = None

    def read_RAW(self, sfreq=200, hfreq=40, lfreq=0.5, plotting=False):
        """
        Uses the ANT .cnt file reader to read raw eeg data. Needs modifying to take other data formats.
        Then filters and resamples data.

        @param sfreq: Resampling frequency
        @param hfreq: lowpass filter frequency.
        @param lfreq: highpass filter frequency.
        @param plotting: show plots or not
        """
        self.RAW, self.events, self.picks, self.report = resample_and_bandpass_raw(
            raw_fname=self.raw_fname,
            ref_channel=self.ref_channel,
            eog_channel=self.EOG_channel,
            montage=self.montage,
            plotting=plotting,
            report=self.report,
            sfreq=sfreq, hfreq=hfreq, lfreq=lfreq,
        )

    def preprocess_RAW(self, tmin, bmax, tmax, plotting=False):
        """
        Splits bandpassed and resampled data into epochs and applies the FASTER protocol to them.
        Also clears previous information from the report if we've preprocessed this participant before

        Parameters
        ----------
        tmin : float
            Starting time for epoch and baseline in seconds. Should be negative, as 0 is the event time.
        bmax : float
            The end of the baseline. Usually 0 unless there is a time period before the event you need to avoid
            including in the baseline.
        tmax : float
            The end time for the epoch.
        plotting : bool
        """
        # Todo remove everything that doesn't have the RAW tag instead
        for element in ['Time course (EEG)', 'Topographies', 'Global field power']:
            self.report.remove(title=element, tags=('evoked',), remove_all=True)
        for element in ['Info', 'ERP image (EEG)', 'Drop log', 'PSD']:
            self.report.remove(title=element, tags=('epochs',), remove_all=True)

        self.epochs, self.evoked_before, self.evoked_after, _, self.report = preprocess_with_faster(self.RAW,
                                                                                                    self.events,
                                                                                                    self.event_ids,
                                                                                                    self.picks,
                                                                                                    tmin=tmin,
                                                                                                    bmax=bmax,
                                                                                                    tmax=tmax,
                                                                                                    plotting=plotting,
                                                                                                    report=self.report)
        self.status = 'raw_filtered'

    def replace_events(self, event_file, keep_original_events=False):
        """
        Adds new annotations and events to RAW data

        Parameters
        ----------
        event_file : str
            Must be a csv with three columns: label, start, end
            Multiple labels can be assigned to the same trigger time by separating the label names with '/'.
            start is the start time of the trigger in ms.
            All values in end should be 0
        keep_original_events : bool
            If false original triggers will be replaced with the new ones.
        """
        df = pd.read_csv(event_file, names=['label', 'start', 'end'])
        my_annot = mne.Annotations(onset=df['start'], duration=df['end'], description=df['label'])
        self.RAW.set_annotations(my_annot)
        if keep_original_events:
            annot_from_events = mne.annotations_from_events(events=self.events,
                                                            event_desc={val: key for key, val in
                                                                        self.event_ids.items()},
                                                            sfreq=self.RAW.info['sfreq'],
                                                            orig_time=self.RAW.info['meas_date'])
            self.RAW.annotations.__add__(annot_from_events)
        self.events, self.event_ids = mne.events_from_annotations(self.RAW)

    def save(self, make_report=True):
        """Save Participant object as .pickle.

        Parameters
        ----------
        make_report : bool
            Choose to save separate html report file at the same time.
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)
        if make_report:
            self.save_report()

    @classmethod
    def load(cls, filename):
        """Load Participant object.

        Parameters
        ----------
        filename : str or PosixPath
        """
        with open(filename, 'rb') as f:
            # print('loading from pickle...')
            return pickle.load(f)

    def get_epochs(self, by_events=''):
        """
        mne.Epochs object getter.

        :param by_events: str
            Events to select epochs in epochs object by. Can be multiple events if they're separated by '/'.
        :return: mne.Epochs object
        """
        return self.epochs[by_events]

    def save_report(self):
        self.report.save(self.filename.with_suffix('.html'), overwrite=True, open_browser=False)


class EEG_Experiment:
    """
    Class for loading and preprocessing multiple data files at once. Stores them as EEG_Participant objects.
    """

    def __init__(self, exp_filepath, output_path, event_ids, montage):
        """
        :type exp_filepath: str or PosixPath
        :type output_path: str or PosixPath
        :type event_ids: dict
        :type montage: DigMontage
        """
        self.exp_file = pd.read_csv(exp_filepath)
        self.output_path = output_path
        self.event_ids = event_ids
        self.montage = montage
        self.participants = []

        for idx, row in self.exp_file.iterrows():
            self.participants.append(
                EEG_Participant(pid=row.pid, ppt_num=row.ppt_num, data_path=row.data_path, data_format=row.raw_format,
                                ref_channel=row.ref_channel,
                                EOG_channel=row.EOG_channel, event_ids=self.event_ids, montage=self.montage,
                                status=row.status, output_path=output_path))

    def read_RAWs(self, sfreq=200, hfreq=40, lfreq=0.5, plotting=False, skip_existing=True):
        """
        Read multiple raw eeg files

        @param sfreq: sample rate
        @param hfreq: high frequency stop
        @param lfreq: low frequency stop
        @param plotting: Run with or without pausing to plot data
        @param skip_existing: If participant already has filtered or epoched data skip
        """
        for participant in self.participants:
            if skip_existing is True:
                if participant.status == 'raw_filtered' or participant.status == 'epoched':
                    print(f'{participant.pid} raw data already filtered. Skipping')
                    continue
            participant.read_RAW(sfreq, hfreq, lfreq, plotting)
            participant.save(make_report=True)
            # Clear RAW from memory otherwise we might run out if we load a lot of participants
            del participant.RAW

    def preprocess_RAWs(self, tmin, bmax, tmax, additional_events_fname=None, plotting=False, skip_existing=True):
        for participant in self.participants:
            if participant.status == '':
                print(f'{participant.pid} raw data not filtered. Skipping')
                continue
            elif participant.status == 'epoched' and skip_existing is True:
                print(f'{participant.pid} epoch data already exists. Skipping')
                continue

            new_data_path = participant.data_path
            new_filename = Path(self.output_path, participant.pid).with_suffix('.pickle')
            participant = participant.load(new_filename)
            participant.data_path = new_data_path
            participant.filename = new_filename

            if participant.epochs is not None and skip_existing is True and participant.status == 'epoched':
                print(f'{participant.pid} epoch data already exists. Skipping')
                continue

            if 'extra_events_path' in self.exp_file.columns:
                additional_events_fname = self.exp_file[self.exp_file['ppt_num'] == int(participant.ppt_num)][
                    'extra_events_path'].values[0]
                participant.replace_events(Path(additional_events_fname))
            elif additional_events_fname is not None:
                participant.replace_events(Path(participant.data_path, additional_events_fname).with_suffix('.csv'))

            participant.preprocess_RAW(tmin, bmax, tmax, plotting)
            participant.save()
            # Clear RAW from memory otherwise we might run out if we load a lot of participants
            del participant.RAW


def run_with_UI():
    """
    Run preprocessor over experiment_participant_list csv,
    building EEG_Experiment object and outputting preprocessed eeg data.
    """
    box = dialogue_window(title='Preprocessing Setup',
                          default_plist='/Volumes/psgroups/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Radiologists/experiment_participant_list.csv',
                          default_output='/Volumes/psgroups/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/output',
                          default_trg_labels='/Volumes/psgroups/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Radiologists/experiment_trigger_labels.csv')
    box.show()
    settings = box.get_output()
    print(settings)
    assert settings['Output DB:'].exists()

    ANTwave64 = mne.channels.read_custom_montage(fname=Path('montages', 'waveguard64_rescaled_small.xyz'),
                                                 coord_frame="unknown")
    # The basic trigger values and their labels in each recording.
    trigger_df = pd.read_csv(settings['Trigger labels:'])
    event_ids = {row['label']: row['value'] for idx, row in trigger_df.iterrows() if row['active'] == 1}

    study = EEG_Experiment(exp_filepath=settings['Participant List File:'],
                           output_path=settings['Output DB:'],
                           event_ids=event_ids,
                           montage=ANTwave64)

    if settings['Filter raw']:
        study.read_RAWs(skip_existing=settings['skip existing'])

    study.preprocess_RAWs(tmin=settings['tmin'], bmax=settings['bmax'], tmax=settings['tmax'],
                          additional_events_fname=settings['additional_events_fname'],
                          plotting=settings['plotting'],
                          skip_existing=settings['skip existing'])


if '__main__' in __name__:
    run_with_UI()

    # default_plist = '/Users/llr510/PycharmProjects/EEGpykit/experiments/e1/experiment_participant_list_dots.csv'
    # default_output = '/Volumes/psgroups/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK/MNE_preprocessing_db'
    # default_trg_labels = '/Users/llr510/PycharmProjects/EEGpykit/experiments/e1/experiment_trigger_labels.csv'

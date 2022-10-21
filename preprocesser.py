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
from pathlib import Path
import shutil
import sys
import mne
from mne import io
import io as io2
# pip install https://github.com/wmvanvliet/mne-faster/archive/refs/heads/main.zip
from mne_faster import (find_bad_channels, find_bad_epochs,
                        find_bad_components, find_bad_channels_in_epochs)
from read_antcnt import read_raw_antcnt, read_events_trg
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import Counter
from datetime import datetime


def resample_and_bandpass_raw(raw_fname, ref_channel, eog_channel, montage,
                              sfreq=200, hfreq=40, lfreq=0.5, plotting=False, report=None):
    event_fname = Path(raw_fname).with_suffix('.trg')
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

    # if labelled_events_fname is not None:
    #     df = pd.read_csv(labelled_events_fname, names=['label', 'start', 'end'])
    #     my_annot = mne.Annotations(onset=df['start'], duration=df['end'], description=df['label'])
    #     raw.set_annotations(my_annot)

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

    # annot_from_events = mne.annotations_from_events(events=events,
    #                                                 event_desc={val: key for key, val in event_ids.items()},
    #                                                 sfreq=raw.info['sfreq'], orig_time=raw.info['meas_date'])
    # raw.annotations.__add__(annot_from_events)
    # events, event_ids = mne.events_from_annotations(raw)

    report.add_raw(raw=raw, title='Raw after filtering')
    return raw, events, picks, report


def preprocess_with_faster(raw, events, event_ids, picks, pid, tmin=-0.5, bmax=0, tmax=1, plotting=False,
                           report=None):
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
    """Counts all events in an epoch object.
    Outputs a dictionary with event counts by event labels."""
    ids = {event: event_id for event_id, event in epochs.event_id.items()}
    events = [ids[e[2]] for e in epochs.events]
    event_counts = dict(Counter(events))
    return event_counts


def get_img_from_fig(fig, dpi=180):
    """converts numpy figure to image file and then reads that in as a numpy array"""
    with io2.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))

    return im


def combine_plots(figs=[], fig_path='', pid=str, epoch=str, channel=str):
    """Creates the combined before and after preprocessing plots"""
    dats = []
    for fig in figs:
        data = get_img_from_fig(fig, dpi=300)
        dats.append(data)
    dats = np.concatenate(dats)
    plt.imshow(dats)
    plt.axis('off')
    plt.title(f'Participant: {pid}, Epochs: {epoch}, Channel(s): {channel}')
    plt.tight_layout()
    plt.savefig(Path(fig_path, f'example-fig_{pid}_{epoch}_{channel}.png'), dpi=150, bbox_inches='tight')


def example_erp_plotting(evoked, pid, channels, fig_path):
    if channels == 'all':
        picks = None
    else:
        picks = channels

    before = evoked['before'].plot(picks=picks, gfp=True, titles='Before FASTER', show=True, selectable=False)
    after = evoked['after'].plot(picks=picks, gfp=True, titles='After FASTER', show=True, selectable=False)
    combine_plots(figs=[before, after], fig_path=fig_path, pid=pid, epoch='all', channel=channels)


def add_to_log_csv(logdict, output_path):
    """Takes log dictionary and saves it to a csv.
    If csv already exists append a row.
    Participant already in dict, overwrite their row with the new one."""

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
    """FASTER misses some dead channels as they look like the inverse of the reference.
    This function identifies them and interpolates them."""
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
    """Defines participant object that contains all the information needed for preprocessing an individual's data."""
    def __init__(self, pid='', ppt_num=None, data_path='None', ref_channel=None, EOG_channel=None, event_ids=None,
                 montage=None, status=None):
    # def __init__(self, pid, ppt_num, data_path, ref_channel, EOG_channel, event_ids,
    #              montage, status):
        output_path = Path(
            '/Volumes/psgroups/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK/MNE_preprocessing_db')
        self.status = status
        self.pid = pid
        self.raw_fname = Path(data_path, pid).with_suffix('.cnt')
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

    def read_RAW(self, sfreq=200, hfreq=40, lfreq=0.5, plotting=True):
        self.RAW, self.events, self.picks, self.report = resample_and_bandpass_raw(
            raw_fname=self.raw_fname,
            ref_channel=self.ref_channel,
            eog_channel=self.EOG_channel,
            montage=self.montage,
            plotting=plotting,
            report=self.report,
            sfreq=sfreq, hfreq=hfreq, lfreq=lfreq,
        )

    def preprocess_RAW(self, tmin, bmax, tmax, plotting=True):
        # Todo remove everything that doesn't have the RAW tag instead
        for element in ['Time course (EEG)', 'Topographies', 'Global field power']:
            self.report.remove(title=element, tags=('evoked',), remove_all=True)
        for element in ['Info', 'ERP image (EEG)', 'Drop log', 'PSD']:
            self.report.remove(title=element, tags=('epochs',), remove_all=True)

        self.epochs, self.evoked_before, self.evoked_after, _, self.report = preprocess_with_faster(self.RAW,
                                                                                                    self.events,
                                                                                                    self.event_ids,
                                                                                                    self.picks,
                                                                                                    self.pid,
                                                                                                    tmin=tmin,
                                                                                                    bmax=bmax,
                                                                                                    tmax=tmax,
                                                                                                    plotting=plotting,
                                                                                                    report=self.report)

    def replace_events(self, event_file, keep_original_events=False):
        """Adds new annotations and events to RAW data"""
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
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)
        if make_report:
            self.save_report()

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            print('loading from pickle')
            return pickle.load(f)

    def get_epochs(self, by_events=''):
        return self.epochs[by_events]

    def save_report(self):
        self.report.save(self.filename.with_suffix('.html'), overwrite=True, open_browser=False)


class EEG_Experiment:
    """Class for loading and preprocessing multiple data files at once. Stores them as EEG_Participant objects."""

    def __init__(self, exp_filepath, output_path, event_ids, montage):
        self.exp_file = pd.read_csv(exp_filepath)
        self.output_path = output_path
        self.event_ids = event_ids
        self.montage = montage
        self.participants = []

        for idx, row in self.exp_file.iterrows():
            self.participants.append(EEG_Participant(pid=row.pid,
                                                     ppt_num=row.ppt_num,
                                                     data_path=row.data_path,
                                                     ref_channel=row.ref_channel,
                                                     EOG_channel=row.EOG_channel,
                                                     event_ids=self.event_ids,
                                                     montage=self.montage,
                                                     status=row.status))

    def read_RAWs(self, sfreq=200, hfreq=40, lfreq=0.5, plotting=False):
        for participant in self.participants:
            if participant.status == 'raw_filtered':
                continue
            participant.read_RAW(sfreq, hfreq, lfreq, plotting)
            participant.save(make_report=True)
            # Clear RAW from memory otherwise we might run out if we load a lot of participants
            del participant.RAW

    def preprocess_RAWs(self, tmin, bmax, tmax, additional_events_fname=None, plotting=False):
        for participant in self.participants:
            if participant.status != 'raw_filtered':
                continue
            participant = participant.load(participant.filename)

            if additional_events_fname is not None:
                participant.replace_events(Path(participant.data_path, additional_events_fname).with_suffix('.csv'))
            participant.preprocess_RAW(tmin, bmax, tmax, plotting)
            participant.save()
            # Clear RAW from memory otherwise we might run out if we load a lot of participants
            del participant.RAW


if '__main__' in __name__:
    output_path = Path(
        '/Volumes/psgroups/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK/MNE_preprocessing_db')
    pid = '20220318_1418PPT1NEW'
    ANTwave64 = mne.channels.read_custom_montage(fname=Path('montages', 'waveguard64_rescaled_small.xyz'),
                                                 coord_frame="unknown")
    event_ids = {'T1/S': 11,
                 'T1/NS': 12,
                 'T2/S': 21,
                 'T2/NS': 22, }

    study = EEG_Experiment(exp_filepath='experiment_participant_list.csv',
                           output_path=output_path,
                           event_ids=event_ids,
                           montage=ANTwave64)
    # study.read_RAWs()
    study.preprocess_RAWs(tmin=-0.2, bmax=0, tmax=0.8, additional_events_fname='new_markers', plotting=False)
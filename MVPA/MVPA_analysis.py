import argparse
import copy
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import mne
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib import rcParams
from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.stats import ttest_ind_no_p, spatio_temporal_cluster_test
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

rcParams['axes.unicode_minus'] = False


# from MVPA_utils import *


def get_filepaths_from_file(analysis_file):
    """
    Reads a path list file with the path of each input epoched data file.
    Gets participant number and session and returns that information so it can be added to the trigger labels for
    group analyses.

    @param analysis_file: csv file with mne epo.fif files listed
    @return: list of filepath, list of lists of extra event label metadata
    """
    df = pd.read_csv(analysis_file)

    files = []
    extra = []
    for idx, row in df.iterrows():
        files.append(row['epo_path'])
        # get ppt num and session number
        extra.append([f"ppt_{row['ppt_num']}", f"sesh_{row['sesh_num']}"])

    return files, extra


def recode_label(event, extra_labels=None, sep='/'):
    """
    Used to adjust event names so that they have the desired format

    @param extra_labels: list of extra labels to join with
    @param event: epochs.event_id.items
    @param sep: character that seperates labels
    @return: tags
    """
    tags = sep.join([event] + extra_labels)

    return tags


def events_in_epochs(event_labels, epochs):
    """
    Check if a list of event labels can be found in the event keys of an epochs object.
    Helps avoid keys errors when indexing data from participants that didn't do ever condition

    @param event_labels: list or nested list of event labels
    @param epochs: an mne.epochs object
    @return: boolean
    """
    try:
        epochs[event_labels]
    except KeyError:
        return False
    return True


def plot_svm_scores(times, scores, scoring="roc_auc", title='', filename='scores.png'):
    """
    Make a basic line plot of SVM accuracy

    @param times: array of samples on y axis
    @param scores: array of accuracy scores on x axis
    @param scoring: scoring method (sklearn.metrics.get_scorer_names())
    @param title: title of plot
    """
    fig, ax = plt.subplots()
    ax.plot(times, scores, label="score")
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(scoring)
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title(title)
    plt.savefig(filename, dpi=150)
    plt.close()


def average_epochs(X):
    """
    Get the mean of an epochs array across the epochs dimensions.
    Used to created evoked objects that can used for 2d+ plotting

    @param X: epochs x channels x samples
    @return: channels x samples
    """
    return X.mean(axis=0)


def sort_channels(epochs):
    """
    Sort channels in row-major snakelike ordering according to the layout of a 64 channel ANTneuro Waveguard cap.
    """
    sorted_ch_names = ['Fp1', 'Fpz', 'Fp2',
                       'AF8', 'AF4', 'AF3', 'AF7',
                       'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                       'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'FC1', 'FC3', 'FC5', 'FT7',
                       'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M2',
                       'TP8', 'CP6', 'CP4', 'CP2', 'CPz', 'CP1', 'CP3', 'CP5', 'TP7', 'M1',
                       'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                       'PO8', 'PO6', 'PO4', 'POz', 'PO3', 'PO5', 'PO7',
                       'O1', 'Oz', 'O2']

    new_channel_order = [x for x in sorted_ch_names if x in epochs.ch_names]
    if len(new_channel_order) == len(epochs.ch_names):
        epochs.reorder_channels(new_channel_order)
    else:
        print('Mismatch between sorted channel names and epochs channels names. Sorting not applied.')
    return epochs


def movingaverage(y, window_length):
    """
    Used for smoothed the scores using a 3 point moving average to smooth out spurious fluctuations.
    From: https://github.com/SridharJagannathan/decAlertnessDecisionmaking_JNeuroscience2021/blob/main/Scripts/notebooks/Figure6_temporaldecodingresponses.ipynb

    @param y: np array of scores
    @param window_length:
    @return: array of smoothed scores
    """
    y_smooth = np.convolve(y, np.ones(window_length, dtype='float'), 'same') / \
               np.convolve(np.ones(len(y)), np.ones(window_length), 'same')
    return y_smooth


def _stat_fun_2samp(a, b, sigma=0):
    """
    This secondary function reduces the time of computation of p-values and adjusts for small-variance values
    """
    t_values = ttest_ind_no_p(a, b, sigma=sigma, equal_var=True)
    t_values[np.isnan(t_values)] = 0
    return t_values


def cluster_stats_2samp(X_list, n_jobs=-1):
    """
    Cluster statistics to control for multiple comparisons. Repeated measures 2 sample t test

    adapted from https://github.com/kingjr/decod_unseen_maintenance/blob/master/scripts/base.py
    performs stats of the group level.
    X is usually nsubj x ntpts -> composed of mean roc scores per subj per timepoint.
    performs cluster stats on X to identify regions of tpts that have roc significantly differ from chance.


    Parameters
    ----------
    X : list of array, shape (n_samples, n_space, n_times)
        The data, chance is assumed to be 0.
    n_jobs : int
        The number of parallel processors.
    """

    # assert len(X_list[0]) == len(X_list[1])
    n_subjects = len(X_list[0])
    X_list = [np.array(X) for X in X_list]
    X_list = [X[:, :, None] if X.ndim == 2 else X for X in X_list]
    # this functions gets the t-values and performs a cluster permutation test on them to determine p-values.
    p_threshold = 0.05
    # Repeated measures t-test so degrees of freedom is n-1
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
    print('number of subjects/trials:', n_subjects)
    print('t-threshold is:', t_threshold)
    print('p-threshold is:', p_threshold)
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_test(
        X_list, out_type='mask', stat_fun=_stat_fun_2samp, n_permutations=2 ** 12, seed=1234,
        n_jobs=n_jobs, threshold=t_threshold)
    p_values_ = np.ones_like(X_list[0][0]).T
    # rearrange the p-value per cluster.
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    return np.squeeze(p_values_).T


def activity_map_plots(epochs, group1, group2, plot_significance=True, alpha=0.05, abs_difference = True):
    """
    Plot figures for individual participant sensor activity comparing two different conditions together.
    Significant differences are calculating using a spatio-temporal cluster t-test.
    Plots are of weighted  activity deltas.

    @param abs_difference: whether to plot topomaps with absolute difference values or negative and positive difference values
    @param epochs: mne.Epochs object
    @param group1: list of event labels
    @param group2: list of event labels
    @param plot_significance: filters and only plots significant data if True
    @param alpha: p level for filtering and mask
    @return: heatmap plot, topomap plot, animated topomap plot
    """
    mask_params = dict(markersize=10, markerfacecolor="y")

    if type(epochs) is dict:
        # If input is a 2 key dictionary of evoked objects, get a group plot instead
        evoked_group1 = mne.grand_average(epochs['group1'])
        evoked_group2 = mne.grand_average(epochs['group2'])
        times = evoked_group1.times
        ch_names = evoked_group1.ch_names
        # combine data along first axis: participant x channels x times
        # dstack combines 2d data along a 3rd dimension, so that axis has to be moved to the front
        data_group1 = np.moveaxis(np.dstack([i.get_data() for i in epochs['group1']]), -1, 0)
        data_group2 = np.moveaxis(np.dstack([i.get_data() for i in epochs['group2']]), -1, 0)
    else:
        times = epochs.times
        ch_names = epochs.ch_names
        evoked_group1 = epochs[group1].average(method=average_epochs)
        evoked_group2 = epochs[group2].average(method=average_epochs)
        data_group1 = epochs[group1].get_data()
        data_group2 = epochs[group2].get_data()

    # subtract evoked conditions
    evoked_diff = mne.combine_evoked([evoked_group1, evoked_group2], weights=[1, -1])
    X_diff = evoked_diff.get_data()

    if plot_significance:
        # Filter data by significance
        indiv_pvalues = cluster_stats_2samp([data_group1, data_group2], n_jobs=-1)
        # evoked_sig = evoked_diff.copy()
        # evoked_sig.data = X_diff
        if indiv_pvalues.all() > alpha:
            print('no significant clusters for individual plots')
            mask = None
            sig = f' p not < {alpha}'
        else:
            # set non-significant values to 0
            X_diff[np.where(indiv_pvalues > alpha)] = 0
            # Define a threshold and create the mask
            mask = indiv_pvalues < alpha
            sig = f' p<{alpha}'
    else:
        mask = None
        sig = ''

    # Make heatmap plot
    heat, ax = plt.subplots(figsize=(12, 10))
    plot = ax.pcolormesh(times, ch_names, X_diff, cmap='twilight', norm=TwoSlopeNorm(0))

    if X_diff.min() == X_diff.max():
        heat.colorbar(plot, label='ΔµV')
    else:
        ticks = ticker.MultipleLocator(1e-6 / 2)
        # tick_list = ticks.tick_values(np.min(X_diff), np.max(X_diff))
        heat.colorbar(plot, label='ΔµV', ticks=ticks)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_ylabel('Channels')
    ax.set_xlabel('Times (ms)')
    ax.set_title(f"{'-'.join(group1)} vs {'-'.join(group2)} - Heatmap{sig}")

    # Make topomap plot
    if abs_difference:
        evoked_diff_abs = evoked_diff.copy()
        evoked_diff_abs.data = abs(evoked_diff_abs.get_data())
        topo = evoked_diff_abs.plot_topomap(times="peaks", ch_type="eeg", mask=mask, mask_params=mask_params, show=True,
                                            cmap='Reds', vlim=(0, None))
    else:
        topo = evoked_diff.plot_topomap(times="peaks", ch_type="eeg", mask=mask, mask_params=mask_params, show=True)
    topo.suptitle(f"{'-'.join(group1)} vs {'-'.join(group2)} - Topomap{sig}")

    # Make animated topoplot
    # fig, anim = evoked_sig.animate_topomap(times=epochs.times, ch_type="eeg", frame_rate=12, show=False,
    # blit=False, time_unit='s')  # , mask=mask, mask_params=mask_params)
    return {'heatmap.png': heat, 'topomap.png': topo}  # , 'animated_topomap.mp4': anim}


def decodingplot(scores_cond, p_values_cond, times, alpha=0.05, color='r', tmin=-0.8, tmax=0.3):
    """
    Plot temporal decoding MVPA results on a nicer graph. Can also display time points of statistical significance

    @param scores_cond: if an array get the mean of scores, if just a list plot as is
    @param p_values_cond: array of p values of length of scores_cond
    @param times: array of time points in data
    @param alpha: significance threshold
    @param color: error bar colour
    @param tmin: epoch start time
    @param tmax: event start time
    @return: Object with figure axis, times, scores. Also returns another empty object.
    """
    fig, ax1 = plt.subplots(nrows=1, figsize=[20, 4])

    if type(scores_cond) is not np.ndarray:
        scores = np.array(scores_cond)
        scores_m = np.nanmean(scores, axis=0)
        n = len(scores)
        n -= sum(np.isnan(np.mean(scores, axis=1)))  # identify the nan subjs and remove them.
        sem = np.nanstd(scores, axis=0) / np.sqrt(n)
        ax1.fill_between(times, scores_m - sem, scores_m + sem, color=color, alpha=0.3)
    else:
        scores_m = scores_cond

    sig = p_values_cond < alpha

    ax1.plot(times, scores_m, 'k', linewidth=1)

    split_ydata = scores_m
    split_ydata[~sig] = np.nan
    # shade the significant regions.
    ax1.plot(times, split_ydata, color='k', linewidth=3)
    ax1.fill_between(times, y1=split_ydata, y2=0.5, alpha=0.7, facecolor=color)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax1.grid(True)

    ax1.axhline(y=0.5, linewidth=0.75, color='k', linestyle='--')
    ax1.axvline(x=0, linewidth=0.75, color='k', linestyle='--')

    timeintervals = np.arange(tmin, tmax, 0.1)
    timeintervals = timeintervals.round(decimals=2)

    ax1.set_xlim([tmin, tmax])
    ax1.set_xticks(timeintervals)

    for patch in ax1.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))

    ax1.patch.set_edgecolor('black')

    ax1.patch.set_linewidth(0)

    for a in fig.axes:
        a.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,
            top=False,
            labelbottom=True)  # labels along the bottom edge are on

    class Scratch(object):
        pass

    returnval1 = Scratch()
    returnval2 = Scratch()

    returnval1.axes = ax1
    returnval1.times = times[sig]
    returnval1.scores = scores_m[sig]

    return returnval1, returnval2


def temporal_decoding(x_data, y, scoring="roc_auc", groups=[], jobs=-1):
    """
    https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html

    This strategy consists in fitting a multivariate predictive model on each time instant and evaluating its
    performance at the same instant on new epochs. The mne.decoding.SlidingEstimator will take as input a pair of
    features X and targets y, where X has more than 2 dimensions. For decoding over time the data X is the epochs data of
    shape n_epochs × n_channels × n_times. As the last dimension X of is the time, an estimator will be fit on every
    time instant.

    This approach is analogous to SlidingEstimator-based approaches in fMRI, where here we are interested in when one
    can discriminate experimental conditions and therefore figure out when the effect of interest happens.

    When working with linear models as estimators, this approach boils down to estimating a discriminative spatial
    filter for each time instant.

    @param scoring: what sklearn scoring measure to use. See sklearn.metrics.get_scorer_names() for options
    @param x_data: 3d array of n_epochs, n_meg_channels, n_times
    @param y: array of epoch events
    @param groups: integer array of unique data groups if using LOGO cross validation instead of k-fold.
    @param jobs: number of logical processor cores to use (-1 uses all). Smaller number uses less RAM but takes longer.
    @return score: array of SVM accuracy scores (TODO what shape??)

    Original: https://github.com/SridharJagannathan/decAlertnessDecisionmaking_JNeuroscience2021/blob/main/Scripts/notebooks/Figure6_temporaldecodingresponses.ipynb
    """
    # make an estimator with scaling each channel by across its time pts and epochs.
    # to deal with class imbalance weights classes with formula: n_samples / (n_classes * np.bincount(y))
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced'))
    # Sliding estimator with classification made across each time pt by training with the same time pt.
    time_decod = SlidingEstimator(clf, n_jobs=jobs, scoring=scoring, verbose=False)
    # Compute the cross validation score. If given groups use LOGO otherwise just use 5-fold validation
    if any(groups):
        logo = LeaveOneGroupOut()
        scores = cross_val_multiscore(time_decod, x_data, y, cv=logo, groups=groups, n_jobs=jobs)
    else:
        scores = cross_val_multiscore(time_decod, x_data, y, cv=5, n_jobs=jobs)
    # Mean scores across cross-validation splits.
    score = np.mean(scores, axis=0)
    # Smooth datapoints to ensure no discontinuities.
    score = movingaverage(score, 10)
    return score


def MVPA_analysis(files, var1_events, var2_events, excluded_events=[], scoring="roc_auc", output_dir='',
                  indiv_plot=False, epochs_list=[], extra_event_labels=[], overwrite_output=False,
                  jobs=-1):
    """
    Performs MVPA analysis over multiple participants.
    concatenate all epochs and run MVPA on that instead of individuals.


    @param output_dir: output directory for figures
    @param files: iterable or list of .epo.fif filepaths
    @param var1_events: list of event conditions for MVPA comparison
    @param var2_events: list of event conditions for MVPA comparison
    @param excluded_events: list of events to exclude from analysis
    @param scoring: scoring method for estimator. e.g: 'accuracy', 'roc_auc
    @param indiv_plot: whether to plot individual data
    @param epochs_list: If epochs are already loaded, use this instead of files
    @param extra_event_labels: list of lists
    @param overwrite_output: If true rerun MVPA analysis even if saved output dat file exists
    @param jobs: number of processor cores to use (-1 uses maximum). Smaller number uses less RAM but takes longer.
    """
    # Set seed so MVPA is reproducible
    np.random.seed(1025)
    evoked_list = {'group1': [], 'group2': []}
    X_list = []
    y_list = []
    groups = []
    fname_string = f"{'-'.join(var1_events)}_vs_{'-'.join(var2_events)}".replace('/', '+')
    saved_classifier = Path(output_dir, fname_string).with_suffix('.dat')
    print(var1_events, var2_events)
    print(saved_classifier, saved_classifier.exists())
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not overwrite_output and saved_classifier.exists():
        print('Loading saved classifier scores:', saved_classifier)
        with open(saved_classifier, 'rb') as f:
            X_score, y, times, evoked_list = pickle.load(f)
            # return evoked_list, X_score, y, times
    else:
        if epochs_list:
            epochs_data = copy.deepcopy(epochs_list)

        for n, file in enumerate(files):
            # either load in epochs data here or receive it already loaded from the epochs_data argument
            if epochs_list:
                epochs = epochs_data[n]
            else:
                epochs = mne.read_epochs(file, verbose=False)
            # drop any channels that aren't EEG (usually EOG data)
            epochs.pick_types(eeg=True, exclude="bads")
            # Sort channels in row-major snakelike ordering
            epochs = sort_channels(epochs)
            # Add metadata like participant number and session number to all the event labels for this epochs object
            extra_labels = extra_event_labels[n]
            epochs.event_id = {recode_label(k, extra_labels): v for k, v in epochs.event_id.items()}
            ppt_id = extra_labels[:1]
            print(ppt_id)
            # Get data from epochs object and check its dimensions before event filtering
            X_before = epochs.get_data()  # EEG signals: n_epochs, n_eeg_channels, n_times
            print('X before: ', X_before.shape)

            # Check data for the requested events for the comparison.
            if excluded_events:
                try:
                    to_drop = list(epochs[excluded_events].event_id.values())
                    epochs.drop([True if x in to_drop else False for x in list(epochs.events[:, 2])])
                except KeyError:
                    pass

            if events_in_epochs(var1_events, epochs):
                var1_values = list(epochs[var1_events].event_id.values())
                print(var1_values)
            else:
                var1_values = []
                print(f'{var1_events} not found in {ppt_id}')

            if events_in_epochs(var1_events, epochs):
                var2_values = list(epochs[var2_events].event_id.values())
                print(var2_values)
            else:
                var2_values = []
                print(f'{var2_events} not found in {ppt_id}')
            # Added list of event ids together and make sure some are present in the data
            all_event_values = var1_values + var2_values
            if all_event_values:
                # drop epochs from the data that we're not interested in right now
                epochs = epochs[var1_events + var2_events]
            else:
                print(f'No specified events found in {ppt_id}')
                continue
            # Create evoked objects for each participant while preserving channel dimension
            if events_in_epochs(var1_events, epochs):
                evoked_1 = epochs[var1_events].average(method=average_epochs)
                evoked_list['group1'].append(evoked_1)

            if events_in_epochs(var1_events, epochs):
                evoked_2 = epochs[var2_events].average(method=average_epochs)
                evoked_list['group2'].append(evoked_2)

            # Get data from epochs object as numpy array with following dimensions
            X = epochs.get_data()  # EEG signals: n_epochs, n_eeg_channels, n_times
            print('X after: ', X.shape)

            # Get the participant id and add it to group list for multi-session leave one group out temporal decoding
            group_id = ppt_id * X.shape[0]
            groups.extend(group_id)
            # Get an array of event_ids in epochs
            y = epochs.events[:, 2]
            # Convert event_ids into truth values depending on which group they belong to
            y[np.argwhere(np.isin(y, var1_values)).ravel()] = 0
            y[np.argwhere(np.isin(y, var2_values)).ravel()] = 1
            # Get array of times for plotting purposes
            times = epochs.times
            # Add data to list for concatenation later
            X_list.append(X)
            y_list.append(y)
            # Make and save individual plots
            if indiv_plot:
                plots = activity_map_plots(epochs, group1=var1_events, group2=var2_events, plot_significance=True)
                for key, plot in plots.items():
                    if 'anim' in key:
                        plot.save(Path(output_dir, f'{Path(file).with_suffix("").stem}_{key}'))
                    else:
                        plot.savefig(Path(output_dir, f'{Path(file).with_suffix("").stem}_{key}'), dpi=240)

                    plot.clf()
                    plt.close()

                filename = Path(output_dir, f'temp_decod_{Path(file).with_suffix("").stem}.png')
                scores = temporal_decoding(X, y, scoring=scoring, jobs=jobs)
                plot_svm_scores(times, scores, scoring, title=filename.stem, filename=filename)

            # clean up epochs object to save on memory
            del epochs
        if epochs_list:
            del epochs_data

        # Combine all data along epochs axis of arrays
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        # Get a group value for each datapoint for use with leave one group out cross-validation
        # One fold per participant can take a long time so use k fold validation if you value speed.
        groups = pd.factorize(groups)[0]
        print(groups)
        print(f'''X:{X.shape}\nY:{y.shape}\nGroups:{groups.shape}''')
        print('Conditions:', np.array(np.unique(y, return_counts=True)))
        X_score = temporal_decoding(X, y, scoring=scoring, groups=groups, jobs=jobs)

        # Save data output as a pickle so plots can be changed later without loading all the data or rerunning the MVPA
        try:
            with open(Path(output_dir, fname_string).with_suffix('.dat'), 'wb') as file:
                pickle.dump((X_score, y, times, evoked_list), file)
        except Exception as e:
            print("Failed to save.")
            print(e)

    # Better plot with significance
    # Just one output so no significance testing is done. Just set p to 1.0 for every time point.
    pvalues = np.ones(len(times))
    funcreturn, _ = decodingplot(scores_cond=X_score.copy(), p_values_cond=pvalues, times=times,
                                 alpha=0.05, color='r', tmin=0, tmax=times[-1])
    funcreturn.axes.set_title(f"{'-'.join(var1_events)}_vs_{'-'.join(var2_events)} - Sensor space decoding")

    # pd.DataFrame.from_dict({'X_scores': X_score, 'times': times})

    # Limit y axis to 0.75 if performance isn't higher
    if X_score.max() > 0.75:
        ymax = 1.0
    else:
        ymax = 0.75
    if X_score.min() < 0.45:
        ymin = 0
    else:
        ymin = 0.45
    funcreturn.axes.set_ylim(ymin, ymax)

    funcreturn.axes.set_xlabel('Time (sec)')
    funcreturn.axes.set_ylabel(scoring)
    plt.savefig(Path(output_dir, 'group' + '_' + fname_string).with_suffix('.png'), dpi=240)
    plt.close()

    # Make and save group plots
    plots = activity_map_plots(evoked_list, group1=var1_events, group2=var2_events, plot_significance=True)
    for key, plot in plots.items():
        if 'anim' in key:
            plot.save(Path(output_dir, f'Group_{key}'))
        else:
            plot.savefig(Path(output_dir, f'Group_{key}'), dpi=240)
        plot.clf()
        plt.close()

    return evoked_list, X_score, y, times


def delta_evoked_MVPA(evoked_dict, condition_vars, title):
    """
    Performs an MVPA analysis on the difference between conditions

    e.g: evoked_dict['condition']['group'] = list(individual evoked objects)
    Get the difference between 'group' for each individual and then feed that into the MVPA analysis

    @param evoked_dict: dictionary of evoked objects with condition keys
    @param condition_vars: list of two conditions to index the evoked_dict by
    @param title: the title of the analysis
    """
    X_diff = []
    y = []
    groups = []
    scoring = "roc_auc"
    for n_cond, cond in enumerate(condition_vars):
        evoked_list_1 = evoked_dict[cond]['group1']
        evoked_list_2 = evoked_dict[cond]['group2']
        assert len(evoked_list_1) == len(evoked_list_2)
        for n_participant, evokeds in enumerate(zip(evoked_list_1, evoked_list_2)):
            evoked_diff = mne.combine_evoked(evokeds, weights=[1, -1])
            X_diff.append(evoked_diff.get_data())
            y.append(n_cond)
            groups.append(n_participant)

    X = np.array(X_diff)
    y = np.array(y)
    X_score = temporal_decoding(X, y, scoring, groups)
    times = evoked_list_1[0].times

    # Just one output so no significance testing is done. Just set p to 1.0 for every time point.
    pvalues = np.ones(len(times))
    funcreturn, _ = decodingplot(scores_cond=X_score.copy(), p_values_cond=pvalues, times=times,
                                 alpha=0.05, color='r', tmin=0, tmax=times[-1])
    funcreturn.axes.set_title(f"{title} - Sensor space decoding")
    # Limit y axis to 0.75 if performance isn't higher
    if X_score.max() > 0.75:
        ymax = 1.0
    else:
        ymax = 0.75
    if X_score.min() < 0.45:
        ymin = 0
    else:
        ymin = 0.45
    funcreturn.axes.set_ylim(ymin, ymax)

    funcreturn.axes.set_xlabel('Time (sec)')
    funcreturn.axes.set_ylabel(scoring)
    plt.savefig(Path(title).with_suffix('.png'), dpi=240)
    plt.close()


def run_with_cli():
    """
    Run the MVPA with a command line interface
    """
    print("################# STARTING #################")
    parser = argparse.ArgumentParser(description='Analyse EEG data with MVPA')
    parser.add_argument('--file_list', type=str, required=True, help="The epoched participant list file")
    parser.add_argument('--output', type=str, required=True, help="The output directory for the analysis")
    parser.add_argument('--var1', type=str, required=True, help="Events for condition 1")
    parser.add_argument('--var2', type=str, required=True, help="Events for condition 2")
    parser.add_argument('--excluded_events', type=str, required=False, default='', help="Events for condition 2")
    parser.add_argument('--jobs', type=int, required=False, default=-1, help="how many processes to spawn. By default "
                                                                             "uses all available processes.")
    parser.add_argument('--scoring', type=str, required=False, default='roc_auc')
    parser.add_argument('--indiv_plot', action='store_true', required=False, help="")
    parser.add_argument('--save_output', action='store_true', required=False, help="")

    args = parser.parse_args()

    files, extra = get_filepaths_from_file(args.file_list)
    print('Found epoched data files:', files)
    MVPA_analysis(files=files,
                  var1_events=args.var1.split(','),
                  var2_events=args.var2.split(','),
                  excluded_events=args.excluded_events.split(','),
                  scoring=args.scoring,
                  output_dir=args.output,
                  indiv_plot=args.indiv_plot,
                  save_output=args.save_output,
                  epochs_list=[], extra_event_labels=extra, jobs=args.jobs)


if '__main__' in __name__:
    run_with_cli()
    quit()

    files, extra = get_filepaths_from_file('../analyses/MVPA/MVPA_analysis_list_rads.csv')
    extra = extra[:3]
    files = files[:3]
    print(files)
    MVPA_analysis(files=files,
                  var1_events=['Normal/Correct'],
                  var2_events=['Obvious/Correct', 'Subtle/Correct'],
                  excluded_events=[], scoring="roc_auc",
                  output_dir='../analyses/MVPA/rads',
                  indiv_plot=False,
                  epochs_list=[], extra_event_labels=extra, jobs=-1)

    # files, extra = get_filepaths_from_file('../analyses/MVPA/MVPA_analysis_list.csv')
    # epochs_list = []
    # files = files[:6]
    # extra = extra[:6]
    # for file in files:
    #     epochs = mne.read_epochs(file, verbose=False)
    #     epochs_list.append(epochs)
    #
    # MVPA_analysis(files=files,
    #               var1_events=['sesh_1/Obvious'],
    #               var2_events=['sesh_2/Obvious'],
    #               excluded_events=[], scoring="roc_auc",
    #               output_dir='../analyses/MVPA/naives',
    #               indiv_plot=False,
    #               concat_participants=True,
    #               epochs_list=epochs_list, extra_event_labels=extra, jobs=2)

    # MVPA_analysis(files=files,
    #               var1_events=['sesh_1/Subtle'],
    #               var2_events=['sesh_2/Subtle'],
    #               excluded_events=[], scoring="roc_auc",
    #               output_dir='../analyses/MVPA/naives',
    #               indiv_plot=False,
    #               concat_participants=True,
    #               epochs_list=epochs_list, extra_event_labels=extra, jobs=2)

    # files1, _ = get_filepaths_from_file('../analyses/MVPA/MVPA_analysis_list_sesh1.csv')
    # files2, _ = get_filepaths_from_file('../analyses/MVPA/MVPA_analysis_list_sesh2.csv')
    # MVPA_group_analysis(groups={'session1': files1, 'session2': files2},
    #                     var1_events=['Obvious', 'Subtle'],
    #                     var2_events=['Normal', 'Global'],
    #                     excluded_events=['Missed', 'Rate'],
    #                     scoring="roc_auc",
    #                     output_dir='../analyses/MVPA_group',
    #                     indiv_plot=False, jobs=-1)
    # test_data_mvpa()

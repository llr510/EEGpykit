import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.stats import ttest_1samp_no_p, spatio_temporal_cluster_1samp_test
from scipy import stats
from MVPA.MVPAnalysis import MVPAnalysis, cluster_stats_2samp, activity_map_plots, temporal_decoding, \
    plot_svm_scores


def len_match_arrays(X, y, sanity_check=False):
    """
    A glm or svm can potentially learn to distinguish groups by a case imbalance.
    This function randomly drops epochs from the longer of two cases so that they are the same length.

    @param sanity_check: add a large number to the values of one condition
    @param X: 3d epochs data array.
    @param y: 1d epochs binary event type array.
    @return: len matched X,y.
    """
    x1 = X[np.where(y == 0)[0], :, :]
    x2 = X[np.where(y == 1)[0], :, :]
    print('before: ', x1.shape, x2.shape)
    if sanity_check:
        x1[:, :, 200:] = 1000000.0
    else:
        if x1.shape[0] > x2.shape[0]:
            idxs = np.random.choice(x1.shape[0], size=x2.shape[0], replace=False)
            x1 = x1[idxs, :, :]
        elif x2.shape[0] > x1.shape[0]:
            idxs = np.random.choice(x2.shape[0], size=x1.shape[0], replace=False)
            x2 = x2[idxs, :, :]
    print('after: ', x1.shape, x2.shape)
    assert x1.shape[0] == x2.shape[0]

    X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.zeros(x1.shape[0]), np.ones(x2.shape[0])])

    return X, y


def MVPA_group_analysis(groups, var1_events, var2_events, excluded_events=[], scoring="roc_auc", output_dir='',
                        jobs=-1):
    """
    For doing individual MVPA analyses and comparing them with a group permutation cluster test of those analyses

    @param groups: a dict of group labels and epoch filepaths within those groups.
                    e.g {'session1': files1, 'session2': files2}
    @param var1_events: event labels for analysis and plots
    @param var2_events: event labels for analysis and plots
    @param excluded_events:
    @param scoring:
    @param output_dir:
    @param jobs:
    """
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not Path(output_dir, 'indiv.pickle').exists():
        X_list = []
        for label, group in groups.items():
            X, y, times = MVPAnalysis(files=group, var1_events=var1_events, var2_events=var2_events,
                                      excluded_events=excluded_events, scoring=scoring,
                                      output_dir=output_dir + '/' + label, indiv_plot=False, jobs=jobs)
            X_list.append(X)
        with open(Path(output_dir, 'indiv.pickle'), 'wb') as f:
            pickle.dump([X_list, times], f)
    else:
        with open(Path(output_dir, 'indiv.pickle'), 'rb') as f:
            X_list, times = pickle.load(f)

    group_MVPA_and_plot(X_list, list(groups.keys()), var1_events, var2_events, times, output_dir, scoring, jobs)


def group_MVPA_and_plot(X_list, labels, var1_events, var2_events, times, output_dir='', scoring='roc_auc', jobs=-1):
    group_pvalues = cluster_stats_2samp(X_list, jobs)
    # Better plot with significance
    funcreturn, _ = decodingplot_group(scores_cond_group=X_list, p_values_cond=group_pvalues, times=times,
                                       labels=labels, alpha=0.05, tmin=times[0], tmax=times[-1])
    funcreturn.axes.set_title(f"{'-'.join(var1_events)}_vs_{'-'.join(var2_events)} - Sensor space decoding")
    funcreturn.axes.set_ylim(0.45, 0.75)
    funcreturn.axes.set_xlabel('Time (sec)')
    funcreturn.axes.set_ylabel(scoring)

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.savefig(Path(output_dir, "Group_Sensor-space-decoding_plot.png"), dpi=240)
    plt.close()


def decodingplot_group(scores_cond_group, p_values_cond, times, labels, alpha=0.05, colors=['r', 'b'], tmin=-0.8,
                       tmax=0.3):
    fig, ax1 = plt.subplots(nrows=1, figsize=[20, 4])
    split_ydata_group = []
    for group_n, scores_cond in enumerate(scores_cond_group):
        scores = np.array(scores_cond)
        sig = p_values_cond < alpha

        scores_m = np.nanmean(scores, axis=0)
        n = len(scores)
        n -= sum(np.isnan(np.mean(scores, axis=1)))  # identify the nan subjs and remove them.
        sem = np.nanstd(scores, axis=0) / np.sqrt(n)

        ax1.plot(times, scores_m, 'k', linewidth=1)
        ax1.fill_between(times, scores_m - sem, scores_m + sem, color=colors[group_n], alpha=0.3,
                         label=labels[group_n])

        split_ydata = scores_m
        split_ydata[~sig] = np.nan

        # shade the significant regions..
        ax1.plot(times, split_ydata, color='k', linewidth=3)
        split_ydata_group.append(split_ydata)

    ax1.fill_between(times, y1=split_ydata_group[0], y2=split_ydata_group[1], alpha=0.7, facecolor='g',
                     label=f'p<={alpha}')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax1.grid(True)
    ax1.legend()
    ax1.axhline(y=0.5, linewidth=0.75, color='k', linestyle='--')
    ax1.axvline(x=0, linewidth=0.75, color='k', linestyle='--')

    timeintervals = np.arange(tmin, tmax, 0.1)
    timeintervals = timeintervals.round(decimals=2)

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


def _stat_fun_1samp(x, sigma=0, method='relative'):
    """
    This secondary function reduces the time of computation of p-values and adjusts for small-variance values
    """
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


def cluster_stats_1samp(X, n_jobs=-1):
    """
    Cluster statistics to control for multiple comparisons. 1 sample t test
    Takes multiple individuals SVM outputs and determines if the mean is above chance

    adapted from https://github.com/kingjr/decod_unseen_maintenance/blob/master/scripts/base.py
    performs stats of the group level.
    X is usually nsubj x ntpts -> composed of mean roc scores per subj per time point.
    performs cluster stats on X to identify regions of tpts that have roc significantly differ from chance.


    Parameters
    ----------
    X : array, shape (n_samples, n_space, n_times) or (n_subjects, n_times)
        The data, chance is assumed to be 0.
    n_jobs : int
        The number of parallel processors.
    """
    n_subjects = len(X)
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    # this functions gets the t-values and performs a cluster permutation test on them to determine p-values..
    p_threshold = 0.05
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2, n_subjects - 1)
    print('number of subjects:', n_subjects)
    print('t-threshold is:', t_threshold)
    print('p-threshold is:', p_threshold)
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask', stat_fun=_stat_fun_1samp, n_permutations=2 ** 3, seed=1234,
        n_jobs=n_jobs, threshold=t_threshold)
    p_values_ = np.ones_like(X[0]).T
    # rearrange the p-value per cluster..
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    return np.squeeze(p_values_).T


def test_data_mvpa(scoring="roc_auc", var1_events=['auditory'], var2_events=['visual'], excluded_events=['buttonpress'],
                   indiv_plot=False):
    """
    Loads some MNE test data, preprocesses it and runs an MVPA analysis on it
    """
    filename = Path('../utils/mne_test_data-epo.fif')
    if filename.exists():
        epochs = mne.read_epochs(filename)

    else:
        sample_data_folder = mne.datasets.sample.data_path()
        sample_data_raw_file = (
                sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
        )
        raw = mne.io.read_raw_fif(sample_data_raw_file)
        from preprocessor import preprocess_with_faster
        event_dict = {
            "auditory/left": 1,
            "auditory/right": 2,
            "visual/left": 3,
            "visual/right": 4,
            "smiley": 5,
            "buttonpress": 32,
        }
        events = mne.find_events(raw, stim_channel="STI 014")
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)

        epochs, evoked_before, evoked_after, logdict, report = preprocess_with_faster(raw, events, event_ids=event_dict,
                                                                                      picks=picks, tmin=-0.2, bmax=0,
                                                                                      tmax=0.5, plotting=False,
                                                                                      report=None)
        epochs.save(filename)

    epochs.pick_types(eeg=True, exclude="bads")

    if indiv_plot:
        evoked = epochs.average(method='mean')
        evoked.plot()

    var1 = list(epochs[var1_events].event_id.values())
    var2 = list(epochs[var2_events].event_id.values())

    epochs = epochs[var1_events + var2_events]

    try:
        to_drop = list(epochs[excluded_events].event_id.values())
        epochs.drop([True if x in to_drop else False for x in list(epochs.events[:, 2])])
    except KeyError:
        pass

    plots = activity_map_plots(epochs, group1=var1_events, group2=var2_events, plot_significance=False)
    for key, plot in plots.items():
        if 'anim' in key:
            plot.save(f'../analyses/mne_test_data_{key}')
        else:
            plot.savefig(f'../analyses/mne_test_data_{key}', dpi=240)
        plot.clf()
        plt.close()

    X = epochs.get_data()  # EEG signals: n_epochs, n_eeg_channels, n_times
    y = epochs.events[:, 2]
    y[np.argwhere(np.isin(y, var1)).ravel()] = 0
    y[np.argwhere(np.isin(y, var2)).ravel()] = 1
    # X, y = len_match_arrays(X, y)
    print(X.shape, y.shape)
    score = temporal_decoding(X, y, scoring=scoring)
    filename = Path('../analyses/temporal_decode_test_data.png')
    plot_svm_scores(epochs.times, score, scoring, title=filename.stem, filename=filename)

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from math import floor, ceil
import mne
from mne.datasets import sample
from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    Scaler,
    cross_val_multiscore,
    LinearModel,
    get_coef,
    Vectorizer,
    CSP,
)
from random import shuffle
from tqdm import tqdm
import scipy
from scipy import stats
from mne.stats import ttest_1samp_no_p
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       permutation_cluster_1samp_test)

chance = .5
alpha = 0.05


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


def len_match_arrays(X, y, sanity_check=False):
    """
    In theory a glm or svm can learn to distinguish groups by a case imbalance.
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


def plot_svm_scores(times, scores, scoring="roc_auc", title='', p_values=None):
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


def temporal_decoding(times, X, y, filename, plotting=False, scoring="roc_auc", jobs=-1):
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

    @param plotting: whether to plot or not
    @param scoring: what sklearn scoring measure to use. See sklearn.metrics.get_scorer_names() for options
    @param X: 3d array of n_epochs, n_meg_channels, n_times
    @param y: array of epoch events
    @param filename: plot filename for saving
    @param jobs: number of processor cores to use (-1 uses maximum). Smaller number uses less RAM but takes longer.
    @return scores: array of SVM accuracy scores
    """
    # bins = 10
    # X = X[:, :, :480]
    # X = X.reshape((X.shape[0], X.shape[1], bins, round(X.shape[2] / bins)))

    # Make logistic regression pipeline
    # clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    time_decode = SlidingEstimator(clf, n_jobs=jobs, scoring=scoring, verbose=False)
    # set cross-validation to 5, so that 20% of test is validation and 80% is train data
    scores = cross_val_multiscore(time_decode, X, y, cv=5, n_jobs=jobs)
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    pvalues = significance_test(scores, model=time_decode, X_test=X, y_test=y, n_permutations=1000)

    plot_svm_scores(times, scores, scoring, Path(filename).stem, pvalues)
    plt.savefig(filename, dpi=150)
    if plotting:
        plt.show(block=True)

    return scores


def significance_test(scores, model, X_test, y_test, n_permutations=1000):
    """Get p values for individual data by permuting y and then scoring X 1000 times"""
    permutation_scores = np.zeros((n_permutations, X_test.shape[-1]))
    model.fit(X_test, y_test)

    for i in tqdm(range(n_permutations)):
        shuffle(y_test)
        permutation_score = model.score(X_test, y_test)
        permutation_scores[i, :] = permutation_score

    pvalues = []
    for sample_n in range(X_test.shape[-1]):
        pvalue = (np.sum(permutation_scores[:, sample_n] >= scores[sample_n]) + 1.0) / (n_permutations + 1)
        pvalues.append(pvalue)
    return pvalues


# def get_score(permutation_score, X_test, y_test, model):
#     shuffle(y_test)
#     permutation_score.append(model.score(X_test, y_test))
#     return permutation_score
#
# def significance_test_mp(scores, model, X_test, y_test, n_permutations=1000):
#     from multiprocessing import Pool
#     from functools import partial
#     model.fit(X_test, y_test)
#
#     permutation_scores = [[] for _ in range(n_permutations)]
#
#     processes = 4
#     with Pool(processes) as pool:
#         get_score_partial = partial(get_score, X_test=X_test, y_test=y_test, model=model)
#         permutation_scores = pool.map(get_score_partial, permutation_scores)
#         # permutation_scores = pool.map(lambda item: your_func(item, X_test, y_test, model), permutation_scores)
#
#     permutation_scores = np.vstack(permutation_scores)
#
#     pvalues = []
#     for sample_n in range(X_test.shape[-1]):
#         pvalue = (np.sum(permutation_scores[:, sample_n] >= scores[sample_n]) + 1.0) / (n_permutations + 1)
#         pvalues.append((pvalue))
#     return pvalues


def movingaverage(y, window_length):
    """
    https://github.com/SridharJagannathan/decAlertnessDecisionmaking_JNeuroscience2021/blob/main/Scripts/notebooks/Figure6_temporaldecodingresponses.ipynb
    """
    y_smooth = scipy.convolve(y, np.ones(window_length, dtype='float'), 'same') / scipy.convolve(np.ones(len(y)),
                                                                                                 np.ones(window_length),
                                                                                                 'same')
    return y_smooth


def temporal_decoding_with_smoothing(times, x_data, y, filename, plotting=False, scoring="roc_auc", jobs=-1):
    """
    https://github.com/SridharJagannathan/decAlertnessDecisionmaking_JNeuroscience2021/blob/main/Scripts/notebooks/Figure6_temporaldecodingresponses.ipynb
    """
    # make an estimator with scaling each channel by across its time pts and epochs..
    # model parameters estimated with logistic regression(classification)..
    # clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(max_iter=1000)))
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    # Sliding estimator with classification made across each time pt by training with the same time pt..
    time_decod = SlidingEstimator(clf, n_jobs=jobs, scoring='roc_auc')

    # compute the cross validation score..
    scores = cross_val_multiscore(time_decod, x_data, y, cv=5, n_jobs=jobs)

    # Mean scores across cross-validation splits..
    score = np.mean(scores, axis=0)
    # smooth datapoints to ensure no discontinuities..
    score = movingaverage(score, 10)

    # time_decod.fit(x_data, y)
    # coef = get_coef(time_decod, 'patterns_', inverse_transform=True)

    plot_svm_scores(times, score, scoring, Path(filename).stem)
    plt.savefig(filename, dpi=150)
    if plotting:
        plt.show(block=True)

    # # store the evoked patterns that are more neurophysiologically interpretable..
    # evoked = mne.EvokedArray(coef, X.info, tmin=X.times[0])
    # fpath = outputfolder + "s_" + subj_id + "_" + cond_str + "_scores"
    # with open(fpath, 'wb') as f:
    #     pickle.dump([score, epoch_clean.times], f)
    # fname = outputfolder + "s_" + subj_id + "_" + cond_str + "_evoked-ave.fif"
    # evoked.save(fname)

    return score


def _stats(X, connectivity=None, n_jobs=-1):
    """
    Cluster statistics to control for multiple comparisons.

    adapted from https://github.com/kingjr/decod_unseen_maintenance/blob/master/scripts/base.py
    performs stats of the group level.
    X is usually nsubj x ntpts -> composed of mean roc scores per subj per timepoint.
    performs cluster stats on X to identify regions of tpts that have roc significantly differ from chance.


    Parameters
    ----------
    X : array, shape (n_samples, n_space, n_times)
        The data, chance is assumed to be 0.
    connectivity : None | array, shape (n_space, n_times)
        The connectivity matrix to apply cluster correction. If None uses
        neighboring cells of X.
    n_jobs : int
        The number of parallel processors.
    """
    n_subjects = len(X)
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    # this functions gets the t-values and performs a cluster permutation test on them to determine p-values..
    p_threshold = 0.05
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
    print('number of subjects:', n_subjects)
    print('t-threshold is:', t_threshold)
    print('p-threshold is:', p_threshold)
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask', stat_fun=_stat_fun, n_permutations=2 ** 12, seed=1234,
        n_jobs=n_jobs, threshold=t_threshold)
    p_values_ = np.ones_like(X[0]).T
    # rearrange the p-value per cluster..
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    return np.squeeze(p_values_).T


def _stat_fun(x, sigma=0, method='relative'):
    """
    This secondary function reduces the time of computation of p-values and adjusts for small-variance values
    """
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


def test_data_mvpa():
    """
    Loads some MNE test data, preprocesses it and runs an MVPA analysis on it
    """
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

    scoring = "roc_auc"
    var1_events = ['auditory']
    var2_events = ['visual']
    excluded_events = ['buttonpress']
    indiv_plot = True

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

    X = epochs.get_data()  # EEG signals: n_epochs, n_eeg_channels, n_times
    y = epochs.events[:, 2]

    y[np.argwhere(np.isin(y, var1)).ravel()] = 0
    y[np.argwhere(np.isin(y, var2)).ravel()] = 1

    X, y = len_match_arrays(X, y)

    print(X.shape, y.shape)

    temporal_decoding(epochs.times, X, y, filename=f'../analyses/temporal_decode_test_data.png',
                      plotting=indiv_plot, scoring=scoring)


def MVPA_analysis(files, var1_events, var2_events, excluded_events=[], scoring="roc_auc", output_dir='',
                  indiv_plot=False, concat_participants=False, epochs_list=[], extra_event_labels=[], jobs=-1,
                  pickle_ouput=False):
    """
    Performs MVPA analysis over multiple participants.
    If you want to compare across multiple sessions concat_participants must be true

    @param output_dir: output directory for figures
    @param concat_participants: if true concatenate all epochs and run MVPA on that instead of individuals
    @param files: iterable or list of .epo.fif filepaths
    @param var1_events: list of event conditions for MVPA comparison
    @param var2_events: list of event conditions for MVPA comparison
    @param excluded_events: list of events to exclude from analysis
    @param scoring: scoring method for estimator. e.g: 'accuracy', 'roc_auc
    @param indiv_plot: whether to plot individual data
    @param epochs_list: If epochs are already loaded, use this instead of files
    @param extra_event_labels: list of lists
    @param jobs: number of processor cores to use (-1 uses maximum). Smaller number uses less RAM but takes longer.
    """

    scores_list = []
    X_list = []
    y_list = []

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for n, file in enumerate(files):
        if epochs_list:
            epochs = epochs_list[n]
        else:
            epochs = mne.read_epochs(file)
        epochs.pick_types(eeg=True, exclude="bads")

        if indiv_plot:
            evoked = epochs.average(method='mean')
            evoked.plot()

        if extra_event_labels:
            extra_labels = extra_event_labels[n]
            epochs.event_id = {recode_label(k, extra_labels): v for k, v in epochs.event_id.items()}

        try:
            var1 = list(epochs[var1_events].event_id.values())
        except KeyError:
            var1 = []
        try:
            var2 = list(epochs[var2_events].event_id.values())
        except KeyError:
            var2 = []

        epochs = epochs[var1_events + var2_events]

        if len(excluded_events) > 0:
            try:
                to_drop = list(epochs[excluded_events].event_id.values())
                epochs.drop([True if x in to_drop else False for x in list(epochs.events[:, 2])])
            except KeyError:
                pass

        X = epochs.get_data()  # EEG signals: n_epochs, n_eeg_channels, n_times

        y = epochs.events[:, 2]
        times = epochs.times
        print(times)
        quit()
        del epochs

        y[np.argwhere(np.isin(y, var1)).ravel()] = 0
        y[np.argwhere(np.isin(y, var2)).ravel()] = 1

        if pickle_ouput:
            # X, y = len_match_arrays(X, y)
            d = {'data': X, 'label': y, 'key': {0: "normal", 1: "global"}}
            with open(Path(output_dir, f'{Path(file).with_suffix("").stem}.pkl'), 'wb') as f:
                pickle.dump(d, f)

        elif concat_participants:
            X_list.append(X)
            y_list.append(y)
        else:
            X, y = len_match_arrays(X, y)
            scores = temporal_decoding_with_smoothing(times, X, y,
                                                      filename=Path(output_dir,
                                                                    f'temp_decod_{Path(file).with_suffix("").stem}.png'),
                                                      plotting=indiv_plot, scoring=scoring, jobs=jobs)
            scores_list.append(scores)

    if not pickle_ouput:
        if concat_participants:
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            X, y = len_match_arrays(X, y)
            print(X.shape)
            print(np.array(np.unique(y, return_counts=True)))
            scores = temporal_decoding_with_smoothing(times, X, y,
                                       filename=Path(output_dir,
                                                     f"group_{'-'.join(var1_events)}_vs_{'-'.join(var2_events)}.png".replace(
                                                         '/', '+')),
                                       plotting=indiv_plot, scoring=scoring, jobs=jobs)
            all_data = pd.DataFrame([scores], columns=times)
            all_data.to_csv(Path(output_dir, "all_data.csv"), index=False)
        else:
            X = np.vstack(scores_list)
            scores_pvalues = _stats(X - chance, n_jobs=jobs)
            print(scores_pvalues)

            m_scores = np.mean(scores_list, axis=0)
            plot_svm_scores(times, m_scores, scoring,
                            title=f"{'-'.join(var1_events)}_vs_{'-'.join(var2_events)} - Sensor space decoding")
            plt.savefig(Path(output_dir, "Mean_Sensor-space-decoding_plot.png"), dpi=150)
            # plt.show(block=True)

            all_data = pd.DataFrame(scores_list, columns=times)
            info = pd.DataFrame(extra_event_labels, columns=['ppt_num', 'sesh_num'])
            all_data = pd.concat([info, all_data], axis=1)
            all_data.to_csv(Path(output_dir, "all_data.csv"), index=False)

            # Better plot with significance
            funcreturn, _ = decodingplot(scores_cond=X, p_values_cond=scores_pvalues, times=times, rts=None,
                                                   alpha=0.05, color='r', tmin=times[0], tmax=times[-1])
            funcreturn.axes.set_ylim(0.45, 0.75)
            funcreturn.axes.set_xlabel('Time (sec)')
            funcreturn.axes.set_ylabel('AUC')
            plt.savefig(Path(output_dir, "Mean_Sensor-space-decoding_plot2.png"), dpi=240)


def decodingplot(scores_cond, p_values_cond, times, rts=None, alpha=0.05, color='r', tmin=-0.8, tmax=0.3):
    scores = np.array(scores_cond)
    sig = p_values_cond < alpha

    scores_m = np.nanmean(scores, axis=0)
    n = len(scores)
    n -= sum(np.isnan(np.mean(scores, axis=1)))  # identify the nan subjs and remove them..
    sem = np.nanstd(scores, axis=0) / np.sqrt(n)

    fig, ax1 = plt.subplots(nrows=1, figsize=[20, 4])

    ax1.plot(times, scores_m, 'k', linewidth=1, )
    ax1.fill_between(times, scores_m - sem, scores_m + sem, color=color, alpha=0.3)

    split_ydata = scores_m
    split_ydata[~sig] = np.nan

    # shade the significant regions..
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

    ax1.set_xticks(timeintervals)
    # ax1.axes.xaxis.set_ticklabels([])

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


if '__main__' in __name__:
    files, extra = get_filepaths_from_file('../analyses/MVPA/MVPA_analysis_list.csv')
    MVPA_analysis(files=files,
                  var1_events=['Obvious', 'Subtle'],
                  var2_events=['Normal'],
                  excluded_events=['Missed', 'Rate'], scoring="roc_auc",
                  output_dir='../analyses/MVPA',
                  indiv_plot=False,
                  concat_participants=False, epochs_list=[], extra_event_labels=[], jobs=-1,
                  pickle_ouput=False)
    # test_data_mvpa()

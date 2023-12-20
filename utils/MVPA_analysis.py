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
    print(x1.shape, x2.shape)
    if sanity_check:
        x1[:, :, 200:] = 1000000.0
    else:
        if x1.shape[0] > x2.shape[0]:
            idxs = np.random.choice(x1.shape[0], size=x2.shape[0], replace=False)
            x1 = x1[idxs, :, :]
        elif x2.shape[0] > x1.shape[0]:
            idxs = np.random.choice(x2.shape[0], size=x1.shape[0], replace=False)
            x2 = x2[idxs, :, :]
    print(x1.shape, x2.shape)
    assert x1.shape[0] == x2.shape[0]

    X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.zeros(x1.shape[0]), np.ones(x2.shape[0])])

    return X, y


def plot_svm_scores(times, scores, scoring="roc_auc", title=''):
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

    # Make logistic regression pipeline
    # clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    time_decode = SlidingEstimator(clf, n_jobs=jobs, scoring=scoring, verbose=False)
    # set cross-validation to 5, so that 20% of test is validation and 80% is train data
    scores = cross_val_multiscore(time_decode, X, y, cv=5, n_jobs=jobs)
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    plot_svm_scores(times, scores, scoring, str(filename.stem))
    plt.savefig(filename, dpi=150)
    if plotting:
        plt.show(block=True)

    return scores


def temporal_generalization(epochs, X, y, filename='temp_gen_plot.png', plotting=False, scoring="roc_auc"):
    """
    Temporal generalization is an extension of the decoding over time approach. It consists in evaluating whether the
    model estimated at a particular time instant accurately predicts any other time instant. It is analogous to
    transferring a trained model to a distinct learning problem, where the problems correspond to decoding the
    patterns of brain activity recorded at distinct time instants.

    The object to for Temporal generalization is mne.decoding.GeneralizingEstimator. It expects as input X and y
    (similarly to SlidingEstimator) but generates predictions from each model for all time instants. The class
    GeneralizingEstimator is generic and will treat the last dimension as the one to be used for generalization
    testing. For convenience, here, we refer to it as different tasks. If corresponds to epochs data then the last
    dimension is time.

    @param epochs: mne epochs object - used for x axis of plot
    @param X: 3d numpy array of n_epochs, n_meg_channels, n_times
    @param y: array of epoch events
    @param filename: plot filename for saving
    @param plotting: whether to plot or not
    @param scoring: what sklearn scoring measure to use. See sklearn.metrics.get_scorer_names() for options
    """

    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))  # 'rbf'
    # define the Temporal generalization object
    time_gen = GeneralizingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=True)

    scores = cross_val_multiscore(time_gen, X, y, cv=5, n_jobs=-1)

    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
    fig, ax = plt.subplots()
    ax.plot(epochs.times, np.diag(scores), label="score")
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(scoring)
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title("Decoding EEG sensors over time")

    plt.savefig(filename, dpi=150)
    plt.show(block=True)


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

    temporal_decoding(epochs, X, y, filename=f'../analyses/temporal_decode_test_data.png',
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
        else:
            extra_labels = None

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
            scores = temporal_decoding(times, X, y,
                                       filename=Path(output_dir, f'temp_decod_{Path(file).with_suffix("").stem}.png'),
                                       plotting=indiv_plot, scoring=scoring, jobs=jobs)
            # temporal_generalization(epochs, X, y)
            scores_list.append(scores)

    if not pickle_ouput:
        if concat_participants:
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            X, y = len_match_arrays(X, y)
            print(X.shape)
            print(np.array(np.unique(y, return_counts=True)))
            scores = temporal_decoding(times, X, y,
                              filename=Path(output_dir,f"group_{'-'.join(var1_events)}_vs_{'-'.join(var2_events)}.png".replace('/','+')),
                              plotting=indiv_plot, scoring=scoring, jobs=jobs)
            all_data = pd.DataFrame([scores], columns=times)
            all_data.to_csv(Path(output_dir, "all_data.csv"), index=False)
        else:
            m_scores = np.mean(scores_list, axis=0)
            plot_svm_scores(times, m_scores, scoring,
                            title=f"{'-'.join(var1_events)}_vs_{'-'.join(var2_events)} - Sensor space decoding")
            plt.savefig(Path(output_dir, "Mean_Sensor-space-decoding_plot.png"), dpi=150)
            # plt.show(block=True)

            all_data = pd.DataFrame(scores_list, columns=times)
            info = pd.DataFrame(extra_event_labels, columns=['ppt_num', 'sesh_num'])
            all_data = pd.concat([info, all_data], axis=1)
            all_data.to_csv(Path(output_dir, "all_data.csv"), index=False)


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




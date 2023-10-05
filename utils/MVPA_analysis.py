from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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


def recode_label(event, sep='/'):
    """

    @param event: epochs.event_id.items
    @param sep:
    @return:
    """

    e = event.split(sep)
    e[-1] = 'resp_' + e[-1]
    tags = sep.join(e)

    # labels = ['Rate', 'View',
    #           'Global', 'Obvious', 'Subtle', 'Normal',
    #           'Correct', 'Incorrect', 'Missed',
    #           'PerceivedAbnormal', 'PerceivedN']
    #
    # event = event.replace('PerceivedNormal', 'PerceivedN')
    # label_list = []
    # for label in labels:
    #     if label in event:
    #         label_list.append(label)
    # tags = sep.join(label_list)
    # tags = tags.replace('PerceivedN', 'PerceivedNormal')
    return tags


def len_match_arrays(X, y, sanity_check=False):
    """

    @param X: 3d epochs data array.
    @param y: 1d epochs binary event type array.
    @return: len matched X,y.
    """

    x1 = X[np.where(y == 0)[0], :, :]
    # x2 = x1.copy()
    x2 = X[np.where(y == 1)[0], :, :]

    x1 = x1[:, :, :]
    x1 = x1[:, :, :]

    if sanity_check:
        x1[:, :, 200:] = 1000000.0
    else:
        if x1.shape[0] > x2.shape[0]:
            idxs = np.random.choice(x1.shape[0], size=x2.shape[0], replace=False)
            x1 = x1[idxs, :, :]
        elif x2.shape[0] > x1.shape[0]:
            idxs = np.random.choice(x2.shape[0], size=x1.shape[0], replace=False)
            x2 = x2[idxs, :, :]
    assert x1.shape[0] == x2.shape[0]

    X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.zeros(x1.shape[0]), np.ones(x2.shape[0])])

    return X, y


def temporal_decoding(epochs, X, y, filename, plotting=False, scoring="roc_auc"):
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
    @param epochs: mne epochs object
    @param X: 3d array of n_epochs, n_meg_channels, n_times
    @param y: array of epoch events
    @param filename: plot filename for saving
    """

    # Make logistic regression pipeline
    # clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    time_decode = SlidingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=False)
    # set cross-validation to 5, so that 20% of data is validation and 80% is test data
    scores = cross_val_multiscore(time_decode, X, y, cv=5, n_jobs=-1)
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    fig, ax = plt.subplots()
    ax.plot(epochs.times, scores, label="score")
    ax.axhline(0.5, color="k", linestyle="--", label="chance")

    ax.set_ylim([0, 1])

    ax.set_xlabel("Times")
    ax.set_ylabel(scoring)  # Area Under the Curve
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title("Sensor space decoding")
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
    @return:
    """

    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))  # 'rbf'
    # define the Temporal generalization object
    time_gen = GeneralizingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=True)

    # again, cv=3 just for speed
    scores = cross_val_multiscore(time_gen, X, y, cv=5, n_jobs=-1)

    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
    fig, ax = plt.subplots()
    ax.plot(epochs.times, np.diag(scores), label="score")
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.set_xlabel("Times")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title("Decoding MEG sensors over time")

    plt.savefig(filename, dpi=150)
    plt.show(block=True)


def MVPA_analysis(files, var1_events, var2_events, excluded_events, scoring="roc_auc", indiv_plot=False,
                  sanity_check=False, epochs_list=[]):
    scores_list = []
    if len(epochs_list) > 0:
        files = epochs_list
    for file in files:
        epochs = mne.read_epochs(file)

        epochs.pick_types(eeg=True, exclude="bads")

        # if indiv_plot:
        #     evoked = epochs.average(method='mean')
        #     evoked.plot()

        epochs.event_id = {recode_label(k): v for k, v in epochs.event_id.items()}
        print(epochs.event_id)

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

        X, y = len_match_arrays(X, y, sanity_check=False)

        # rng = ''
        # if sanity_check:
        #     # X = np.random.normal(0, 1, size=X.shape)
        #     # y = np.concatenate((np.zeros(ceil(len(y) / 2)), np.ones(floor(len(y) / 2))))
        #     np.random.shuffle(y)
        #     rng = '_rng'

        print(X.shape, y.shape)

        scores = temporal_decoding(epochs, X, y, filename=f'../analyses/temp_decod_{file.with_suffix("").stem}.png',
                                   plotting=indiv_plot, scoring=scoring)
        # temporal_generalization(epochs, X, y)
        scores_list.append(scores)

    m_scores = np.mean(scores_list, axis=0)

    fig, ax = plt.subplots()
    ax.plot(epochs.times, m_scores, label="score")
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.set_xlabel("Time")
    ax.set_ylabel(scoring)
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title("Sensor space decoding")
    plt.savefig(f"../analyses/{'-'.join(var1_events)}_vs_{'-'.join(var2_events)}.png", dpi=150)
    plt.show(block=True)


def test_data_mvpa():
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
    print(len(picks))

    epochs, evoked_before, evoked_after, logdict, report = preprocess_with_faster(raw, events, event_ids=event_dict,
                                                                                  picks=picks, tmin=-0.2, bmax=0,
                                                                                  tmax=0.5, plotting=False,
                                                                                  report=None)

    scoring = "roc_auc"
    var1_events = ['resp_left']
    var2_events = ['resp_right']
    excluded_events = ['smiley', 'buttonpress']
    indiv_plot = True

    epochs.pick_types(eeg=True, exclude="bads")

    if indiv_plot:
        evoked = epochs.average(method='mean')
        evoked.plot()

    epochs.event_id = {recode_label(k): v for k, v in epochs.event_id.items()}
    print(epochs.event_id)

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

    X, y = len_match_arrays(X, y, sanity_check=True)

    print(X.shape, y.shape)

    scores = temporal_decoding(epochs, X, y, filename=f'../analyses/temp_decod_test_data.png',
                               plotting=indiv_plot, scoring=scoring)


if '__main__' in __name__:
    # test_data_mvpa()
    # quit()
    files = Path(
        '/Volumes/psgroups/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Radiologists/output/').glob(
        'EEGTraining_Rad*.epo.fif')
    scoring = "roc_auc"
    # var1_events = ['Normal']
    # var2_events = ['Obvious', 'Subtle']

    var1_events = ['Obvious/resp_Abnormal', 'Subtle/resp_Abnormal']
    var2_events = ['Normal/resp_Normal']
    excluded_events = ['Rate', 'Missed']

    MVPA_analysis(files, var1_events, var2_events, excluded_events, scoring, indiv_plot=True, sanity_check=False)

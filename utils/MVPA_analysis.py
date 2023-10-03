import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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


def csp_score(epochs, X, y):
    """
    Common spatial pattern (CSP) algorithm
    https://en.wikipedia.org/wiki/Common_spatial_pattern

    @param epochs: mne epochs object
    @param X: 3d array of n_epochs, n_meg_channels, n_times
    @param y: array of epoch events
    @return:
    """
    csp = CSP(n_components=3, norm_trace=False)
    clf_csp = make_pipeline(csp, LinearModel(LogisticRegression(solver="liblinear")))
    scores = cross_val_multiscore(clf_csp, X, y, cv=5, n_jobs=None)
    print("CSP: %0.1f%%" % (100 * scores.mean(),))

    csp.fit(X, y)
    fig_pattern = csp.plot_patterns(epochs.info)
    fig_filters = csp.plot_filters(epochs.info, scalings=1e-9)
    plt.figure(fig_filters.number)
    plt.show(block=True)


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

    @param scoring: what sklearn scoring measure to use. See sklearn.metrics.get_scorer_names() for options
    @param epochs: mne epochs object
    @param X: 3d array of n_epochs, n_meg_channels, n_times
    @param y: array of epoch events
    @param filename: plot filename for saving
    """

    # Make logistic regression pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    time_decode = SlidingEstimator(clf, n_jobs=8, scoring=scoring, verbose=True)
    # set cross-validation to 5, so that 20% of data is validation and 80% is test data
    scores = cross_val_multiscore(time_decode, X, y, cv=5, n_jobs=None)
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    if plotting:
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
        plt.show(block=True)

    return scores


def spatiotemporal_decoding_plot(epochs, X, y):
    """

    @param epochs: mne epochs object
    @param X: 3d array of n_epochs, n_meg_channels, n_times
    @param y: array of epoch events
    @return:
    """
    clf = make_pipeline(
        StandardScaler(), LinearModel(LogisticRegression(solver="liblinear"))
    )
    time_decode = SlidingEstimator(clf, n_jobs=None, scoring="roc_auc", verbose=True)
    time_decode.fit(X, y)

    coef = get_coef(time_decode, "patterns_", inverse_transform=True)
    evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
    joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))
    evoked_time_gen.plot_joint(
        times=np.arange(0.0, 1.000, 0.200), title="patterns", **joint_kwargs
    )


if '__main__' in __name__:
    sanity_check = False
    pnum = 1
    # epochs = mne.read_epochs(f'/Volumes/psgroups/AttentionPerceptionLabStudent/PROJECTS/EEG_MAMMO_EXPERMENT'
    #                          f'/MNE_preprocessing_db/EEGTraining_Rad{pnum}/EEGTraining_Rad{pnum}_epochs_output.fif')
    scores_list = []
    for pnum in range(1, 6):
        epochs = mne.read_epochs(
            f'/Volumes/psgroups/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Radiologists/output/EEGTraining_Rad{pnum}.epo.fif')
        epochs.pick_types(eeg=True, exclude="bads")
        epochs.event_id = {recode_label(k): v for k, v in epochs.event_id.items()}
        print(epochs.event_id)
        # epochs.plot(n_epochs=1, butterfly=True, block=True)

        var1_events = ['Normal']
        var2_events = ['Obvious']

        var1 = list(epochs[var1_events].event_id.values())
        var2 = list(epochs[var2_events].event_id.values())

        epochs = epochs[var1_events + var2_events]

        # # var_exclude = ['Rate', 'Missed']
        # to_drop = list(epochs[var_exclude].event_id.values())
        # epochs.drop([True if x in to_drop else False for x in list(epochs.events[:, 2])])

        X = epochs.get_data()  # EEG signals: n_epochs, n_eeg_channels, n_times
        print(X.shape)

        if sanity_check:
            X = np.random.normal(0, 1, size=X.shape)
        y = epochs.events[:, 2]
        print(epochs.events)

        y[np.argwhere(np.isin(y, var1)).ravel()] = 0
        y[np.argwhere(np.isin(y, var2)).ravel()] = 1

        zs = np.zeros(len(y/2))
        os = np.ones(len(y/2))

        # csp_score(epochs, X, y)
        scores = temporal_decoding(epochs, X, y, filename=f'../analyses/temporal_decoding_Rad{pnum}_1.png',
                                   plotting=False, scoring='accuracy')
        scores_list.append(scores)
    m_scores = np.mean(scores_list, axis=0)

    fig, ax = plt.subplots()

    ax.plot(epochs.times, m_scores, label="score")
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.set_xlabel("Time")
    ax.set_ylabel('Mean Accuracy')
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title("Sensor space decoding")
    plt.savefig(f"{'-'.join(var1_events)}_vs_{'-'.join(var2_events)}.png", dpi=150)
    plt.show(block=True)
    # spatiotemporal_decoding_plot(epochs, X, y)

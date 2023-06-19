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
    labels = ['Rate', 'View',
              'Global', 'Obvious', 'Subtle', 'Normal',
              'Correct', 'Incorrect', 'Missed',
              'PerceivedAbnormal', 'PerceivedN']

    event = event.replace('PerceivedNormal', 'PerceivedN')
    label_list = []
    for label in labels:
        if label in event:
            label_list.append(label)
    tags = sep.join(label_list)
    tags = tags.replace('PerceivedN', 'PerceivedNormal')
    return tags


def csp_score(epochs, X, y):
    csp = CSP(n_components=3, norm_trace=False)
    clf_csp = make_pipeline(csp, LinearModel(LogisticRegression(solver="liblinear")))
    scores = cross_val_multiscore(clf_csp, X, y, cv=5, n_jobs=None)
    print("CSP: %0.1f%%" % (100 * scores.mean(),))

    csp.fit(X, y)
    csp.plot_patterns(epochs.info)
    csp.plot_filters(epochs.info, scalings=1e-9)
    plt.show(block=True)


def temporal_decoding_plot(epochs, X, y):
    # We will train the classifier on all left visual vs auditory trials on MEG

    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))

    time_decod = SlidingEstimator(clf, n_jobs=None, scoring="roc_auc", verbose=True)
    # here we use cv=3 just for speed
    scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=None)

    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(epochs.times, scores, label="score")
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.set_xlabel("Times")
    ax.set_ylabel("AUC")  # Area Under the Curve
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title("Sensor space decoding")
    plt.show(block=True)


def temporal_decoding_plot_2(epochs, X, y):
    clf = make_pipeline(
        StandardScaler(), LinearModel(LogisticRegression(solver="liblinear"))
    )
    time_decod = SlidingEstimator(clf, n_jobs=None, scoring="roc_auc", verbose=True)
    time_decod.fit(X, y)

    coef = get_coef(time_decod, "patterns_", inverse_transform=True)
    evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
    joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))
    evoked_time_gen.plot_joint(
        times=np.arange(0.0, 1.000, 0.200), title="patterns", **joint_kwargs
    )


if '__main__' in __name__:
    pnum = 5
    epochs = mne.read_epochs(f'/Volumes/psgroups/AttentionPerceptionLabStudent/PROJECTS/EEG_MAMMO_EXPERMENT'
                             f'/MNE_preprocessing_db/EEGTraining_Rad{pnum}/EEGTraining_Rad{pnum}_epochs_output.fif')

    epochs.pick_types(eeg=True, exclude="bads")
    epochs.apply_baseline(baseline=(None, -0.2))
    # epochs.crop(tmin=0)
    epochs.event_id = {recode_label(k): v for k, v in epochs.event_id.items()}

    # epochs.plot(n_epochs=1, butterfly=True, block=True)

    var1_events = 'View/Normal'
    var2_events = ['View/Obvious', 'View/Subtle', 'View/Global']
    var_exclude = ['Rate', 'Missed']

    # var1_events = 'Rate'
    # var2_events = 'View'

    to_drop = list(epochs[var_exclude].event_id.values())
    epochs.drop([True if x in to_drop else False for x in list(epochs.events[:, 2])])

    var1 = list(epochs[var1_events].event_id.values())
    var2 = list(epochs[var2_events].event_id.values())

    X = epochs.get_data()  # MEG signals: n_epochs, n_meg_channels, n_times
    y = epochs.events[:, 2]  # target: auditory left vs visual left
    y[np.argwhere(np.isin(y, var1)).ravel()] = 0
    y[np.argwhere(np.isin(y, var2)).ravel()] = 1

    csp_score(epochs, X, y)
    temporal_decoding_plot(epochs, X, y)



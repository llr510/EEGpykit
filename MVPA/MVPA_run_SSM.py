from pathlib import Path

import mne
from MVPA.MVPAnalysis import MVPAnalysis


def load_data(dest):
    files = list(dest.glob('*-epo.fif'))
    files = files[0:10]
    extra = []
    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file, verbose=False)
        epochs_list.append(epochs)
        extra.append([file.stem.split('-')[0]])

    return files, epochs_list, extra


def run_analysis(dest, files, epochs_list, extra):

    MVPAnalysis(files, var1_events=['single/T1_hit'],
                var2_events=['dual/T1_hit/T2_hit'],
                scoring="roc_auc",
                output_dir=Path(dest, f'OneTargetvsTwo'), indiv_plot=False,
                epochs_list=epochs_list, extra_event_labels=extra, jobs=4,
                overwrite_output=True)


if '__main__' in __name__:
    dest = Path('/Users/lyndonrakusen/Library/CloudStorage/Box-Box/AdamoLab/SSM_EEG_Analysis')
    files, epochs_list, extra = load_data(dest)
    run_analysis(dest, files, epochs_list, extra)
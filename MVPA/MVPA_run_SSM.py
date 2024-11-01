from pathlib import Path

import mne
from MVPA.MVPAnalysis import MVPAnalysis


def load_data(dest):
    files = list(dest.glob('*-epo.fif'))
    files = files[0:4]
    extra = []
    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file, verbose=False)
        epochs_list.append(epochs)
        extra.append([file.stem.split('-')[0]])

    return files, epochs_list, extra


def run_analysis(dest):
    files, epochs_list, extra = load_data(dest)

    MVPAnalysis(files, var1_events=['single'],
                var2_events=['dual'],
                scoring="roc_auc",
                output_dir=Path(dest, f'SingleVSdual'), indiv_plot=False,
                epochs_list=epochs_list, extra_event_labels=extra, jobs=4,
                overwrite_output=True)


if '__main__' in __name__:
    dest = Path('/Users/lyndonrakusen/Library/CloudStorage/Box-Box/AdamoLab/SSM_EEG_Analysis')
    run_analysis(dest)
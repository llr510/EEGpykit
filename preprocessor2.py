from pathlib import Path
import mne
from preprocessor import preprocess_with_faster
from utils.read_custom_brainvision import read_stephens_data
import numpy as np
import pandas as pd

def trial_labels_maker(events, label_file = None):
    events = events[0]
    df = pd.read_csv(label_file)
    label_dict = {}
    for idx, row in df.iterrows():
        if type(row.label) == str:
            label_dict[row.trigger] = row.label

    start_trigger = 9
    trial_events = np.split(events, np.where(events[:, 2] == start_trigger)[0][1:])
    # print(trial_events)

    event_list = []
    for trial in trial_events:
        # Get the sample number for the second trigger after the start_trigger
        samples = trial[np.where(trial[:, 2] == start_trigger)[0] +1 ,0]

        trial_triggers = trial[:, 2]
        trial_labels = []
        for x in trial_triggers:
            l = label_dict.get(x)
            if l:
                trial_labels.append(l)
            else:
                print(f'missing trigger label: {x}')
                trial_labels.append(str(x))

        label = '/'.join(trial_labels)
        event_list.append(dict(label=label, start=int(samples), end=0))
    event_df = pd.DataFrame(event_list)
    return event_df

def preprocess_one(header_path, output_path=None, report_path=None, verbose=False):
    raw, events, picks = read_stephens_data(header_path, verbose=verbose)
    df = trial_labels_maker(events, label_file='/Users/lyndonrakusen/Library/CloudStorage/Box-Box/AdamoLab/SSM_EEG_Analysis/trigger_labels.csv')
    my_annot = mne.Annotations(onset=df['start']/raw.info['sfreq'], duration=df['end'], description=df['label'])
    raw.set_annotations(my_annot)
    events, event_ids = mne.events_from_annotations(raw)

    report = mne.Report(title=header_path.name)
    report.add_raw(raw=raw, title='Raw before filtering')
    epochs, evoked_before, evoked_after, _, report = preprocess_with_faster(raw, events, event_ids, picks,
                                                                            tmin=-.200, tmax=.600, plotting=verbose,
                                                                            report=report)
    if report_path:
        report.save(report_path, overwrite=True, open_browser=verbose)
    if output_path:
        epochs.save(output_path, overwrite=True)
    if verbose:
        epochs.plot(block=True)
    return epochs


if '__main__' in __name__:
    wd = Path('/Users/lyndonrakusen/Documents/Data/OSF Uploads for '
                       'Lyndon/EEG_Attentional_Template_SSM_NCI_Experiment1_K99_40/SSM_EEG/')
    header_paths = wd.glob('*.vhdr')
    dest = Path('/Users/lyndonrakusen/Library/CloudStorage/Box-Box/AdamoLab/SSM_EEG_Analysis')

    for header_path in header_paths:
        print(header_path.stem)
        output_path = Path(dest, header_path.stem + '-epo.fif')
        report_path = Path(dest, header_path.stem + '.html')
        epochs = preprocess_one(header_path, output_path, report_path, verbose=False)


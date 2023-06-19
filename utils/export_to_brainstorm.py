from pathlib import Path
import mne
from preprocesser import EEG_Participant
import pickle
import pandas as pd


def recode_label(event):
    labels = ['Rate', 'View',
              'Global', 'Obvious', 'Subtle', 'Normal',
              'Correct', 'Incorrect', 'Missed',
              'PerceivedAbnormal', 'PerceivedN']

    event = event.replace('PerceivedNormal', 'PerceivedN')
    label_list = []
    for label in labels:
        if label in event:
            label_list.append(label)
    tags = ' | '.join(label_list)
    tags = tags.replace('PerceivedN', 'PerceivedNormal')
    return tags


if '__main__' in __name__:
    rec = 2
    # f'EEGTraining_Rad{n}/EEGTraining_Rad{n}_epochs_output.pickle'
    wd = Path('/Volumes/psgroups/AttentionPerceptionLabStudent/PROJECTS/EEG_MAMMO_EXPERMENT/MNE_preprocessing_db/')

    for n in range(1, 17):
        if n == 3:
            continue
        inf = Path(wd, f'EEGTraining_Sub{n}Rec{rec}/EEGTraining_Sub{n}Rec{rec}_epochs_output.pickle')

        outf = inf.with_suffix('.fif')

        with open(inf, 'rb') as f:
            dat = pickle.load(f)

        if type(dat) == dict:
            epochs = dat['all']

        ids = epochs.events[:, 2]
        assert len(ids) == len(epochs)
        labels = epochs.event_id
        labels = {v: k for k, v in labels.items()}

        tags = []
        epochs_n = []
        for i in range(len(ids)):
            epochs_n.append(i + 1)
            event = labels[ids[i]]

            tag = recode_label(event)
            tags.append(tag)

        df = pd.DataFrame({'epoch': epochs_n, 'label': tags})
        df.to_csv(inf.with_suffix('.csv'), index=False)

        epochs.times.flags['WRITEABLE'] = False
        epochs.save(outf, overwrite=True)

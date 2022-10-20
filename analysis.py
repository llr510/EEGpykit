from pathlib import Path

import mne

from preprocesser import EEG_Participant, EEG_Experiment
import matplotlib

gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui, force=True)
        from matplotlib import pyplot as plt

        break
    except:
        continue
print("Using:", matplotlib.get_backend())

output_path = Path(
    '/Volumes/psgroups/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK/MNE_preprocessing_db')


participants = output_path.glob('*.pickle')
ppts = []
for pth in participants:
    participant = EEG_Participant().load(Path(output_path, '20220131_1255ppt1.pickle'))

    ppts.append(participant['T2'].epochs.average())

mne.grand_average(ppts).plot()
# participant.epochs['T1'].average().plot()
# participant.epochs['T2/NS/scene'].average().plot()
# participant.epochs['T2/NS/dot'].average().plot()
# participant.epochs['T2/S/scene'].average().plot()
# participant.epochs['T2/S/dot'].average().plot()

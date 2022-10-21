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
participants = list(participants)
# participants[:3]
ppts = {'T2': [], 'T1': [], 'T2/attentional_blink': [], 'T2/correct': []}
for pth in participants:
    participant = EEG_Participant().load(pth)
    for events_group in ppts.keys():
        epochs = participant.get_epochs(by_events=events_group)
        ppts[events_group].append(epochs.average())

ppts = {events: mne.grand_average(evokeds) for events, evokeds in ppts.items()}
for events, evoked in ppts.items():
    events = events.replace('/', '-')
    evoked.plot_joint(title=events, show=False).savefig(Path(f'joint_{events}.jpg'))

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
    '/Volumes/psgroups-1/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK/MNE_preprocessing_db')

participants = output_path.glob('*.pickle')
participants = list(participants)

event_dict = {'T2': [],
              'T1': [],
              'T2/scene/attentional_blink': [],
              'T2/scene/correct': [],
              'T2/scene/incorrect': [],
              'dot/NS-S': [],
              'dot/S-NS': [],
              'scene/NS-S': [],
              'scene/S-NS': []}

for pth in participants:
    participant = EEG_Participant.load(pth)
    for events_group in event_dict.keys():
        epochs = participant.get_epochs(by_events=events_group)
        event_dict[events_group].append(epochs.average())

event_dict = {events: mne.grand_average(evokeds) for events, evokeds in event_dict.items()}
for events, evoked in event_dict.items():
    events = events.replace('/', '-')
    evoked.plot_joint(title=events, show=False).savefig(Path('figures', f'joint_{events}.jpg'))

from pathlib import Path
import mne
from preprocesser import EEG_Participant
import pickle

inf = Path(
    '/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/PROJECTS/EEG_MAMMO_EXPERMENT/MNE_preprocessing_db/EEGTraining_Rad1/EEGTraining_Rad1_epochs_output.pickle')

outf = Path('test-epo.fif')

with open(inf, 'rb') as f:
    dat = pickle.load(f)

if type(dat) == dict:
    epochs = dat['all']

print(epochs)
print(epochs.events)
print(epochs.event_id)
epochs.times.flags['WRITEABLE'] = False
epochs.save(outf, overwrite=True)

# mne.write_events('events-eve.fif', epochs.events, overwrite=True)
# mne.write_events('events-eve.txt', epochs.events, overwrite=True)

# e = mne.read_epochs(outf)
#
# print(e)
# print(e.event_id)

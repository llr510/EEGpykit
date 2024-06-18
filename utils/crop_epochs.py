import mne
from pathlib import Path


def crop_epochs(source, dest, new_tmin, new_tmax):
    epochs = mne.read_epochs(source)
    print('Before: ', epochs.tmin, epochs.tmax)
    epochs.crop(new_tmin, new_tmax)
    print('After: ', epochs.tmin, epochs.tmax)
    epochs.save(dest, overwrite=False)


source_dir = '/mnt/scratch/projects/psy-opt-2022/EEGprojects/data/Mammography/epoched_2k'
dest_dir = '/mnt/scratch/projects/psy-opt-2022/EEGprojects/data/Mammography/epoched'
orignals = Path(source_dir).glob('*.fif')

for source in orignals:
    print(source.name)
    dest = Path(dest_dir, source.name)
    crop_epochs(source, dest, 0, 1000)

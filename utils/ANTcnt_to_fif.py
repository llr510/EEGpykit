from pathlib import Path
from utils.read_antcnt import read_raw_antcnt


def cnt_to_fif(rawf, fiff):
    """
    Read ANT Neuro .cnt file and convert it to a mne raw.fif

    @param rawf: cnt raw filepath
    @param fiff: fif raw filepath
    @return:
    """
    raw = read_raw_antcnt(str(rawf), preload=True, eog=['VEOG', 'HEOG'])
    raw.save(fiff, overwrite=True)


if '__main__' in __name__:
    # directory = '/Volumes/psgroups-1/AttentionPerceptionLab/AttentionPerceptionLabStudent/PROJECTS/EEG_MAMMO_EXPERMENT/eeg data/'
    # filename = 'EEGTraining_Sub*Rec*.cnt'
    # files = Path(directory).rglob(filename)
    # print(Path(
    #     '/Volumes/psgroups/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Naives/EEG/').exists())
    #
    # for file in files:
    #     output = Path('/Volumes/psgroups/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Naives/EEG/', file.stem+'.fif')
    #     if output.exists():
    #         continue
    #
    #     cnt_to_fif(file, output)


    directory = '/Volumes/psgroups-1/AttentionPerceptionLab/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK/DATA/'
    filename = '*.cnt'
    files = Path(directory).rglob(filename)
    for file in files:
        print(file.name)
        output = Path('/Volumes/psgroups-1/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/AB/EEG/', file.stem+'.fif')
        if output.exists():
            continue

        cnt_to_fif(file, output)

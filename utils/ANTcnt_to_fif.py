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
    for n in range(2, 6):
        cnt_to_fif(
            f"/Volumes/psgroups/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Radiologists/EEG/EEGTraining_Rad{n}.cnt",
            f"/Volumes/psgroups/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Radiologists/EEG/EEGTraining_Rad{n}.fif"
        )

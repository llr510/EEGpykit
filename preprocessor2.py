from pathlib import Path
import mne
from preprocessor import preprocess_with_faster
from utils.read_custom_brain_vision import read_stephens_data

if '__main__' in __name__:

    working_dir = Path('/Users/lyndonrakusen/Documents/Data/OSF Uploads for Lyndon/EEG_Attentional_Template_SSM_NCI_Experiment1_K99_40/SSM_EEG/')
    data_pth = Path(working_dir, 'SID4.vhdr')
    raw, events, picks = read_stephens_data(data_pth,verbose=True)
    event_ids = {'t1/T1':10,
                 't1/T2':20,
                 't1/T3':30,
                 't1/T4':40,
                 't1/L1':50,
                 't1/L2':60,
                 't1/L3':70,
                 't1/L4':80}
    report = mne.Report(title='SID4')
    report.add_raw(raw=raw, title='Raw before filtering')
    epochs, evoked_before, evoked_after, _, report = preprocess_with_faster(raw, events[0], event_ids, picks,
                                                                                  tmin=-.200, tmax=.800, plotting=False,
                                                                                  report=report)

    report.save(data_pth.with_suffix('.html'), overwrite=True, open_browser=True)
    epochs.plot(block=True)


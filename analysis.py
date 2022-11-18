from pathlib import Path
import mne
from preprocesser import EEG_Participant
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


class BaseAnalysis:
    def __init__(self, analysis_db, data_db):
        self.analysis_db = analysis_db
        self.data_db = data_db
        self.selected_events = None
        self.evoked = None
        self.epochs = None

    def select_events(self, events):
        self.selected_events = events

    def epochs_to_evoked(self):
        self.evoked = self.epochs[self.selected_events].average()

    def plot_channels(self):
        pass

    def plot_heatmaps(self, title, anim=False, show=True, save=True):
        times = [x / 10 for x in range(0, int(self.evoked.tmax * 10) + 1)]
        joint = self.evoked.plot_joint(title=title, show=show)
        topo = self.evoked.plot_topomap(times=times, ch_type='eeg', title=title, show=show)

        if save:
            output = Path(self.analysis_db, 'figs', 'joint', f'joint_{title}.jpg')

            output.parent.mkdir(parents=True, exist_ok=True)
            joint.savefig(output)
            output = Path(self.analysis_db, 'figs', 'topo', f'topo_{title}.jpg')
            output.parent.mkdir(parents=True, exist_ok=True)
            topo.savefig(output)

        if anim:
            times = [x / 100 for x in range(0, 101)]
            fig, anim = self.evoked.animate_topomap(times=times, ch_type='eeg', frame_rate=12, show=show, blit=False,
                                                    time_unit='s')
            if save:
                anim.save(Path(self.analysis_db, 'figs', 'topo', f'topo_{title}.gif'))


class Group(BaseAnalysis):
    def __init__(self, analysis_db, data_db, individual_list):
        super().__init__(analysis_db, data_db)
        self.individual_list = individual_list

    def load_epochs(self):
        epoch_objects = []
        id_offset = 0
        for individual in self.individual_list:
            id_offset += 100
            data = EEG_Participant.load(individual).epochs

            data.event_id = {f'{individual.stem}/' + label: id_val + id_offset
                             for label, id_val in data.event_id.items()}
            data.events[:, 2] = data.events[:, 2] + id_offset

            epoch_objects.append(data)

        self.epochs = mne.epochs.concatenate_epochs(epochs_list=epoch_objects, add_offset=True)
        del epoch_objects, data


class Individual(BaseAnalysis):
    def __init__(self, analysis_db, data_db, filename):
        super().__init__(analysis_db, data_db)
        self.pickle_path = Path(data_db, filename)

    def load_epochs(self):
        data = EEG_Participant.load(self.pickle_path)
        self.epochs = data.epochs


wd = Path(
    '/Volumes/psgroups-1/AttentionPerceptionLab/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK')
data_db = Path(wd, 'MNE_preprocessing_db')
analysis_db = Path(wd, 'MNE_analysis_db')

individual_list = [Path(data_db, pickle) for pickle in ['20221110_0937PPT24_scenes.pickle',
                                                        '20221103_1247_PPT23_scenes.pickle',
                                                        '20220131_1255ppt1.pickle',
                                                        '20220202_0830PPT2.pickle',
                                                        '20220203_1225PPT3.pickle']]

analysis = Group(analysis_db=analysis_db, data_db=data_db, individual_list=individual_list)
# analysis = Individual(analysis_db=analysis_db, data_db=data_db, filename='20221110_0937PPT24_scenes.pickle')
analysis.select_events('T2/correct')
analysis.load_epochs()
analysis.epochs_to_evoked()
analysis.plot_heatmaps(title='test')

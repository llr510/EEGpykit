import pickle
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
        self.window = None
        self.channels = None

    def select_events(self, events, drop_unselected=False):
        self.selected_events = events
        if drop_unselected:
            self.epochs = self.epochs[self.selected_events]

    def select_window(self):
        pass

    def select_channels(self):
        pass

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

    # def save(self, filename):
    #     """Save object as .pickle."""
    #
    #     with open(filename, 'wb') as f:
    #         pickle.dump(self, f)
    #
    # @classmethod
    # def load(cls, filename):
    #     """Load object.
    #
    #     Parameters
    #     ----------
    #     filename : str or PosixPath
    #     """
    #     with open(filename, 'rb') as f:
    #         print(f'loading {filename} from pickle...')
    #         return pickle.load(f)


class Group(BaseAnalysis):
    def __init__(self, analysis_db, data_db, individual_list):
        super().__init__(analysis_db, data_db)
        self.individual_list = individual_list
        self.evokeds = None
        self.grand_evoked = None

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

    def individuals_to_evokeds(self):
        self.evokeds = []
        for individual in self.individual_list:
            self.evokeds.append = (self.epochs[individual].average())

    def evokeds_to_grand_evoked(self):
        self.grand_evoked = mne.grand_average(self.evokeds)


class Individual(BaseAnalysis):
    def __init__(self, analysis_db, data_db, filename):
        super().__init__(analysis_db, data_db)
        self.pickle_path = Path(data_db, filename)

    def load_epochs(self):
        data = EEG_Participant.load(self.pickle_path)
        self.epochs = data.epochs


wd = Path(
    '/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK')
data_db = Path(wd, 'MNE_preprocessing_db')
analysis_db = Path(wd, 'MNE_analysis_db')

individual_list = [Path(data_db, pickle).with_suffix('.pickle') for pickle in [
    # '20220131_1255ppt1',
    # '20220318_1418PPT1NEW',
    # '20220202_0830PPT2',
    # '20220203_1225PPT3',
    # '20220204_0855PPT4',
    # '20220209_1456PPT6',
    # '20220211_0846PPT7',
    # '20220218_0920PPT8',
    # '20220218_1242PPT9',
    # '20220225_1019PPT11',
    # '20220302_0952PPT12',
    # '20220310_0848PPT13',
    # '20220311_1132PPT15',
    # '20220311_1445PPT16',
    # '20220314_0847PPT17',
    # '20220316_1141PPT19',
    # '20220318_1142ppt22',
    '20221110_0937PPT24_scenes',
    '20221103_1247_PPT23_scenes',
    '20221111_1152PPT25_scenes',
    '20221122_1258_PPT26_scenes',
]]

analysis = Group(analysis_db=analysis_db, data_db=data_db, individual_list=individual_list)
# analysis.save(Path(analysis_db, 'group_analysis.pickle'))

# analysis = Individual(analysis_db=analysis_db, data_db=data_db, filename='20221110_0937PPT24_scenes.pickle')
analysis.load_epochs()
quit()
# analysis = Group.load(Path(analysis_db, 'group_analysis.pickle'))
for block in ['S-S', 'NS-NS', 'S-NS', 'NS-S']:
    for T in ['T1', 'T2']:
        if T == 'T1':
            condition = f'scene/{block}/{T}/correct'
            analysis.select_events(condition)
            analysis.epochs_to_evoked()
            analysis.plot_heatmaps(title=condition.replace('/', '_'))
        else:
            for lag in ['lag1', 'lag3']:
                for blink in ['correct', 'attentional_blink']:
                    condition = f'scene/{block}/{T}/{lag}/{blink}'
                    analysis.select_events(condition)
                    analysis.epochs_to_evoked()
                    analysis.plot_heatmaps(title=condition.replace('/', '_'))

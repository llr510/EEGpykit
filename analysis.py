from pathlib import Path
import mne
import numpy as np
from preprocesser import EEG_Participant
from epoch_evoked_plotter import plot_compare_evokeds
import matplotlib
import itertools
from scipy import stats
import pandas as pd
from tqdm import tqdm

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
    """Contains all the functions common between individual and group analyses"""

    def __init__(self, analysis_db, data_db):
        self.analysis_db = analysis_db
        self.data_db = data_db
        self.selected_events = None
        self.evoked = None
        self.epochs = None
        self.channels = None

    def select_events(self, events):
        self.selected_events = events

    def select_channels(self, channels: list):
        self.channels = channels

    def epochs_to_evoked(self):
        self.evoked = self.epochs[self.selected_events].average()

    def plot_channels(self):
        raise NotImplementedError

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
        self.evokeds_cropped = None
        self.epoch_objects = None
        self.individual_list = individual_list

        self.evokeds = None
        self.grand_evoked = None
        self.means = None
        self.peaks = None

    def load_epochs(self):
        """loads all epoch object pickles in individual list"""
        self.epoch_objects = {}
        for individual in tqdm(self.individual_list):
            data = EEG_Participant.load(individual).epochs
            self.epoch_objects[individual.name] = data
        # self.channels = individual.ch_names
        del data

    def individuals_to_evokeds(self, exclude_list):
        """converts epoch objects to separate evoked objects."""
        self.evokeds = []
        for individual, epochs in self.epoch_objects.items():
            if individual not in exclude_list:
                m = epochs[self.selected_events].average()
                self.evokeds.append(m)

    def check_for_missing_events(self, conditions):
        """converts epoch objects to separate evoked objects."""
        exclude_list = []
        for condition in conditions:
            for individual, epochs in self.epoch_objects.items():
                try:
                    if len(epochs[condition]) < 1:
                        print('no epochs found for selected event types')
                        exclude_list.append(individual)
                except KeyError as e:
                    print(e)
                    exclude_list.append(individual)
        return exclude_list

    def window_crop(self, tmin, tmax):
        self.evokeds_cropped = []
        for evoked in self.evokeds:
            self.evokeds_cropped.append(evoked.copy().crop(tmin=tmin, tmax=tmax).pick_channels(self.channels))

    def window_average(self, tmin, tmax, plotting=False):
        """gets mean value for all evokeds in specific time window"""
        self.means = []
        for evoked in self.evokeds:
            e = evoked.copy().crop(tmin=tmin, tmax=tmax).pick_channels(self.channels).data.mean()
            self.means.append(e)
            if plotting:
                e.plot()

    def window_peak(self, tmin, tmax, mode="abs"):
        """gets peak value for all evokeds in specific time window"""
        self.peaks = []
        for evoked in self.evokeds:
            self.peaks.append(
                evoked.copy()
                .crop(tmin=tmin, tmax=tmax)
                .pick_channels(self.channels)
                .get_peak(tmin=tmin, tmax=tmax, mode=mode, return_amplitude=True)[2])

    def evokeds_to_grand_evoked(self, tmin, tmax):
        if tmin is not None or tmax is not None:
            cropped = [evoked.copy().crop(tmin=tmin, tmax=tmax).pick_channels(self.channels) for evoked in self.evokeds]
            self.grand_evoked = mne.grand_average(cropped)
        else:
            self.grand_evoked = mne.grand_average(self.evokeds)

    def plot_conditions(self):
        for block in ['S-S', 'NS-NS', 'S-NS', 'NS-S']:
            for T in ['T1', 'T2']:
                if T == 'T1':
                    condition = f'scene/{block}/{T}/correct'
                    self.select_events(condition)
                    self.epochs_to_evoked()
                    self.plot_heatmaps(title=condition.replace('/', '_'))
                else:
                    for lag in ['lag1', 'lag3']:
                        for blink in ['correct', 'attentional_blink']:
                            condition = f'scene/{block}/{T}/{lag}/{blink}'
                            self.select_events(condition)
                            self.epochs_to_evoked()
                            self.plot_heatmaps(title=condition.replace('/', '_'), show=False)


class Individual(BaseAnalysis):
    def __init__(self, analysis_db, data_db, filename):
        super().__init__(analysis_db, data_db)
        self.pickle_path = Path(data_db, filename)

    def load_epochs(self):
        data = EEG_Participant.load(self.pickle_path)
        self.epochs = data.epochs
        self.channels = self.epochs.ch_names
        del data


class ERP_component:
    """Event Related Potential object. Used for sliced evoked data and filtering channels for analysis."""

    def __init__(self, name: str, start: int, end: int, channels: list):
        self.name = name
        self.start = start / 1000
        self.end = end / 1000
        self.location = channels


class Statistics:
    """Uses GroupAnalysis object to compute statistics"""

    def __init__(self, data: Group, main_comp: list, event_cond: dict, components: list):
        self.group_data = data
        self.main_comp = main_comp
        self.con_dict = event_cond
        self.combinations = list(itertools.product(*self.con_dict.values()))
        self.components = components
        self.dict_list = []

    def set_conditions(self, main_comp: list, event_cond: dict, components: list):
        self.main_comp = main_comp
        self.con_dict = event_cond
        self.combinations = list(itertools.product(*self.con_dict.values()))
        self.components = components

    def get_window_values(self, condition, component: ERP_component, exclude_list, type='mean', plotting=False):
        """filters by ERP and events before getting values across time window defined by ERP component."""
        self.group_data.select_channels(component.location)
        self.group_data.select_events(condition)
        self.group_data.individuals_to_evokeds(exclude_list)

        if type == 'mean':
            self.group_data.window_average(tmin=component.start, tmax=component.end, plotting=plotting)
            return self.group_data.means
        elif type == 'crop':
            self.group_data.window_crop(tmin=0, tmax=None)
            return self.group_data.evokeds_cropped
        elif type == 'peak':
            self.group_data.window_peak(tmin=component.start, tmax=component.end)
            return self.group_data.peaks
        elif type == 'grand':
            self.group_data.evokeds_to_grand_evoked(tmin=0, tmax=None)
            return self.group_data.grand_evoked

    def compute(self, func=stats.ttest_rel, type='mean', plotting=False):
        """computes statistics for all permutations of events and ERP components of interest"""
        for main_comparison in main_comparisons:
            for component in self.components:
                for combination in self.combinations:
                    combination = '/'.join(combination)

                    row = {}
                    means = []
                    crops = {}
                    row['component'] = component.name
                    row['combination'] = combination

                    main_conditions = ['/'.join([main_condition, combination]) for main_condition in main_comparison]
                    exclude_list = self.group_data.check_for_missing_events(conditions=main_conditions)
                    # print(list(set(exclude_list)))
                    for n, main_condition in enumerate(main_comparison):
                        events = '/'.join([main_condition, combination])
                        mean_windows = self.get_window_values(events, component, exclude_list, type, plotting=False)
                        means.append(mean_windows)

                        row[f'condition{n + 1}'] = main_condition
                        row[f'mean{n + 1}'] = np.mean(mean_windows)

                        if plotting:
                            crops[main_condition] = self.get_window_values(events, component, exclude_list, type='crop')

                    stats_output = func(means[0], means[1])
                    row['n'] = len(self.group_data.evokeds)
                    row = {**row, **stats_output}
                    self.dict_list.append(row)

                    if plotting:
                        # Uses modified function from old version of MNE that allows for confidence interval
                        # plotting on group data
                        title = f"{main_comparison[0]}vs{main_comparison[1]}_{combination.replace('/', '_')}_{component.name}"
                        fig = plot_compare_evokeds(crops, title=title, show=False, show_sensors=True, ci=0.95)[0]

                        # Plot coloured bar behind lines representing ERP component
                        if row['significance']:
                            facecolor = 'g'
                        else:
                            facecolor = 'r'
                        fig.axes[0].axvspan(component.start, component.end, facecolor=facecolor, alpha=0.3,
                                            label=component.name)
                        fig.savefig(fname=Path('figures', 'windows', title))
                        plt.close()

    def save_output(self, filename):
        df = pd.DataFrame(self.dict_list)
        float_cols = df.select_dtypes("number").columns
        for col in float_cols:
            df[col] = df[col].apply(lambda x: format(x, '#.3g'))
        df.to_csv(f"{filename}_{len(self.group_data.epoch_objects)}.csv", index=False)

    @staticmethod
    def bootstrap_paired_t_test(conditionA, conditionB, btval=10000):
        """
        Bootstrapping: Random sampling with replacement
        Written by Emma Raat. Modified by Lyndon Rakusen
        """
        data_array = conditionA + conditionB
        data_array = np.asarray(data_array, dtype=object)
        # Todo The output were backwards in the original code - let Emma know
        actualT, actualP = stats.ttest_rel(conditionA, conditionB)
        actualT = abs(actualT)
        mean_diff = np.mean(conditionA) - np.mean(conditionB)
        # positive = A is higher, negative = B is higher
        bt_Ts = []
        for bt in range(0, btval):
            bt_A = np.random.choice(data_array, len(conditionA))
            bt_B = np.random.choice(data_array, len(conditionB))
            bt_T, bt_P = stats.ttest_rel(bt_A, bt_B)
            bt_Ts.append(abs(bt_T))

        bt_Ts.sort()
        take_this_value = int(np.round((btval - 1) * 0.95))
        bt_95th_T = bt_Ts[take_this_value]

        if actualT > bt_95th_T:
            significance = True
        else:
            significance = False
        return {'actualP': actualP, 'actualT': actualT, 'bt_95th_T': bt_95th_T, 'mean_diff': mean_diff,
                'significance': significance}

    @staticmethod
    def paired_t_test(a, b):
        t_val, p_val = stats.ttest_rel(a, b)
        if p_val < 0.05:
            significance = True
        else:
            significance = False
        return {'t': t_val, 'p': t_val, 'sig': significance}


if '__main__' in __name__:
    wd = Path(
        '/Volumes/psgroups-1/AttentionPerceptionLab/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK')
    if not wd.exists():
        print(f'{wd} does not exist.')
        quit()
    data_db = Path(wd, 'MNE_preprocessing_db')
    analysis_db = Path(wd, 'MNE_analysis_db')

    individual_list = [Path(data_db, pickle).with_suffix('.pickle') for pickle in [
        '20220131_1255ppt1',
        '20220318_1418PPT1NEW',
        '20220202_0830PPT2',
        '20220203_1225PPT3',
        '20220204_0855PPT4',
        '20220209_1456PPT6',
        '20220211_0846PPT7',
        '20220218_0920PPT8',
        '20220218_1242PPT9',
        '20220225_1019PPT11',
        '20220302_0952PPT12',
        '20220310_0848PPT13',
        '20220311_1132PPT15',
        '20220311_1445PPT16',
        '20220314_0847PPT17',
        '20220316_1141PPT19',
        '20220318_1142ppt22',
        '20221110_0937PPT24_scenes',
        '20221103_1247_PPT23_scenes',
        '20221111_1152PPT25_scenes',
        '20221122_1258_PPT26_scenes',
        '20221124_0945PPT30_scenes'
    ]]

    # P3a = ERP_component(name='P3a', start=250, end=280, channels=['Fp1'])
    P3a = ERP_component(name='P3a', start=280, end=380, channels=['Fp1'])
    P3b_long = ERP_component(name='P3b', start=250, end=500, channels=['Pz', 'CPz', 'POz'])
    P3b = ERP_component(name='P3b', start=250, end=350, channels=['Pz', 'CPz', 'POz'])

    data = Group(analysis_db=analysis_db, data_db=data_db, individual_list=individual_list)
    data.load_epochs()

    # Comparisons 1 & 2
    event_conditions = {
        'stimuli': ['scene'],
        'accuracy': ['correct'],
        'time': ['T1', 'T2'],
        'lag': ['lag1', 'lag3']
    }
    main_comparisons = [['S-S', 'NS-NS']]
    components = [P3a, P3b_long]
    plotting = True
    analysis = Statistics(data=data, main_comp=main_comparisons, event_cond=event_conditions, components=components)
    analysis.compute(analysis.bootstrap_paired_t_test, type='mean', plotting=plotting)

    # Comparisons 3
    event_conditions = {
        'stimuli': ['scene'],
        'accuracy': ['correct'],
        'time': ['T2'],
        'lag': ['lag1', 'lag3']
    }
    main_comparisons = [['S-S', 'NS-S'], ['NS-S', 'S-NS'], ['NS-NS', 'S-NS']]
    components = [P3a, P3b_long]
    analysis.set_conditions(main_comp=main_comparisons, event_cond=event_conditions, components=components)
    analysis.compute(analysis.bootstrap_paired_t_test, type="mean", plotting=plotting)

    # Comparisons 4 & 5
    event_conditions = {
        'stimuli': ['scene'],
        'condition': ['S-S', 'NS-NS'],
        'accuracy': ['correct'],
        'time': ['T1', 'T2']
    }
    main_comparisons = [['lag1', 'lag3']]
    components = [P3a, P3b_long]
    analysis.set_conditions(main_comp=main_comparisons, event_cond=event_conditions, components=components)
    analysis.compute(analysis.bootstrap_paired_t_test, type="mean", plotting=plotting)

    # Comparisons 6
    event_conditions = {
        'stimuli': ['scene'],
        'condition': ['S-S', 'NS-NS'],
        'accuracy': ['correct'],
        'lag': ['lag1', 'lag3']
    }
    main_comparisons = [['T1', 'T2']]
    components = [P3a, P3b_long]
    analysis.set_conditions(main_comp=main_comparisons, event_cond=event_conditions, components=components)
    analysis.compute(analysis.bootstrap_paired_t_test, type="mean", plotting=plotting)

    # Comparisons 7
    event_conditions = {
        'stimuli': ['scene'],
        'condition': ['NS-S', 'S-NS'],
        'time': ['T2'],
        'lag': ['lag1', 'lag3']
    }
    main_comparisons = [['correct', 'attentional_blink']]
    components = [P3a, P3b]
    analysis.set_conditions(main_comp=main_comparisons, event_cond=event_conditions, components=components)
    analysis.compute(analysis.bootstrap_paired_t_test, type="mean", plotting=plotting)

    analysis.save_output('bootstraps_mean')

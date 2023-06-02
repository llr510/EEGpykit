from pathlib import Path
import mne
import numpy as np
from preprocesser import EEG_Participant
from utils.epoch_evoked_plotter import plot_compare_evokeds
import matplotlib
import itertools
from scipy import stats
import pandas as pd
from tqdm import tqdm
import yaml
import sys

sys.path.append('utils')

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
        self.excluded_events = None
        self.evoked = None
        self.epochs = None
        self.channels = None

    def select_events(self, events):
        self.selected_events = events

    def exclude_events(self, events):
        self.excluded_events = events

    def select_channels(self, channels: list):
        self.channels = channels

    def epochs_to_evoked(self):
        """Applies selection and exclusion criteria, then creates evoked."""

        # if self.excluded_events is not None:
        #     excluded_epochs = []
        #     for event in self.excluded_events:
        #         excluded_epochs += self.epochs[event].selection.tolist()
        #
        #     print(excluded_epochs)
        #     print(len(self.epochs))
        #     self.evoked = self.epochs.drop(excluded_epochs)
        #     self.evoked = self.evoked[self.selected_events].average()

        if self.selected_events is None:
            self.evoked = self.epochs.average()
        else:
            self.evoked = self.epochs[self.selected_events].average()

    def plot_channels(self):
        raise NotImplementedError

    def plot_heatmaps(self, title, anim=False, show=True, save=True, dpi=300):
        if self.evoked is None:
            print('No evoked data to plot')
            return
        times = [x / 10 for x in range(0, int(self.evoked.tmax * 10) + 1)]
        joint = self.evoked.plot(show=show, spatial_colors=True)
        # joint = self.evoked.plot_joint(title=title, show=show)
        topo = self.evoked.plot_topomap(times=times, ch_type='eeg', title=title, show=show)

        if save:
            output = Path(self.analysis_db, 'figs', 'joint', f'joint_{title}.jpg')
            output.parent.mkdir(parents=True, exist_ok=True)
            joint.savefig(output, dpi=dpi)
            output = Path(self.analysis_db, 'figs', 'topo', f'topo_{title}.jpg')
            output.parent.mkdir(parents=True, exist_ok=True)
            topo.savefig(output, dpi=dpi)

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
        """converts epoch objects to separate evoked objects.
        @param exclude_list: list of individuals to exclude from evokeds
        """
        self.evokeds = []
        for individual, epochs in self.epoch_objects.items():
            if individual not in exclude_list:
                if self.selected_events is None:
                    m = epochs.average()
                else:
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
        """crops evokeds to time window without averaging"""
        self.evokeds_cropped = []
        for evoked in self.evokeds:
            self.evokeds_cropped.append(evoked.copy().crop(tmin=tmin, tmax=tmax).pick_channels(self.channels))

    def window_average(self, tmin, tmax, plotting=False):
        """gets single mean value for all evokeds in specific time window"""
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

    # Unused
    # def plot_conditions(self):
    #     for block in ['S-S', 'NS-NS', 'S-NS', 'NS-S']:
    #         for T in ['T1', 'T2']:
    #             if T == 'T1':
    #                 condition = f'scene/{block}/{T}/correct'
    #                 self.select_events(condition)
    #                 self.epochs_to_evoked()
    #                 self.plot_heatmaps(title=condition.replace('/', '_'))
    #             else:
    #                 for lag in ['lag1', 'lag3']:
    #                     for blink in ['correct', 'attentional_blink']:
    #                         condition = f'scene/{block}/{T}/{lag}/{blink}'
    #                         self.select_events(condition)
    #                         self.epochs_to_evoked()
    #                         self.plot_heatmaps(title=condition.replace('/', '_'), show=False)


class Individual(BaseAnalysis):
    def __init__(self, analysis_db, data_db, filename):
        super().__init__(analysis_db, data_db)
        self.pickle_path = Path(data_db, filename)
        self.filename = Path(filename)
        self.pid = Path(filename).stem

    def load_epochs(self):
        data = EEG_Participant.load(self.pickle_path)
        self.epochs = data.epochs
        self.channels = self.epochs.ch_names
        del data

    def window_crop(self, tmin, tmax):
        self.epochs.copy().crop(tmin=tmin, tmax=tmax).pick_channels(self.channels)

    def window_average(self, tmin, tmax, plotting=False):
        """gets mean value for all epochs in specific time window"""
        self.epochs.copy().crop(tmin=tmin, tmax=tmax).pick_channels(self.channels).data.mean()

    def plot_evokeds(self, main_comparison=[], component=None, show=True):
        self.select_channels(component.location)

        main_conditions = {main_condition: '/'.join([main_condition, self.selected_events]) for main_condition in
                           main_comparison}
        try:
            main_comparison = {condition: self.epochs[selection] for condition, selection in main_conditions.items()}
            # plot with 95% confidence intervals
            if self.channels is None:
                picks = 'eeg'
            else:
                picks = self.channels
            title = f"{self.pid}_{list(main_comparison.keys())[0]}vs{list(main_comparison.keys())[1]}_{self.selected_events.replace('/', '_')}_{component.name}"
            fig = plot_compare_evokeds(main_comparison, picks=picks, combine='mean', ci=0.95, title=title,
                                       show_sensors=True, show=False)[0]
        except KeyError:
            return {}
        fig.axes[0].axvspan(component.start, component.end, facecolor='black', alpha=0.3,
                            label=component.name)
        if show:
            fig.show()
            plt.pause(10)
        fig.savefig(Path('figures', 'individual', title).with_suffix('.png'))
        # Todo why does this turn into a list?
        return {key: val[0].average() for key, val in main_comparison.items()}


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
        for main_comparison in self.main_comp:
            for component in self.components:
                for combination in self.combinations:
                    combination = '/'.join(combination)

                    row = {}
                    means = []
                    crops = {}
                    row['component'] = component.name
                    row['combination'] = combination

                    main_conditions = ['/'.join([main_condition, combination]) for main_condition in main_comparison]
                    main_conditions = [self.expand_events(main_condition) for main_condition in main_conditions]

                    exclude_list = self.group_data.check_for_missing_events(conditions=main_conditions)

                    for n, main_condition in enumerate(main_comparison):
                        events = '/'.join([main_condition, combination])
                        events = self.expand_events(events)
                        mean_windows = self.get_window_values(events, component, exclude_list, type, plotting=False)
                        means.append(mean_windows)

                        row[f'condition{n + 1}'] = main_condition
                        row[f'mean{n + 1} (µV)'] = np.mean(mean_windows)

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

                        fig = plot_compare_evokeds(crops, title=title, show=False, show_sensors=True, ci=0.95,
                                                   combine='mean')[0]

                        # Plot coloured bar behind lines representing ERP component
                        if row['significance']:
                            facecolor = 'g'
                        else:
                            facecolor = 'r'
                        fig.axes[0].axvspan(component.start, component.end, facecolor=facecolor, alpha=0.3,
                                            label=component.name)

                        fig_path = Path(self.group_data.analysis_db, 'figures', 'windows', title)
                        fig_path.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(fname=fig_path)
                        plt.close()

    def save_output(self, filename):
        df = pd.DataFrame(self.dict_list)
        volt_cols = ['mean1 (µV)', 'mean2 (µV)', 'mean_diff (µV)']
        # convert from volts to microvolts and then round to 3 decimal places
        for col in volt_cols:
            df[col] = df[col].apply(lambda x: format((x * 1000000), '#.3g'))

        pth = Path(self.group_data.analysis_db, f"{filename}_n{len(self.group_data.epoch_objects)}.csv")
        if pth.exists():
            dfo = pd.read_csv(pth)
            df = dfo.append(df)
        df.to_csv(pth, index=False)

    @staticmethod
    def expand_events(events):
        """For when multiple events are selected with &, expand them into a nested list"""
        events = [e.split('&') for e in events.split('/')]
        events = list(itertools.product(*events))
        return ['/'.join(x) for x in events]

    @staticmethod
    def bootstrap_paired_t_test(conditionA, conditionB, btval=10000):
        """
        Bootstrapping: Random sampling with replacement
        Written by Emma Raat. Modified by Lyndon Rakusen
        """
        data_array = conditionA + conditionB
        data_array = np.asarray(data_array, dtype=object)
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
        return {'actualP': round(actualP, 2),
                'actualT': round(actualT, 2),
                'bt_95th_T': round(bt_95th_T, 2),
                'mean_diff (µV)': mean_diff,
                'significance': round(significance, 2)}

    @staticmethod
    def paired_t_test(a, b):
        t_val, p_val = stats.ttest_rel(a, b)
        if p_val < 0.05:
            significance = True
        else:
            significance = False
        return {'t': round(t_val, 2), 'p': round(t_val, 2), 'sig': round(significance, 2)}


def plot_individuals(filename, individuals, con_dict, components, main_comparisons):
    combinations = list(itertools.product(*con_dict.values()))
    dict_list = []
    for individual in individuals:
        i = Individual(analysis_db=analysis_db, data_db=data_db, filename=individual)
        i.load_epochs()
        for main_comparison in main_comparisons:
            for component in components:
                for combination in combinations:
                    combination = '/'.join(combination)

                    row = {'pid': i.pid, 'component': component.name, 'combination': combination}

                    i.select_events(combination)
                    evokeds = i.plot_evokeds(main_comparison=main_comparison, component=component, show=False)
                    for n, (con, evoked), in enumerate(evokeds.items()):
                        row[f'condition{n + 1}'] = con
                        row[f'mean{n + 1}'] = evoked.copy().crop(tmin=component.start,
                                                                 tmax=component.end).pick_channels(
                            component.location).data.mean()
                        row[f'n_epochs{n + 1}'] = evoked.nave
                    dict_list.append(row)

    df = pd.DataFrame(dict_list)
    float_cols = df.select_dtypes("number").columns
    for col in float_cols:
        df[col] = df[col].apply(lambda x: format(x, '#.3g'))
    df.to_csv(f"{filename}_{len(individuals)}.csv", index=False)


if '__main__' in __name__:
    wd = Path(
        '/Volumes/psgroups/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK')

    if not wd.exists():
        print(f'{wd} does not exist.')
        quit()

    experiment = 'scene'
    analysis_db = Path(wd, 'MNE ANALYSIS LAG1 & LAG3')
    config_fname = Path(analysis_db, f'{experiment}s_config.yaml')

    with open(config_fname, 'r') as f:
        analysis_config = yaml.safe_load(f)

    components = []
    for comp in analysis_config['components'].values():
        components.append(
            ERP_component(name=comp['name'], start=comp['start'], end=comp['end'], channels=comp['channels'])
        )

    if experiment == 'scene':
        data_db = Path(wd, 'MNE_preprocessing_db')

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
    elif experiment == 'dot':
        data_db = Path(wd, 'MNE_preprocessing_db')
        # analysis_db = Path(wd, 'MNE ANALYSIS LAGS 2-4')

        individual_list = [Path(data_db, pickle).with_suffix('.pickle') for pickle in [
            '20220202_0830PPT2',
            '20220203_1225PPT3',
            '20220204_0855PPT4',
            '20220211_0846PPT7',
            '20220302_0952PPT12',
            '20220311_0900PPT14',
            '20230111_0903PPT1_dots',
            '20230126_0917PPT5_dots',
            '20230210_0840PPT23_dots',
            '20230215_0843PPT24_dots',
            '20230217_0846PPT25_dots',
            '20230224_1654PPT26_dots',
            '20230222_1647PPT27_dots',
            '20230307_1010PPT28_dots',
            '20230310_1258PPT29_dots',
            '20230313_0844PPT30_dots',
            '20230316_0904PPT31_dots',
            '20230317_1307PPT32_dots',
            # '20230321_1359PPT33_dots',
            '20230329_1401PPT34_dots',
            '20230330_0902PPT35_dots',
            '20230331_0852PPT36_dots',
            '20230331_1305PPT33-2_dots',
        ]]

    data = Group(analysis_db=analysis_db, data_db=data_db, individual_list=individual_list)
    data.load_epochs()
    plotting = True

    for name, settings in analysis_config['analyses'].items():
        print('Comparison(s) ', name)

        analysis = Statistics(data=data,
                              main_comp=settings['main_comparisons'], event_cond=settings['event_conditions'],
                              components=components)
        analysis.compute(analysis.bootstrap_paired_t_test, type='mean', plotting=plotting)
        analysis.save_output(f'comparisons_{name}')

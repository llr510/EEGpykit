import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

from MVPAnalysis import MVPAnalysis, get_filepaths_from_file, delta_evoked_MVPA

matplotlib.use('Agg')
plt.rcParams['animation.ffmpeg_path'] = '/users/llr510/.local/share/ffmpeg-downloader/ffmpeg'
np.random.seed(1025)


def run_training_analysis(input_file, output_dir, jobs=-1, indiv_plot=False):
    """
    Individual, group, and session level analysis for mammography training experiment
    """

    files, extra = get_filepaths_from_file(input_file)

    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file, verbose=False)
        epochs_list.append(epochs)

    # 1.	No main effect of Session in behavioral
    # a.	What about in the brain?
    # i.	Contrast brain responses in session 1 and session 2
    MVPAnalysis(files, var1_events=['sesh_1'], var2_events=['sesh_2'], excluded_events=['Rate', 'Missed'],
                scoring="roc_auc", output_dir=Path(output_dir, 'Naives/between_session/all/'), indiv_plot=indiv_plot,
                epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)

    for resp in ['', '/Correct']:
        for sesh in ['/sesh_1', '/sesh_2']:
            # 1.	No main effect of Session in behavioral
            # a.	What about in the brain?
            # ii.	Contrast all normal vs all abnormal in session 1
            # iii.	Contrast all normal vs all abnormal in session 2
            MVPAnalysis(files, var1_events=[f'normal{sesh}{resp}'],
                        var2_events=[f'malignant{sesh}{resp}', f'global{sesh}{resp}'],
                        excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                        output_dir=Path(output_dir, f'Naives{sesh}/normal_vs_abnormal/{resp}'), indiv_plot=indiv_plot,
                        epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)

            # MVPA_analysis(files,
            #               var1_events=[f'normal{sesh}'],
            #               var2_events=[f'malignant{sesh}'],
            #               excluded_events=['Rate', 'Missed'],
            #               scoring="roc_auc",
            #               output_dir=Path(output_dir, f'Naives{sesh}/normal_vs_malignant/'),
            #               indiv_plot=indiv_plot,
            #               jobs=jobs,
            #               epochs_list=epochs_list,
            #               extra_event_labels=extra
            #               )
            #
            # MVPA_analysis(files,
            #               var1_events=[f'normal{sesh}'],
            #               var2_events=[f'global{sesh}'],
            #               excluded_events=['Rate', 'Missed'],
            #               scoring="roc_auc",
            #               output_dir=Path(output_dir, f'Naives{sesh}/normal_vs_global/'),
            #               indiv_plot=indiv_plot,
            #               jobs=jobs,
            #               epochs_list=epochs_list,
            #               extra_event_labels=extra
            #               )
        # 2.	There is a main effect of image target present type in behavioral
        # a.	What about the brain?
        # i.	Across both session Obvious vs Priors (in addition just for hits)
        MVPAnalysis(files, var1_events=[f'obvious{resp}'], var2_events=[f'prior{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Naives/across_session/obvious_vs_priors{resp}'),
                    indiv_plot=indiv_plot, epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)
        # ii.	Across both session Contra vs Priors (in addition just for hits)
        MVPAnalysis(files, var1_events=[f'contra{resp}'], var2_events=[f'prior{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Naives/across_session/contra_vs_priors{resp}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)
        # iii.	Across both session Subtle vs Priors (in addition just for hits)
        MVPAnalysis(files, var1_events=[f'subtle{resp}'], var2_events=[f'prior{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Naives/across_session/subtle_vs_priors{resp}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)

        # 3.	If you want to look at the interaction and image type then you would run these contrasts:
        # i.	Session 1 vs Session 2 just for obvious images (in addition just for hits)
        MVPAnalysis(files, var1_events=[f'sesh_1/obvious{resp}'], var2_events=[f'sesh_2/obvious{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Naives/between_session/Obvious{resp}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)
        # ii.	Session 1 vs Session 2 just for priors images (in addition just for hits)
        MVPAnalysis(files, var1_events=[f'sesh_1/prior{resp}'], var2_events=[f'sesh_2/prior{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Naives/between_session/Priors{resp}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)

        # iii.	Session 1 vs Session 2 just for contra images (in addition just for hits)
        MVPAnalysis(files, var1_events=[f'sesh_1/contra{resp}'], var2_events=[f'sesh_2/contra{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Naives/between_session/Contra{resp}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)

        # iv.	Session 1 vs Session 2 just for subtle images (in addition just for hits)
        MVPAnalysis(files, var1_events=[f'sesh_1/subtle{resp}'], var2_events=[f'sesh_2/subtle{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Naives/between_session/Subtle{resp}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)


def run_rads_analysis(input_file, output_dir, jobs=-1, indiv_plot=False):
    files, extra = get_filepaths_from_file(input_file)

    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file, verbose=False)
        epochs_list.append(epochs)

    for resp in ['', '/Correct']:
        # 1.	No main effect of Session in behavioral
        # a.	What about in the brain?
        # ii.	Contrast all normal vs all abnormal
        MVPAnalysis(files, var1_events=[f'normal{resp}'], var2_events=[f'malignant{resp}', f'global{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Rads/normal_vs_abnormal{resp}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)

        # MVPA_analysis(files,
        #               var1_events=[f'normal'],
        #               var2_events=[f'malignant'],
        #               excluded_events=['Rate', 'Missed'],
        #               scoring="roc_auc",
        #               output_dir=Path(output_dir, f'Rads/normal_vs_malignant{resp}'),
        #               indiv_plot=indiv_plot,
        #               jobs=jobs,
        #               epochs_list=epochs_list,
        #               extra_event_labels=extra
        #               )
        #
        # MVPA_analysis(files,
        #               var1_events=[f'normal'],
        #               var2_events=[f'global'],
        #               excluded_events=['Rate', 'Missed'],
        #               scoring="roc_auc",
        #               output_dir=Path(output_dir, f'Rads/normal_vs_global{resp}'),
        #               indiv_plot=indiv_plot,
        #               jobs=jobs,
        #               epochs_list=epochs_list,
        #               extra_event_labels=extra
        #               )
        # 2.	There is a main effect of image target present type in behavioral
        # a.	What about the brain?
        # i.	Across both session Obvious vs Priors (in addition just for hits)
        MVPAnalysis(files, var1_events=[f'obvious{resp}'], var2_events=[f'prior{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Rads/obvious_vs_priors{resp}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)
        # ii.	Across both session Contra vs Priors (in addition just for hits)
        MVPAnalysis(files, var1_events=[f'contra{resp}'], var2_events=[f'prior{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Rads/contra_vs_priors{resp}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)
        # iii.	Across both session Subtle vs Priors (in addition just for hits)
        MVPAnalysis(files, var1_events=[f'subtle{resp}'], var2_events=[f'prior{resp}'],
                    excluded_events=['Rate', 'Missed'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'Rads/subtle_vs_priors{resp}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, jobs=jobs)


def run_ab_analysis(input_file, output_dir, jobs=-1, indiv_plot=False, regen_dat=False):
    if 'scene' in input_file:
        stim = 'scene'
    elif 'dot' in input_file:
        stim = 'dot'
    else:
        stim = ''

    files, extra = get_filepaths_from_file(Path(output_dir, f'MVPA_analysis_list_{stim}.csv'))

    epochs_list = []
    # for file in files:
    #     epochs = mne.read_epochs(file)
    #     epochs_list.append(epochs)

    conditions = ['S-S', 'NS-NS', 'NS-S', 'S-NS']
    lags = ['', '/lag1', '/lag2', '/lag3', '/lag4']

    """
    # First analysis
    1.	In Block 1 (S-S) compare T1 vs T2 (collapsed across lags and for each lag separately)
    2.	In Block 4 (NS-NS) compare T1 vs T2 (collapsed across lags and for each lag separately)
    
    # Second analysis
    1.	In Block 3 (NS-S) compare T1 vs T2 (collapsed across lags and for each lag separately)
    2.	In Block 2 (S-NS) compare T1 vs T2 (collapsed across lags and for each lag separately)
    """
    for lag in lags:
        evoked_dict = {}
        for cond in conditions:
            # Perform ordinary MVPA analysis returning dictionary of evoked objects for each condition
            evoked_dict[cond], X_score, y, times = MVPAnalysis(files, var1_events=[f'{stim}/{cond}/T1{lag}'],
                                                               var2_events=[f'{stim}/{cond}/T2{lag}'],
                                                               scoring="roc_auc", output_dir=Path(output_dir,
                                                                                                  f'{stim}/T1vsT2/{cond}{lag}'),
                                                               indiv_plot=indiv_plot, epochs_list=epochs_list,
                                                               extra_event_labels=extra, overwrite_output=regen_dat,
                                                               jobs=jobs)

        """
        First analysis
        5. Comparison Diff T1-T2 in S-S vs Diff T1-T2 in NS-NS (all and for each lag)
        """
        delta_evoked_MVPA(evoked_dict, condition_vars=['S-S', 'NS-NS'],
                          title=f'{stim}_S-SvsNS-NS_T1-T2delta{lag.replace("/", "_")}')
        """
        Second analysis
        5.	Comparison Diff T1-T2 in Block 2 (S-NS) vs Diff T1-T2 in Block 3 (NS-S) (all and for each lag)
        """
        delta_evoked_MVPA(evoked_dict, condition_vars=['S-NS', 'NS-S'],
                          title=f'{stim}_S-NSvsNS-S_T1-T2delta{lag.replace("/", "_")}')

    """
    First analysis
    3. Compare T1 in Block 1 (S-S) & Block 2 (S-NS) vs T1 in Block 4 (NS-NS) & Block 3 (NS-S) 
    (independent of the starting point of T1)
    """
    MVPAnalysis(files, var1_events=[f'{stim}/S-S/T1', f'{stim}/S-NS/T1'],
                var2_events=[f'{stim}/NS-NS/T1', f'{stim}/NS-S/T1'], scoring="roc_auc",
                output_dir=Path(output_dir, f'{stim}/T1/SvsNS'), indiv_plot=indiv_plot, epochs_list=epochs_list,
                extra_event_labels=extra, overwrite_output=regen_dat, jobs=jobs)

    """
    First analysis
    4. Compare T2 in Block 1 (S-S) to T2 in Block 4 (NS-NS) (collapsed across lags and for each lag separately)
    """
    for lag in lags:
        MVPAnalysis(files, var1_events=[f'{stim}/S-S/T2{lag}'], var2_events=[f'{stim}/NS-NS/T2{lag}'],
                    scoring="roc_auc", output_dir=Path(output_dir, f'{stim}/T2/S-SvsNS-NS{lag}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, overwrite_output=regen_dat, jobs=jobs)

    """
    Second analysis
    3.	Compare T2 in Block 2 (S-NS) to T2 in Block 4 (NS-NS) (collapsed across lags and for each lag separately)
    """
    for lag in lags:
        MVPAnalysis(files, var1_events=[f'{stim}/S-NS/T2{lag}'], var2_events=[f'{stim}/NS-NS/T2{lag}'],
                    scoring="roc_auc", output_dir=Path(output_dir, f'{stim}/T2/S-NSvsNS-NS{lag}'),
                    indiv_plot=indiv_plot, epochs_list=epochs_list, extra_event_labels=extra,
                    overwrite_output=regen_dat, jobs=jobs)
    """
    Second analysis
    4.	Compare T2 in Block 3 (NS-S) to T2 in Block 1 (S-S) (collapsed across lags and for each lag separately)
    """
    for lag in lags:
        MVPAnalysis(files, var1_events=[f'{stim}/NS-S/T2{lag}'], var2_events=[f'{stim}/S-S/T2{lag}'], scoring="roc_auc",
                    output_dir=Path(output_dir, f'{stim}/T2/NS-SvsS-S{lag}'), indiv_plot=indiv_plot,
                    epochs_list=epochs_list, extra_event_labels=extra, overwrite_output=regen_dat, jobs=jobs)


def rename_nested_dirs(wd, target='Correct', new='HITS_vs_TNEGS', reverse=False):
    if reverse:
        target, new = new, target
    dirs = Path(wd).rglob(target)
    for d in dirs:
        d.rename(Path(d.parent, new))


if '__main__' in __name__:
    # wd = Path('/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/INDIVIDUAL FOLDERS/LYNDON/MVPA')
    #
    # run_training_analysis(input_file=Path(wd, 'Naives', 'MVPA_analysis_list.csv'),
    #                       output_dir=wd, indiv_plot=False, jobs=-1)
    # run_training_analysis(input_file=Path(wd, 'Rads', 'MVPA_analysis_list_rads.csv'),
    #                       output_dir=wd, indiv_plot=False, jobs=-1)
    #
    # rename_nested_dirs(wd, reverse=False)
    #
    # quit()

    # wd = Path(
    #     '/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK/analysis/MVPA-group_method/MVPA-AB/')
    # # '/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK/analysis/MVPA-group_method/MVPA-AB/dot/T1vsT2/S-S/dot+S-S+T1_vs_dot+S-S+T2.dat'
    # # '/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK/analysis/MVPA-group_method/MVPA-AB/dot/T1vsT2/S-S/dot+S-S+T1_vs_dot+S-S+T2.dat'
    #
    # run_ab_analysis(input_file=Path(wd, 'MVPA_analysis_list_dot.csv').name, output_dir=wd, jobs=1, indiv_plot=False,
    #                 regen_dat=False)
    # run_ab_analysis(input_file=Path(wd, 'MVPA_analysis_list_scene.csv').name, output_dir=wd, jobs=1, indiv_plot=False,
    #                 regen_dat=False)
    #
    # quit()

    print("################# STARTING #################")
    parser = argparse.ArgumentParser(description='Analyse EEG data with MVPA')
    parser.add_argument('--analysis', type=str, required=True, help="Which analysis to do (Naives or Radiologists)")
    parser.add_argument('--input_file', type=str, required=True, help="List of epoched files to analyse")
    parser.add_argument('--output', type=str, required=True, help="Location to save subdirectories and figures to")
    parser.add_argument('--jobs', type=int, required=False, default=-1, help="how many processes to spawn. By default "
                                                                             "uses all available processes.")

    args = parser.parse_args()

    if args.analysis == 'Naives':
        analysis_func = run_training_analysis
    elif args.analysis == 'Rads':
        analysis_func = run_rads_analysis
    elif args.analysis == 'AB':
        analysis_func = run_ab_analysis
    else:
        raise ValueError

    analysis_func(input_file=args.input_file, output_dir=args.output, indiv_plot=False, jobs=args.jobs)

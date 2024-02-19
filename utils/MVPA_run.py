from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne

from MVPA_analysis import MVPA_analysis, get_filepaths_from_file, group_MVPA_and_plot

matplotlib.use('Agg')
plt.rcParams['animation.ffmpeg_path'] = '/users/llr510/.local/share/ffmpeg-downloader/ffmpeg'


def run_AB_analysis(output_dir, jobs=-1, indiv_plot=True):
    """
    AB MVPA analysis:
    In block 1 (S-S condition) compare T1 vs T2 (Lag 1 vs 4; Lag 2 vs 4; Lag 3 vs 4)
    In block 4 (NS-NS condition) compare T1 vs T2 (Lag 1 vs 4; Lag 2 vs 4; Lag 3 vs 4)
    Compare T2 in Block 1 (S-S) to T2 in Block 4 (NS-NS) across lags

    """

    for stim in ['scene', 'dot']:
        files, extra = get_filepaths_from_file(Path(output_dir, f'MVPA_analysis_list_{stim}.csv'))

        epochs_list = []
        for file in files:
            epochs = mne.read_epochs(file)
            epochs_list.append(epochs)

        MVPA_analysis(files,
                      var1_events=[f'{stim}/T2/S-S'],
                      var2_events=[f'{stim}/T2/NS-NS'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'{stim}/T2_S-SvsNS-NS/all_lags'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs_list)

        out = {}
        for condition in ['S-S', 'NS-NS']:
            out[f'{condition}'], _, times = MVPA_analysis(files,
                                                          var1_events=[f'{stim}/T1/{condition}'],
                                                          var2_events=[f'{stim}/T2/{condition}'],
                                                          scoring="roc_auc",
                                                          output_dir=Path(output_dir,
                                                                          f'{stim}/T1vsT2_{condition}/all_lags'),
                                                          indiv_plot=indiv_plot,
                                                          jobs=jobs,
                                                          epochs_list=epochs_list)

            for lag in [1, 2, 3, 4]:
                out[f'{condition}_{lag}'], _, times = MVPA_analysis(files,
                                                                    var1_events=[f'{stim}/T1/{condition}/lag{lag}'],
                                                                    var2_events=[f'{stim}/T2/{condition}/lag{lag}'],
                                                                    scoring="roc_auc",
                                                                    output_dir=Path(output_dir,
                                                                                    f'{stim}/T1vsT2_{condition}/lag{lag}'),
                                                                    indiv_plot=indiv_plot,
                                                                    jobs=jobs,
                                                                    epochs_list=epochs_list)

        group_MVPA_and_plot([out[f'S-S'], out[f'NS-NS']], labels=['S-S', 'NS-NS'],
                            var1_events=[f'S-S'],
                            var2_events=[f'NS-NS'],
                            times=times,
                            output_dir=Path(output_dir, f'{stim}/T1vsT2_S-SvsNS-NS/all_lags'),
                            jobs=jobs)

        for lag in [1, 2, 3, 4]:
            group_MVPA_and_plot([out[f'S-S_{lag}'], out[f'NS-NS_{lag}']], labels=['S-S', 'NS-NS'],
                                var1_events=[f'S-S/{lag}'],
                                var2_events=[f'NS-NS/{lag}'],
                                times=times,
                                output_dir=Path(output_dir, f'{stim}/T1vsT2_S-SvsNS-NS/lag{lag}'),
                                jobs=jobs)

            # T2/SS vs T2/NS-NS for each lag
            MVPA_analysis(files,
                          var1_events=[f'{stim}/T2/S-S/lag{lag}'],
                          var2_events=[f'{stim}/T2/NS-NS/lag{lag}'],
                          scoring="roc_auc",
                          output_dir=Path(output_dir, f'{stim}/T2_S-SvsNS-NS/lag{lag}'),
                          indiv_plot=indiv_plot,
                          jobs=jobs,
                          epochs_list=epochs_list)

        # T1/SS vs T1/NS-NS
        MVPA_analysis(files,
                      var1_events=[f'{stim}/T1/S-S'],
                      var2_events=[f'{stim}/T1/NS-NS'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'{stim}/T1_S-SvsNS-NS/all_lags'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs_list)

        '''T1 to T2 S-S across lags'''
        '''T1 to T2 NS-NS across lags'''
        '''those two vs each other'''

        '''T1 S-S vs T1 NS-NS for each lag'''


def run_training_analysis(output_dir, jobs=-1, indiv_plot=True):
    """
    Individual, group, and session level analysis for mammography training experiment
    """
    X_dict = {}
    """
    Session 1 Analysis
    """
    files, extra = get_filepaths_from_file(Path(output_dir, 'MVPA_analysis_list_sesh1.csv'))

    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file)
        epochs_list.append(epochs)

    X_dict['NormAbnorm_s1'], \
    y, times = MVPA_analysis(files,
                             var1_events=['Normal'],
                             var2_events=['Obvious', 'Subtle', 'Global'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/pre-training/normal_vs_abnormal/'),
                             indiv_plot=indiv_plot,
                             jobs=jobs,
                             epochs_list=epochs_list)

    X_dict['NormMalig_s1'], \
    y, times = MVPA_analysis(files,
                             var1_events=['Normal'],
                             var2_events=['Obvious', 'Subtle'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/pre-training/normal_vs_malignant/'),
                             indiv_plot=indiv_plot,
                             jobs=jobs,
                             epochs_list=epochs_list)

    X_dict['NormGlobal_s1'], \
    y, times = MVPA_analysis(files,
                             var1_events=['Normal'],
                             var2_events=['Global'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/pre-training/normal_vs_global/'),
                             indiv_plot=indiv_plot,
                             jobs=jobs,
                             epochs_list=epochs_list)

    """
    Session 2 Analysis
    """
    files, extra = get_filepaths_from_file(Path(output_dir, 'MVPA_analysis_list_sesh2.csv'))

    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file)
        epochs_list.append(epochs)

    X_dict['NormAbnorm_s2'], \
    y, times = MVPA_analysis(files,
                             var1_events=['Normal'],
                             var2_events=['Obvious', 'Subtle'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/post-training/normal_vs_abnormal/'),
                             indiv_plot=indiv_plot,
                             jobs=jobs,
                             epochs_list=epochs_list)

    X_dict['NormMalig_s2'], \
    y, times = MVPA_analysis(files,
                             var1_events=['Normal'],
                             var2_events=['Obvious', 'Subtle'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/post-training/normal_vs_malignant/'),
                             indiv_plot=indiv_plot,
                             jobs=jobs,
                             epochs_list=epochs_list)

    X_dict['NormGlobal_s2'], \
    y, times = MVPA_analysis(files,
                             var1_events=['Normal'],
                             var2_events=['Global'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/post-training/normal_vs_global/'),
                             indiv_plot=indiv_plot,
                             jobs=jobs,
                             epochs_list=epochs_list)

    """
    Session Both Analysis
    """
    group_MVPA_and_plot([X_dict['NormAbnorm_s1'], X_dict['NormAbnorm_s2']], labels=['session1', 'session2'],
                        var1_events=['Normal'],
                        var2_events=['Obvious', 'Subtle', 'Global'],
                        times=times,
                        output_dir=Path(output_dir, 'naives/across-training/normal_vs_abnormal/'),
                        jobs=jobs)

    group_MVPA_and_plot([X_dict['NormMalig_s1'], X_dict['NormMalig_s2']], labels=['session1', 'session2'],
                        var1_events=['Normal'],
                        var2_events=['Obvious', 'Subtle'],
                        times=times,
                        output_dir=Path(output_dir, 'naives/across-training/normal_vs_malignant/'),
                        jobs=jobs)

    group_MVPA_and_plot([X_dict['NormGlobal_s1'], X_dict['NormGlobal_s2']], labels=['session1', 'session2'],
                        var1_events=['Normal'],
                        var2_events=['Global'],
                        times=times,
                        output_dir=Path(output_dir, 'naives/across-training/normal_vs_global/'),
                        jobs=jobs)


def run_rads_analysis(output_dir, jobs=-1, indiv_plot=True):
    files, extra = get_filepaths_from_file(Path(output_dir, 'MVPA_analysis_list_rads.csv'))

    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file)
        epochs_list.append(epochs)

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Obvious', 'Subtle', 'Global'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir=Path(output_dir,
                                  'radiologists/normal_vs_abnormal/'),
                  indiv_plot=indiv_plot,
                  jobs=jobs,
                  epochs_list=epochs_list)

    MVPA_analysis(files,
                  var1_events=['resp_Normal/Correct'],
                  var2_events=['resp_Abnormal/Correct'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir=Path(output_dir,
                                  'radiologists/normal_vs_abnormal_hits-tnegs/'),
                  indiv_plot=indiv_plot,
                  jobs=jobs,
                  epochs_list=epochs_list)

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Obvious', 'Subtle'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir=Path(output_dir,
                                  'radiologists/normal_vs_malignant/'),
                  indiv_plot=indiv_plot,
                  jobs=jobs,
                  epochs_list=epochs_list)

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Global'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir=Path(output_dir, 'radiologists/normal_vs_global/'),
                  indiv_plot=indiv_plot,
                  jobs=jobs,
                  epochs_list=epochs_list)

    MVPA_analysis(files,
                  var1_events=['Normal/resp_Normal'],
                  var2_events=['Obvious/resp_Abnormal', 'Subtle/resp_Abnormal'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir=Path(output_dir, 'radiologists/normal_vs_malignant_hits-tnegs/'),
                  indiv_plot=indiv_plot,
                  jobs=jobs,
                  epochs_list=epochs_list)

    MVPA_analysis(files,
                  var1_events=['Normal/resp_Normal'],
                  var2_events=['Global/resp_Abnormal'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir=Path(output_dir, 'radiologists/normal_vs_global_hits-tnegs/'),
                  indiv_plot=indiv_plot,
                  jobs=jobs,
                  epochs_list=epochs_list)


def run_training_hits_tnegs(output_dir, jobs=-1, indiv_plot=True):
    """
    Individual, group, and session level analysis for mammography training experiment
    """

    X_dict = {}

    """
    Session 1 Analysis
    """
    files, extra = get_filepaths_from_file(Path(output_dir, 'MVPA_analysis_list_sesh1.csv'))

    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file)
        epochs_list.append(epochs)

    X_dict['NormAbnorm_s1'], \
    y, times = MVPA_analysis(files,
                             var1_events=['resp_Normal/Correct'],
                             var2_events=['resp_Abnormal/Correct'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/pre-training_hits-tnegs/normal_vs_abnormal/'),
                             indiv_plot=False,
                             jobs=jobs,
                             epochs_list=epochs_list)

    X_dict['NormMalig_s1'], \
    y, times = MVPA_analysis(files,
                             var1_events=['Normal/resp_Normal'],
                             var2_events=['Obvious/resp_Abnormal', 'Subtle/resp_Abnormal'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/pre-training_hits-tnegs/normal_vs_malignant/'),
                             indiv_plot=indiv_plot,
                             jobs=jobs,
                             epochs_list=epochs_list)

    X_dict['NormGlobal_s1'], \
    y, times = MVPA_analysis(files,
                             var1_events=['Normal/resp_Normal'],
                             var2_events=['Global/resp_Abnormal'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/pre-training_hits-tnegs/normal_vs_global/'),
                             indiv_plot=indiv_plot,
                             jobs=jobs,
                             epochs_list=epochs_list)

    """
    Session 2 Analysis
    """
    files, extra = get_filepaths_from_file(Path(output_dir, 'MVPA_analysis_list_sesh2.csv'))

    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file)
        epochs_list.append(epochs)

    X_dict['NormAbnorm_s2'], \
    y, times = MVPA_analysis(files,
                             var1_events=['resp_Normal/Correct'],
                             var2_events=['resp_Abnormal/Correct'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/post-training_hits-tnegs/normal_vs_abnormal/'),
                             indiv_plot=False,
                             jobs=jobs,
                             epochs_list=epochs_list)

    X_dict['NormMalig_s2'], \
    y, times = MVPA_analysis(files,
                             var1_events=['Normal/resp_Normal'],
                             var2_events=['Obvious/resp_Abnormal', 'Subtle/resp_Abnormal'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/post-training_hits-tnegs/normal_vs_abnormal/'),
                             indiv_plot=indiv_plot,
                             jobs=jobs,
                             epochs_list=epochs_list)

    X_dict['NormGlobal_s2'], \
    y, times = MVPA_analysis(files,
                             var1_events=['Normal/resp_Normal'],
                             var2_events=['Global/resp_Abnormal'],
                             excluded_events=['Rate', 'Missed'],
                             scoring="roc_auc",
                             output_dir=Path(output_dir,
                                             'naives/post-training_hits-tnegs/normal_vs_global/'),
                             indiv_plot=indiv_plot,
                             jobs=jobs,
                             epochs_list=epochs_list)

    """
    Session Both Analysis
    """
    group_MVPA_and_plot([X_dict['NormAbnorm_s1'], X_dict['NormAbnorm_s2']], labels=['session1', 'session2'],
                        var1_events=['resp_Normal/Correct'],
                        var2_events=['resp_Abnormal/Correct'],
                        times=times,
                        output_dir=Path(output_dir, 'naives/across-training_hits-tnegs/normal_vs_abnormal/'),
                        jobs=jobs)

    group_MVPA_and_plot([X_dict['NormMalig_s1'], X_dict['NormMalig_s2']], labels=['session1', 'session2'],
                        var1_events=['Normal/resp_Normal'],
                        var2_events=['Obvious/resp_Abnormal', 'Subtle/resp_Abnormal'],
                        times=times,
                        output_dir=Path(output_dir, 'naives/across-training_hits-tnegs/normal_vs_malignant/'),
                        jobs=jobs)

    group_MVPA_and_plot([X_dict['NormGlobal_s1'], X_dict['NormGlobal_s2']], labels=['session1', 'session2'],
                        var1_events=['Normal/resp_Normal'],
                        var2_events=['Global/resp_Abnormal'],
                        times=times,
                        output_dir=Path(output_dir, 'naives/across-training_hits-tnegs/normal_vs_global/'),
                        jobs=jobs)


def run_pickle_rads(jobs=-1):
    files, extra = get_filepaths_from_file('')

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Obvious', 'Subtle'],
                  excluded_events=['Rate'],  # , 'Missed'
                  scoring="roc_auc",
                  output_dir='../analyses/rads_data_pkls/normal_v_abnormal/case_balanced',
                  indiv_plot=False,
                  extra_event_labels=extra,
                  pickle_ouput=True,
                  jobs=jobs)

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Global'],
                  excluded_events=['Rate'],  # , 'Missed'
                  scoring="roc_auc",
                  output_dir='../analyses/rads_data_pkls/normal_v_global/case_imbalanced',
                  indiv_plot=False,
                  extra_event_labels=extra,
                  pickle_ouput=True,
                  jobs=jobs)


if '__main__' in __name__:
    # run_AB_analysis(output_dir='../analyses/MVPA-AB/')
    run_training_analysis(output_dir='../analyses/MVPA-viking/')
    run_training_hits_tnegs(output_dir='../analyses/MVPA-viking/')
    # run_rads_analysis(output_dir='../analyses/MVPA-viking/')

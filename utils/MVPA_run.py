from MVPA_analysis import MVPA_analysis, get_filepaths_from_file
import mne


def run_AB_analysis():
    """
    AB MVPA analysis:
    In block 1 (S-S condition) compare T1 vs T2 (Lag 1 vs 4; Lag 2 vs 4; Lag 3 vs 4)
    In block 4 (NS-NS condition) compare T1 vs T2 (Lag 1 vs 4; Lag 2 vs 4; Lag 3 vs 4)
    Compare T2 in Block 1 (S-S) to T2 in Block 4 (NS-NS) across lags

    """

    for stim in ['scene', 'dot']:
        files, extra = get_filepaths_from_file(
            f'../analyses/MVPA-AB/MVPA_analysis_list_{stim}.csv')

        epochs_list = []
        for file in files:
            epochs = mne.read_epochs(file)
            epochs_list.append(epochs)

        MVPA_analysis(files,
                      var1_events=[f'{stim}/T2/S-S'],
                      var2_events=[f'{stim}/T2/NS-NS'],
                      scoring="roc_auc",
                      output_dir=f'../analyses/MVPA-AB/{stim}/T2_S-SvsNS-NS/',
                      indiv_plot=False,
                      concat_participants=False,
                      jobs=-1,
                      epochs_list=epochs_list)

        for condition in ['S-S', 'NS-NS']:
            MVPA_analysis(files,
                          var1_events=[f'{stim}/T1/{condition}'],
                          var2_events=[f'{stim}/T2/{condition}'],
                          scoring="roc_auc",
                          output_dir=f'../analyses/MVPA-AB/{stim}/T1vsT2_{condition}/all_lags',
                          indiv_plot=False,
                          concat_participants=False,
                          jobs=-1,
                          epochs_list=epochs_list)

            for lag in [1, 2, 3, 4]:
                MVPA_analysis(files,
                              var1_events=[f'{stim}/T1/{condition}/lag{lag}'],
                              var2_events=[f'{stim}/T2/{condition}/lag{lag}'],
                              scoring="roc_auc",
                              output_dir=f'../analyses/MVPA-AB/{stim}/T1vsT2_{condition}/lag{lag}',
                              indiv_plot=False,
                              concat_participants=False,
                              jobs=-1,
                              epochs_list=epochs_list)


def run_across_training():
    """Session level analysis"""

    files, extra = get_filepaths_from_file(
        '../analyses/MVPA/MVPA_analysis_list.csv')

    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file)
        epochs_list.append(epochs)

    MVPA_analysis(files,
                  var1_events=['sesh_1/Normal'],
                  var2_events=['sesh_2/Normal'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir='../analyses/MVPA/naives/across-training/normal/',
                  indiv_plot=False,
                  concat_participants=True,
                  extra_event_labels=extra,
                  jobs=1,
                  epochs_list=epochs_list)

    MVPA_analysis(files,
                  var1_events=['sesh_1/Obvious', 'sesh_1/Subtle'],
                  var2_events=['sesh_2/Obvious', 'sesh_2/Subtle'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir='../analyses/MVPA/naives/across-training/abnormal/',
                  indiv_plot=False,
                  concat_participants=True,
                  extra_event_labels=extra,
                  jobs=1,
                  epochs_list=epochs_list)

    MVPA_analysis(files,
                  var1_events=['sesh_1/Global'],
                  var2_events=['sesh_2/Global'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir='../analyses/MVPA/naives/across-training/global/',
                  indiv_plot=False,
                  concat_participants=True,
                  extra_event_labels=extra,
                  jobs=1,
                  epochs_list=epochs_list)


def run_within_training():
    """Individual level analysis"""

    files, extra = get_filepaths_from_file(
        '../analyses/MVPA/MVPA_analysis_list_sesh1.csv')

    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file)
        epochs_list.append(epochs)

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Obvious', 'Subtle'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir='../analyses/MVPA/naives/pre-training/normal_vs_abnormal/',
                  indiv_plot=False,
                  concat_participants=False,
                  extra_event_labels=extra,
                  jobs=-1,
                  epochs_list=epochs_list)

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Global'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir='../analyses/MVPA/naives/pre-training/normal_vs_global/',
                  indiv_plot=False,
                  concat_participants=False,
                  extra_event_labels=extra,
                  jobs=-1,
                  epochs_list=epochs_list)

    files, extra = get_filepaths_from_file(
        '/analyses/MVPA/MVPA_analysis_list_sesh2.csv')

    epochs_list = []
    for file in files:
        epochs = mne.read_epochs(file)
        epochs_list.append(epochs)

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Obvious', 'Subtle'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir='../analyses/MVPA/naives/post-training/normal_vs_abnormal/',
                  indiv_plot=False,
                  concat_participants=False,
                  extra_event_labels=extra,
                  jobs=-1,
                  epochs_list=epochs_list)

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Global'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir='../analyses/MVPA/naives/post-training/normal_vs_global/',
                  indiv_plot=False,
                  concat_participants=False,
                  extra_event_labels=extra,
                  jobs=-1,
                  epochs_list=epochs_list)


def run_pickle_rads():
    files, extra = get_filepaths_from_file('')

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Obvious', 'Subtle'],
                  excluded_events=['Rate'],  # , 'Missed'
                  scoring="roc_auc",
                  output_dir='../analyses/rads_data_pkls/normal_v_abnormal/case_balanced',
                  indiv_plot=False,
                  concat_participants=False,
                  extra_event_labels=extra,
                  pickle_ouput=True,
                  jobs=-1)

    MVPA_analysis(files,
                  var1_events=['Normal'],
                  var2_events=['Global'],
                  excluded_events=['Rate'],  # , 'Missed'
                  scoring="roc_auc",
                  output_dir='../analyses/rads_data_pkls/normal_v_global/case_imbalanced',
                  indiv_plot=False,
                  concat_participants=False,
                  extra_event_labels=extra,
                  pickle_ouput=True,
                  jobs=-1)

    MVPA_analysis(files,
                  var1_events=['sesh_1/Obvious/resp_Abnormal'],
                  var2_events=['sesh_2/Obvious/resp_Abnormal'],
                  excluded_events=['Rate', 'Missed', 'ppt_6'],
                  scoring="roc_auc",
                  output_dir='../analyses',
                  indiv_plot=False,
                  concat_participants=True,
                  extra_event_labels=extra,
                  jobs=4)


if '__main__' in __name__:
    # run_within_training()
    run_AB_analysis()

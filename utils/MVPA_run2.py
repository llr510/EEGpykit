import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

from MVPA_analysis import MVPA_analysis, get_filepaths_from_file, group_MVPA_and_plot

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
    MVPA_analysis(files,
                  var1_events=['sesh_1'],
                  var2_events=['sesh_2'],
                  excluded_events=['Rate', 'Missed'],
                  scoring="roc_auc",
                  output_dir=Path(output_dir, 'Naives/between_session/all/'),
                  indiv_plot=indiv_plot,
                  jobs=jobs,
                  epochs_list=epochs_list,
                  concat_participants=True,
                  extra_event_labels=extra
                  )

    for resp in ['', '/Correct']:
        for sesh in ['/sesh_1', '/sesh_2']:
            # 1.	No main effect of Session in behavioral
            # a.	What about in the brain?
            # ii.	Contrast all normal vs all abnormal in session 1
            # iii.	Contrast all normal vs all abnormal in session 2
            MVPA_analysis(files,
                          var1_events=[f'normal{sesh}{resp}'],
                          var2_events=[f'malignant{sesh}{resp}', f'global{sesh}{resp}'],
                          excluded_events=['Rate', 'Missed'],
                          scoring="roc_auc",
                          output_dir=Path(output_dir, f'Naives{sesh}/normal_vs_abnormal/{resp}'),
                          indiv_plot=indiv_plot,
                          jobs=jobs,
                          epochs_list=epochs,
                          concat_participants=True,
                          extra_event_labels=extra
                          )

            # MVPA_analysis(files,
            #               var1_events=[f'normal{sesh}'],
            #               var2_events=[f'malignant{sesh}'],
            #               excluded_events=['Rate', 'Missed'],
            #               scoring="roc_auc",
            #               output_dir=Path(output_dir, f'Naives{sesh}/normal_vs_malignant/'),
            #               indiv_plot=indiv_plot,
            #               jobs=jobs,
            #               epochs_list=epochs_list,
            #               concat_participants=True,
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
            #               concat_participants=True,
            #               extra_event_labels=extra
            #               )
        # 2.	There is a main effect of image target present type in behavioral
        # a.	What about the brain?
        # i.	Across both session Obvious vs Priors (in addition just for hits)
        MVPA_analysis(files,
                      var1_events=[f'obvious{resp}'],
                      var2_events=[f'prior{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Naives/across_session/obvious_vs_priors{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )
        # ii.	Across both session Contra vs Priors (in addition just for hits)
        MVPA_analysis(files,
                      var1_events=[f'contra{resp}'],
                      var2_events=[f'prior{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Naives/across_session/contra_vs_priors{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )
        # iii.	Across both session Subtle vs Priors (in addition just for hits)
        MVPA_analysis(files,
                      var1_events=[f'subtle{resp}'],
                      var2_events=[f'prior{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Naives/across_session/subtle_vs_priors{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )

        # 3.	If you want to look at the interaction and image type then you would run these contrasts:
        # i.	Session 1 vs Session 2 just for obvious images (in addition just for hits)
        MVPA_analysis(files,
                      var1_events=[f'sesh_1/obvious{resp}'],
                      var2_events=[f'sesh_2/obvious{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Naives/between_session/Obvious{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )
        # ii.	Session 1 vs Session 2 just for priors images (in addition just for hits)
        MVPA_analysis(files,
                      var1_events=[f'sesh_1/prior{resp}'],
                      var2_events=[f'sesh_2/prior{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Naives/between_session/Priors{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )

        # iii.	Session 1 vs Session 2 just for contra images (in addition just for hits)
        MVPA_analysis(files,
                      var1_events=[f'sesh_1/contra{resp}'],
                      var2_events=[f'sesh_2/contra{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Naives/between_session/Contra{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )

        # iv.	Session 1 vs Session 2 just for subtle images (in addition just for hits)
        MVPA_analysis(files,
                      var1_events=[f'sesh_1/subtle{resp}'],
                      var2_events=[f'sesh_2/subtle{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Naives/between_session/Subtle{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )


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
        MVPA_analysis(files,
                      var1_events=[f'normal{resp}'],
                      var2_events=[f'malignant{resp}', f'global{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Rads/normal_vs_abnormal{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )

        # MVPA_analysis(files,
        #               var1_events=[f'normal'],
        #               var2_events=[f'malignant'],
        #               excluded_events=['Rate', 'Missed'],
        #               scoring="roc_auc",
        #               output_dir=Path(output_dir, f'Rads/normal_vs_malignant{resp}'),
        #               indiv_plot=indiv_plot,
        #               jobs=jobs,
        #               epochs_list=epochs_list,
        #               concat_participants=True,
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
        #               concat_participants=True,
        #               extra_event_labels=extra
        #               )
        # 2.	There is a main effect of image target present type in behavioral
        # a.	What about the brain?
        # i.	Across both session Obvious vs Priors (in addition just for hits)
        MVPA_analysis(files,
                      var1_events=[f'obvious{resp}'],
                      var2_events=[f'prior{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Rads/obvious_vs_priors{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )
        # ii.	Across both session Contra vs Priors (in addition just for hits)
        MVPA_analysis(files,
                      var1_events=[f'contra{resp}'],
                      var2_events=[f'prior{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Rads/contra_vs_priors{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )
        # iii.	Across both session Subtle vs Priors (in addition just for hits)
        MVPA_analysis(files,
                      var1_events=[f'subtle{resp}'],
                      var2_events=[f'prior{resp}'],
                      excluded_events=['Rate', 'Missed'],
                      scoring="roc_auc",
                      output_dir=Path(output_dir, f'Rads/subtle_vs_priors{resp}'),
                      indiv_plot=indiv_plot,
                      jobs=jobs,
                      epochs_list=epochs,
                      concat_participants=True,
                      extra_event_labels=extra
                      )


if '__main__' in __name__:
    # run_AB_analysis(output_dir='../analyses/MVPA-AB/')
    # run_training_analysis(output_dir='../analyses/MVPA-viking/', indiv_plot=False)
    # run_training_hits_tnegs(output_dir='../analyses/MVPA-viking/', indiv_plot=False)
    # run_rads_analysis(output_dir='../analyses/MVPA-viking/')

    run_rads_analysis(input_file='../analyses/MVPA/MVPA_analysis_list_rads.csv',
                      output_dir=args.output, indiv_plot=False, jobs=args.jobs)

    print("################# STARTING #################")
    parser = argparse.ArgumentParser(description='Analyse EEG data with MVPA')
    parser.add_argument('--analysis', type=str, required=True, help="Which analysis to do (Naives or Radiologists)")
    parser.add_argument('--input_file', type=str, required=True, help="List of epoched files to analyse")
    parser.add_argument('--output', type=str, required=True, help="Location to save subdirectories and figures to")
    parser.add_argument('--jobs', type=int, required=False, default=-1, help="how many processes to spawn. By default "
                                                                             "uses all available processes.")

    args = parser.parse_args()
    if args.analysis == 'Naives':
        run_training_analysis(input_file=args.input_file, output_dir=args.output, indiv_plot=False, jobs=args.jobs)
    elif args.analysis == 'Rads':
        run_rads_analysis(input_file=args.input_file, output_dir=args.output, indiv_plot=False, jobs=args.jobs)

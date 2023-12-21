import argparse
from pathlib import Path
import mne
import pandas as pd
from preprocessor import EEG_Experiment


def run_with_cli():
    print("################# STARTING #################")
    parser = argparse.ArgumentParser(description='Preprocess EEG data')
    parser.add_argument('--plist', type=str, required=True, help="The participant list file")
    parser.add_argument('--output', type=str, required=True, help="The output directory for pickles and epochs")
    parser.add_argument('--triggers', type=str, required=True, help="default trigger labels for experiment")

    parser.add_argument('--additional_events', type=str, required=False, help="")

    parser.add_argument('--tmin', type=float, required=True, help="time of epoch baseline start")
    parser.add_argument('--bmax', type=float, required=False, nargs='?', const=0, help="time of epoch baseline end")
    parser.add_argument('--tmax', type=float, required=True, help="time of epoch end")

    parser.add_argument('--filter_raw', action='store_true', required=False, help="")
    parser.add_argument('--skip_existing', action='store_true', required=False, help="")
    parser.add_argument('--plotting', action='store_true', required=False, help="")

    args = parser.parse_args()

    assert args.output.exists()

    ANTwave64 = mne.channels.read_custom_montage(fname=Path('montages', 'waveguard64_rescaled_small.xyz'),
                                                 coord_frame="unknown")
    # The basic trigger values and their labels in each recording.
    trigger_df = pd.read_csv(args.triggers)
    event_ids = {row['label']: row['value'] for idx, row in trigger_df.iterrows() if row['active'] == 1}

    study = EEG_Experiment(exp_filepath=args.plist,
                           output_path=args.output,
                           event_ids=event_ids,
                           montage=ANTwave64)

    if args.filter_raw:
        study.read_RAWs(skip_existing=args.skip_existing)

    study.preprocess_RAWs(tmin=args.tmin, bmax=args.bmax, tmax=args.tmax,
                          additional_events_fname=args.additional_events,
                          plotting=args.plotting,
                          skip_existing=args.skip_existing,
                          export_fif=True)


if '__main__' in __name__:
    run_with_cli()

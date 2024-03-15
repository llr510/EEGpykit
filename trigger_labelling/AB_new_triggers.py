from pathlib import Path
import pandas as pd
import re


def pivot_triggers(df, trial_start_trigger):
    # create new index. Increase index value by one every time a new trial begins
    x = -1
    for idx, row in df.iterrows():
        if row['trigger'] == trial_start_trigger:
            x += 1
        df.at[idx, 'idx'] = x
    # put experiment start trigger on first row so pivoted file is correct length
    df.at[0, 'idx'] = 0

    df = df.pivot_table(index=['idx'], columns='trigger', values='start')
    return df


def unpivot_triggers(df):
    # turns pivot table back into three columns
    df = df.melt(var_name='trigger', value_name='start').dropna().sort_values(
        'start').reset_index(drop=True)
    df['end'] = 0
    return df


def add_new_triggers(df_line, lock_lag_to_T2=True):
    # don't record triggers for practice blocks
    if df_line['BlockNumber'] != 6:
        # make new triggers for T1 and T2
        for T in [1, 2]:
            if df_line['ExperimentNumber'] == 1:
                condition = 'scene'
            elif df_line['ExperimentNumber'] == 2:
                condition = 'dot'
            elif df_line['ExperimentNumber'] == 'scenes':
                condition = 'scene'
            elif df_line['ExperimentNumber'] == 'dots':
                condition = 'dot'
            else:
                return
            selectivity = {}
            for Tx in [1, 2]:
                # If Trigger 11 or 21, must be selective t
                if not pd.isna(df_line[f'{Tx}1']):
                    selectivity[Tx] = 'S'
                # If Trigger 12 or 22, must be non-selective t
                elif not pd.isna(df_line[f'{Tx}2']):
                    selectivity[Tx] = 'NS'
                else:
                    print(df_line)
                    raise ValueError
            selectivity = selectivity[1] + '-' + selectivity[2]
            # # only care about blink at T2
            # if T == 2:
            if df_line['T1Accuracy'] == 1 and df_line['T2Accuracy'] == 1:
                blink = 'correct'
            elif df_line['T1Accuracy'] == 1 and df_line['T2Accuracy'] == 0:
                blink = 'attentional_blink'
            else:
                blink = 'incorrect'

            if lock_lag_to_T2:
                lag = 'lag' + str(int(df_line['T2pos']))
                # T1lag = 'T1lag' + str(int(df_line['T1pos']))
                new_trig = [f'T{T}', selectivity, condition, lag, blink]
                # new_trig = [f'T{T}', selectivity, condition, lag, T1lag, blink]
            else:
                lag = 'lag' + str(int(df_line[f'T{T}pos']))
                new_trig = [f'T{T}', selectivity, condition, lag, blink]

            new_trig = '/'.join(new_trig)
            df_line[new_trig] = df_line[f'T{T}']

    return df_line


def trg_file_parser(fpath, valid_triggers):
    """Reads ANTNeuro's .trg file format"""
    df = pd.read_csv(fpath, sep=" * ", names=['time', 'samples', 'trigger'], engine='python')
    df = df[df['trigger'].isin(valid_triggers)]
    df.reset_index(inplace=True)
    new_df = pd.DataFrame()
    new_df['trigger'] = df['trigger']
    new_df['start'] = df['time']
    new_df['end'] = 0
    return new_df


def make_events(marker_path, data_path, output_path):
    try:
        # read files
        valid_triggers = ['96', '12', '21', '22', '11', '97']
        marker_df = trg_file_parser(fpath=marker_path, valid_triggers=valid_triggers)
        data_df = pd.read_csv(data_path, sep="\t")
    except FileNotFoundError:
        print(f'No marker or behavioural data file for {data_path.name}')
        print(data_path)
        return
    print('behavioural len:', len(data_df), 'triggers len:', len(marker_df))
    # turns trigger file from long to wide
    marker_df = pivot_triggers(marker_df, trial_start_trigger='96')
    print('after pivot: ', 'behavioural len:', len(data_df), 'triggers len:', len(marker_df))
    marker_df.columns = marker_df.columns.map(str)
    # merge eeg triggers with behavioural data
    merged_df = pd.concat([data_df, marker_df], axis=1)
    # merge selective and non-selective triggers together by T
    merged_df['T1'] = merged_df['11'].combine_first(merged_df['12'])
    merged_df['T2'] = merged_df['21'].combine_first(merged_df['22'])
    # Apply all the other new triggers
    # print(merged_df)
    applied_df = merged_df.apply(add_new_triggers, axis=1)
    print('applied trigger len: ', len(applied_df))
    # Save out intermediate file
    # applied_df.to_csv(Path(marker_path.parent / 'AttentionalEEG_new_markers.csv'), index=False)

    # split only new triggers from merged file
    new = list(set(applied_df.columns) - set(merged_df.columns))
    marker_df = applied_df[new]
    # reverse pivoting process and create output file for adding back into brainstorm
    marker_df = unpivot_triggers(marker_df)
    print('new trigger len: ', len(marker_df))
    marker_df.to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    data_dir = Path(
        '/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/AB')
    assert data_dir.exists()

    output_dir = Path(data_dir, 'output')
    behavioural = list(data_dir.rglob("AttentionalEEG*.txt"))
    triggers = list(data_dir.rglob("*.trg"))
    assert len(triggers) == len(behavioural)

    # print(sorted([x.stem.split('_')[-1] for x in triggers]))
    # print(sorted([x.stem.split('_')[-1] for x in behavioural]))

    triggers = sorted(triggers, key=lambda x: x.stem.split('_')[-1])
    behavioural = sorted(behavioural, key=lambda x: x.stem.split('_')[-1])

    output = [Path(output_dir, x.stem).with_suffix('.csv') for x in triggers]

    for t, b, o in zip(triggers, behavioural, output):
        make_events(marker_path=t, data_path=b, output_path=o)


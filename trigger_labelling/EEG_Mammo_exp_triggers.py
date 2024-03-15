import pandas as pd
from pathlib import Path


def trg_file_parser(fpath, valid_triggers):
    """Reads ANTNeuro's .trg file format"""
    df = pd.read_csv(fpath, sep=" * ", names=['time', 'samples', 'trigger'], engine='python')
    df['trigger'] = pd.to_numeric(df['trigger'], errors="coerce")
    df = df[df['trigger'].isin(valid_triggers)]
    df.reset_index(inplace=True)
    new_df = pd.DataFrame()
    new_df['trigger'] = df['trigger']
    new_df['start'] = df['time']
    new_df['end'] = 0
    return new_df


def behavioural_file_parser(fpath, valid_triggers):
    df = pd.read_csv(fpath)
    df = df[df['ImageTag'].isin(valid_triggers)]
    return df


def trg_label_parser(fpath):
    df = pd.read_csv(fpath)
    df = df.loc[df['active'] == 1]
    d = dict(zip(df['label'], df['value']))
    return d


def corresponding_keys(val, dictionary):
    keys = []
    for k, v in dictionary.items():
        if val in v:
            keys.append(k)
    return keys


def make_new_label_file(df_b, df_trgs, label_index):
    l = []
    try:
        assert len(df_b) == len(df_trgs)
    except AssertionError:
        print(len(df_b), len(df_trgs))
        quit()

    for idx in range(len(df_trgs)):
        row_b = df_b.iloc[idx]
        row_t = df_trgs.iloc[idx]
        assert row_b['ImageTag'] == row_t['trigger']
        img_type = corresponding_keys(row_t['trigger'], label_index)
        img_type = '/'.join(img_type)
        time = row_t['start']
        decisiontype = row_b['Response']
        sd = row_b['Score']

        if sd == 'HIT' or sd == 'TN':
            accuracy = 'Correct'
        elif sd == 'MISS' or sd == 'FA':
            accuracy = 'Incorrect'
        elif sd == 'NONE':
            accuracy = 'Missed'
            decisiontype = 'Missed'
        else:
            raise ValueError

        l.append([f'View/{img_type}/{accuracy}/{sd}/resp_{decisiontype}', time])

    return pd.DataFrame(l)


def individual_label_file(behavioural_path, trigger_path, labels_path=None, valid_triggers=None):
    if labels_path:
        valid_triggers = trg_label_parser(labels_path)
        valid_triggers = list(valid_triggers.values())

    df_behavioural = behavioural_file_parser(behavioural_path, valid_triggers)
    df_trgs = trg_file_parser(trigger_path, valid_triggers)

    label_index = {}
    label_index['normal'] = [1, 6]

    label_index['malignant'] = [2, 7, 3, 8]
    label_index['obvious'] = [2, 7]
    label_index['subtle'] = [3, 8]

    label_index['global'] = [4, 5, 9, 10]
    label_index['contra'] = [4, 9]
    label_index['prior'] = [5, 10]

    df = make_new_label_file(df_behavioural, df_trgs, label_index)
    return df


def naives_parser(ID, order, sesh):
    behavioural_name = f'mammoTrain_Sub{ID}_Order{order}_Measurement{sesh}.txt'
    trg_name = f'EEGTraining_Sub{ID}Rec{sesh}.trg'
    newevents_name = f"Events_Sub{ID}Measurement{sesh}.csv"

    return behavioural_name, trg_name, newevents_name


def rads_parser(ID, order, sesh):
    behavioural_name = f'mammoTrain_Sub10{ID}_Order{order}_Measurement{sesh}.txt'
    trg_name = f'EEGTraining_Rad{ID}.trg'
    newevents_name = f"Events_Rad{ID}Measurement{sesh}.csv"

    return behavioural_name, trg_name, newevents_name


def experiment_label_files(userIDs, sessions, exp, labels_path, behavioural_dir, trigger_dir, output_dir=''):
    valid_triggers = trg_label_parser(labels_path)
    valid_triggers = list(valid_triggers.values())

    if exp == 'naives':
        parser = naives_parser
    elif exp == 'rads':
        parser = rads_parser

    for ID in userIDs:
        if int(ID) % 2 == 0:
            order = '0'
        else:
            order = '1'

        for sesh in sessions:
            behavioural_name, trg_name, newevents_name = parser(ID, order, sesh)
            print(behavioural_name)
            filepath = Path(behavioural_dir, behavioural_name)
            assert filepath.exists()
            trg_file = Path(trigger_dir, trg_name)
            assert trg_file.exists()
            neweventspath = Path(output_dir, newevents_name)

            df = individual_label_file(behavioural_path=filepath,
                                       trigger_path=trg_file,
                                       valid_triggers=valid_triggers)
            df.to_csv(neweventspath, index=None, header=False)


if '__main__' in __name__:
    userIDs = [1, 2, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16]
    sessions = [1, 2]
    experiment_label_files(userIDs, sessions, exp='naives',
                           labels_path='/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Naives/experiment_trigger_labels.csv',
                           behavioural_dir='/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Naives/behavioural',
                           trigger_dir='/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Naives/EEG/',
                           output_dir='/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Naives/output')

    userIDs = [1, 2, 3, 4, 5]
    sessions = [1]
    experiment_label_files(userIDs, sessions, exp='rads',
                           labels_path='/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Naives/experiment_trigger_labels.csv',
                           behavioural_dir='/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Radiologists/behavioural',
                           trigger_dir='/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Radiologists/EEG/',
                           output_dir='/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Radiologists/output')

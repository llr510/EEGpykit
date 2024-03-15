from json import loads
from scipy.stats import norm
import numpy as np
import glob
import datetime
from sklearn.metrics import auc
import pandas as pd
import csv
from pathlib import Path


def readTriggerfile(filename):
    triggerreader = csv.reader(open(filename))
    next(triggerreader)
    next(triggerreader)

    allowedtriggers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90]
    triggerlist = []
    for trigger in triggerreader:
        try:
            splittrig = trigger[0].strip().split(" ")
            filteredtrig = list(filter(lambda trig: len(trig.strip()) > 0, splittrig))
            time = float(filteredtrig[0].strip())
            epochtime = filteredtrig[1].strip()
            triggerN = int(filteredtrig[2].strip())

            if triggerN in allowedtriggers:
                triggerlist.append([triggerN, time])
        except Exception as e:
            print(e)
            pass
    return triggerlist


def read_data(filename):
    """
    This reads in the data from 1 filename
    It sorts ratings per block, by trial type (normal, obvious, subtle, priors) and outputs this
    list of ratingspertype: 0 normal, 1 obvious, 2 subtle, 3 global, 4 local, 5 total, 6 practice
    @param filename:
    @return:
    """
    imagetype_index = 1
    rating_index = 2
    rt_index = -2
    index_global = [4, 5, 9, 10]
    index_obvious = [2, 7]
    index_subtle = [3, 8]
    index_normal = [1, 6]
    index_practiceN = [94]
    index_practiceAb = [95]
    index_mirror = [6, 7, 8, 9, 10]
    index_og = [1, 2, 3, 4, 5]
    ratingspertype = [[], [], [], [], [], [], []]
    # ratings_mirrors =  [[],[],[],[], []]
    # ratings_ogs =  [[],[],[],[], []]
    ratings_in_order = []

    reader = csv.reader(open(filename, 'r'))
    next(reader)
    missedsum = 0
    for trial in reader:
        # if trial[rating_index] is not None:
        if int(trial[imagetype_index]) in index_normal:  # normal
            type = 0
        elif int(trial[imagetype_index]) in index_obvious:  # obvious
            type = 1
        elif int(trial[imagetype_index]) in index_subtle:  # subtle
            type = 2
        elif int(trial[imagetype_index]) in index_global:  # no visible abnormalities
            type = 3
        else:  # practice
            type = 6

        if trial[rating_index] == 'Normal':
            ratingscore = 0
            ratingspertype[type].append([ratingscore, trial[rt_index], trial[4]])
        elif trial[rating_index] == 'Abnormal':
            ratingscore = 1
            ratingspertype[type].append([ratingscore, trial[rt_index], trial[4]])
        else:
            ratingscore = np.nan
            missedsum += 1

        ## add type 'local' combining obvious and subtle
        if type == 1 or type == 2:
            type = 4
            ratingspertype[type].append([ratingscore, trial[rt_index], trial[4]])

        # add type "all abnormal"
        if type != 0 and type != 6:
            type = 5
            ratingspertype[type].append([ratingscore, trial[rt_index], trial[4]])
        # if int(trial[imagetype_index]) in index_mirror:
        #     mirroring = 1
        #     ratings_mirrors[type].append([ratingscore, trial[rt_index], trial[4]])
        # elif int(trial[imagetype_index]) in index_og:
        #     mirroring = 0
        #     ratings_ogs[type].append([ratingscore, trial[rt_index], trial[4]])

        ratings_in_order.append([ratingscore, trial[imagetype_index], type])
    print('{} missed {}'.format(filename, missedsum))
    return ratingspertype, ratings_in_order, missedsum


def checkEvents(brainstorm):
    """
    to create a dict of all event counts

    @param brainstorm:
    @return:
    """
    triggerlist = readTriggerfile(brainstorm)
    trigger_counts = {}
    for trigger in triggerlist:
        if trigger[0] in trigger_counts:
            trigger_counts[trigger[0]] += 1
        else:
            trigger_counts[trigger[0]] = 1
    return trigger_counts


def createEvents(trg_file, ratings_in_order, tablename, trialcounter=0):
    """

    @param brainstorm:
    @param ratings_in_order:
    @param tablename:
    @param trialcounter:
    @return:
    """
    triggerlist = readTriggerfile(trg_file)
    eventtable = open(tablename, 'w')

    index_global = [4, 5, 9, 10]
    index_obvious = [2, 7]
    index_subtle = [3, 8]
    index_normal = [1, 6]
    lookforresponse = 0
    brainstormcounter = 0  # one later than the viewing would be
    trialcounter = 0
    for brainstormtrigger in triggerlist:
        trigger = brainstormtrigger[0]
        brainstormcounter += 1
        if trigger < 11:
            if trigger != int(ratings_in_order[trialcounter][1]):
                print("THIS TRIGGER IS A GHOST OR SOMETHING WEIRD HAS HAPPENED")
                print(brainstormtrigger)
                print("ALARM ALARM")

            viewingtimestamp = brainstormtrigger[1]
            if trigger in index_normal:
                type = 'Normal'
                if ratings_in_order[trialcounter][0] == 0:
                    decision = 'Correct'
                    decisiontype = 'Normal'
                elif ratings_in_order[trialcounter][0] == 1:
                    decision = 'Incorrect'
                    decisiontype = 'Abnormal'
                else:
                    decision = 'Missed'
                    decisiontype = 'Missed'

            if trigger in index_obvious:
                type = 'Obvious'
                if ratings_in_order[trialcounter][0] == 0:
                    decision = 'Incorrect'
                    decisiontype = 'Normal'
                elif ratings_in_order[trialcounter][0] == 1:
                    decision = 'Correct'
                    decisiontype = 'Abnormal'
                else:
                    decision = 'Missed'
                    decisiontype = "Missed"
            if trigger in index_subtle:
                type = 'Subtle'
                if ratings_in_order[trialcounter][0] == 0:
                    decision = 'Incorrect'
                    decisiontype = 'Normal'
                elif ratings_in_order[trialcounter][0] == 1:
                    decision = 'Correct'
                    decisiontype = 'Abnormal'
                else:
                    decision = 'Missed'
                    decisiontype = "Missed"

            if trigger in index_global:
                type = 'Global'
                if ratings_in_order[trialcounter][0] == 0:
                    decision = 'Incorrect'
                    decisiontype = 'Normal'

                elif ratings_in_order[trialcounter][0] == 1:
                    decision = 'Correct'
                    decisiontype = 'Abnormal'
                else:
                    decision = 'Missed'
                    decisiontype = "Missed"

            eventtable.write(f'View/{type}/{decision}/resp_{decisiontype}, {viewingtimestamp}\n')
            trialcounter += 1

            # try:
            #     ##check for rating time
            #     potentialratingresponse = triggerlist[brainstormcounter]
            #     if potentialratingresponse[0] == 90:
            #         ratingtimestamp = potentialratingresponse[1]
            #         eventtable.write('Rate{}{}, {}\n'.format(type, decision, ratingtimestamp))
            #         eventtable.write('Rate{}, {}\n'.format(decision, ratingtimestamp))
            #         eventtable.write('RatePerceived{}, {}\n'.format(decisiontype, ratingtimestamp))
            # except:
            #     pass

    eventtable.close()
    read_file = pd.read_csv(tablename)
    read_file.to_csv(tablename, index=None)


def calc_dPrime_C_RT(values):
    d_primes = []
    criterions = []
    reactiontimes = []
    prophit_all = []
    propfa_all = []
    ntrials = {}
    for type in range(1, 6):
        negatives = values[0]
        positives = values[type]

        total_neg = len(negatives)
        total_pos = len(positives)

        ntrials[0] = total_neg  # keep track of how many trials they answered
        ntrials[type] = total_pos

        prophit = 0
        propmiss = 0
        proptn = 0
        propfa = 0

        rthit = []
        rtmiss = []
        rttn = []
        rtfa = []
        rttotal = []

        # count TN, FA, Misses and Hits
        for trial in negatives:
            rt_trial = float(trial[1])
            rttotal.append(rt_trial)
            if trial[0] == 0:
                proptn += 1
                rttn.append(rt_trial)
            elif trial[0] == 1:
                propfa += 1
                rtfa.append(rt_trial)

        for trial in positives:
            rt_trial = float(trial[1])
            rttotal.append(rt_trial)
            if trial[0] == 0:
                propmiss += 1
                rtmiss.append(rt_trial)
            elif trial[0] == 1:
                prophit += 1
                rthit.append(rt_trial)

        ## REACTION TIMES
        # Outliers: Filter out any above 4 STD
        meanrt_total = np.mean(rttotal)
        std_total = np.nanstd(rttotal)
        upper_threshold = meanrt_total + 3 * std_total
        lower_threshold = meanrt_total - 3 * std_total
        rttotal = np.clip(rttotal, lower_threshold, upper_threshold)

        # Calculate mean reaction times (rating)
        meanrt_tn = np.mean(rttn)
        meanrt_fa = np.mean(rtfa)
        meanrt_hit = np.mean(rthit)
        meanrt_miss = np.mean(rtmiss)
        meanrt_total = np.nanmean(rttotal)
        reaction_set = [meanrt_total, meanrt_hit, meanrt_miss, meanrt_tn, meanrt_fa]
        reactiontimes.append(reaction_set)

        # calculate mean viewing time

        # check if any values are zero, if so, set to small amount
        # this prevents z-score transform from outputting infinity
        if propfa == 0:
            propfa = 0.5
            total_neg += 0.5
        if prophit == 0:
            prophit = 0.5
            total_pos += 0.5
        if propmiss == 0:
            propmiss = 0.5
            total_pos += 0.5
        if proptn == 0:
            proptn = 0.5
            total_neg += 0.5

        proptn = proptn / total_neg
        propfa = propfa / total_neg
        prophit = prophit / total_pos
        propmiss = propmiss / total_pos

        prophit_all.append(prophit)
        propfa_all.append(propfa)

        z_tn = zscore_calc(proptn)
        z_fa = zscore_calc(propfa)
        z_hit = zscore_calc(prophit)
        z_miss = zscore_calc(propmiss)

        criterions.append((z_hit + z_fa) / -2)
        d_primes.append(z_hit - z_fa)
        # print(reaction_set)
    return d_primes, criterions, reactiontimes, prophit_all, propfa_all, ntrials


def calc_ratingtime(values, index):
    ratetimes = {}
    ratemedians = {}
    for imagetype in range(0, 6):
        ratingvalues = values[imagetype]
        ratingtimes = []
        for trial in range(len(ratingvalues)):
            ratingtimes.append(float(ratingvalues[trial][index]))

        ratemedians[imagetype] = np.nanmedian(ratingtimes)
        stdrate = np.nanstd(ratingtimes)
        meanrate = np.nanmean(ratingtimes)
        cutoffhigh = meanrate + 3 * stdrate
        cutofflow = meanrate - 3 * stdrate
        for ii in range(len(ratingtimes)):
            rate = ratingtimes[ii]
            if rate < cutofflow or rate > cutoffhigh:
                ratingtimes[ii] = np.nan
        ratetimes[imagetype] = np.nanmean(ratingtimes)
    return ratetimes, ratemedians


def create_AUC(ratings):
    cutoffs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    cutoffs = range(1, 100)
    falsealarms_ROC = [1]
    hits_O = [1]
    hits_S = [1]
    hits_PC = [1]
    for cutoff in cutoffs:
        x, y, z, prophits, propfa = calc_dPrime_C_RT(ratings, cutoff)
        falsealarms_ROC.append(propfa[0])
        hits_O.append(prophits[0])
        hits_S.append(prophits[1])
        hits_PC.append(prophits[2])
    falsealarms_ROC.append(0)
    hits_O.append(0)
    hits_PC.append(0)
    hits_S.append(0)

    ROC_obvious = np.column_stack((np.flip(falsealarms_ROC), np.flip(hits_O)))
    ROC_subtle = np.column_stack((np.flip(falsealarms_ROC), np.flip(hits_S)))
    ROC_PC = np.column_stack((np.flip(falsealarms_ROC), np.flip(hits_PC)))

    AUC_obvious = auc(falsealarms_ROC, hits_O)
    AUC_subtle = auc(falsealarms_ROC, hits_S)
    AUC_PC = auc(falsealarms_ROC, hits_PC)
    return ROC_obvious, ROC_subtle, ROC_PC, AUC_obvious, AUC_subtle, AUC_PC


def zscore_calc(proportion):
    return norm.ppf(proportion)


def createAllEventFiles_Rad(userIDs, sessions, eventStart, behavioural_dir, trg_dir, output_dir):
    """
    Combines eeg triggers with behavioural data for multiple participants to create more detailed triggers for
    epoching EEG data during preprocessing and analysis.

    @param userIDs: List of participant numbers
    @param sessions: List of sessions (just 1, unless multi part study)
    @param eventStart: Skips the first n rows of the behavioural file. These are practice trials.
    @param behavioural_dir: Location of behavioural .txt files
    @param trg_dir: Location of .trg EEG triggers
    @param output_dir: Location for output csv with new trigger labels
    @return:
    """
    analysis_date = datetime.datetime.now().strftime('%d%m%Y')
    for ID in userIDs:
        for sesh in sessions:

            if int(ID) % 2 == 0:
                order = '0'
            else:
                order = '1'

            behavioural_name = f'mammoTrain_Sub10{ID}_Order{order}_Measurement{sesh}.txt'
            filepath = Path(behavioural_dir, behavioural_name)
            ratings, ratings_in_order, missedsum = read_data(filepath)

            # reading in event file
            trg_name = f'EEGTraining_Rad{ID}.trg'
            trg_file = Path(trg_dir, trg_name)
            neweventspath = Path(output_dir, f"Events_Rad{ID}Measurement{sesh}.csv")
            createEvents(trg_file, ratings_in_order[eventStart:], neweventspath)


def createAllEventFiles(userIDs, sessions, eventStart, behavioural_dir, trg_dir, output_dir):
    """
    Combines eeg triggers with behavioural data for multiple participants to create more detailed triggers for
    epoching EEG data during preprocessing and analysis.

    @param userIDs: List of participant numbers
    @param sessions: List of sessions (just 1, unless multi part study)
    @param eventStart: Skips the first n rows of the behavioural file. These are practice trials.
    @param behavioural_dir: Location of behavioural .txt files
    @param trg_dir: Location of .trg EEG triggers
    @param output_dir: Location for output csv with new trigger labels
    @return:
    """
    analysis_date = datetime.datetime.now().strftime('%d%m%Y')
    for ID in userIDs:
        for sesh in sessions:

            if int(ID) % 2 == 0:
                order = '0'
            else:
                order = '1'

            behavioural_name = f'mammoTrain_Sub{ID}_Order{order}_Measurement{sesh}.txt'
            filepath = Path(behavioural_dir, behavioural_name)
            ratings, ratings_in_order, missedsum = read_data(filepath)

            # reading in event file

            trg_name = f'EEGTraining_Sub{ID}Rec{sesh}.trg'
            trg_file = Path(trg_dir, trg_name)
            neweventspath = Path(output_dir, f"Events_Sub{ID}Measurement{sesh}.csv")
            createEvents(trg_file, ratings_in_order[eventStart:], neweventspath)


if "__main__" in __name__:
    userIDs = [1, 2, 3, 4, 5]
    sessions = [1]
    working_dir = '/Volumes/psgroups/AttentionPerceptionLab/AttentionPerceptionLabStudent/UNDERGRADUATE PROJECTS/EEG MVPA Project/data/Radiologists'

    # userIDs = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # sessions = [1, 2]
    # working_dir = r'S:\AttentionPerceptionLabStudent\UNDERGRADUATE PROJECTS\EEG MVPA Project\data\Naives'

    behavioural_dir = Path(working_dir, 'behavioural')
    trg_dir = Path(working_dir, 'EEG')
    output_dir = Path(working_dir, 'output/')
    eventStart = 30  # set to 30 to exclude practice trials, 0 for all trials included

    createAllEventFiles_Rad(userIDs, sessions, eventStart, behavioural_dir, trg_dir, output_dir)
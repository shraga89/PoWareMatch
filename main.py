from os import listdir,path
from sklearn.model_selection import KFold
import pandas as pd
import random, time, datetime
import HHandler as HH
import Evaluator as E
from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTM import LSTMNet
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print('Number of available GPUs:', torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# torch.manual_seed(1)


def bulid_consensus(matches):
    consensus = {}
    for m in matches:
        for c in matches[m]:
            if c[0] not in consensus:
                consensus[c[0]] = 0
            consensus[c[0]] += 1
    return consensus


def bulid_consensus_seq(consensus, match_seq):
    consensus_seq = []
    for m in match_seq:
        val = 0
        if m in consensus:
            val = consensus[m]
        consensus_seq += [val, ]
    return consensus_seq


def build_feature_seq(seqs, is_multi_algs = False):
    if is_multi_algs:
        temp = zip(*seqs)
        out = list()
        for t in temp:
            out.append(list(t)[:-1] + t[-1])
        return out
    else:
        return zip(*seqs)


# def create_participants_groups():
#     sugs = {'with':[], 'without':[], 'ones':[]}
#     sugs['without'] += listdir('C:/Users/shrag/Dropbox/2StepManualMatch/ExperimentsNew/0')
#     sugs['without'] += listdir('C:/Users/shrag/Dropbox/2StepManualMatch/Experiments/0')
#     sugs['with'] += listdir('C:/Users/shrag/Dropbox/2StepManualMatch/ExperimentsNew/2')
#     sugs['ones'] += listdir('C:/Users/shrag/Dropbox/2StepManualMatch/Experiments/1a')
#     sugs['ones'] += listdir('C:/Users/shrag/Dropbox/2StepManualMatch/Experiments/1b')
#     sugs['with'] += listdir('C:/Users/shrag/Dropbox/2StepManualMatch/Experiments/2')
#     df = pd.read_csv(str(dir + 'participants.csv'))
#     sugs['without'] += list(df[df['group'] == 0])
#     sugs['with'] += list(df[df['group'] == 2])
#     return sugs


def create_algs_dict():
    alg_matches = {}
    algs = pd.read_csv(str(dir + 'algs.csv'))
    for alg in list(algs.columns) :
        if alg in ['candName', 'targName']: continue
        temp = algs[['candName', 'targName', alg]].values.tolist()
        alg_matches[alg] = {(val[1], val[0]): val[2] for val in temp}
    return alg_matches


def algs_seq(match_seq, alg_matches, alg = 'all'):
    alg_seq = []
    if alg != 'all':
        alg_match = alg_matches[alg]
        for m in match_seq:
            sim = alg_match[m] if m in alg_match else 0
            alg_seq += [sim, ]
    else:
        for m in match_seq:
            sims = []
            for alg in alg_matches:
                sim = alg_matches[alg][m] if m in alg_matches[alg] else 0
                sims += [sim, ]
            alg_seq += [sims, ]
    return alg_seq

matchers_full = listdir(str(dir + 'ExperimentData/'))
matchers = []
for m in matchers_full:
    if path.exists(dir + 'ExperimentData/' + m + '/Excel - CIDX/report.log') and m not in groups['ones']:
        matchers += [m]
print('found ', len(matchers), ' matchers')
matchers_ids = dict(enumerate(matchers))

evaluator = E.Evaluator()
quality = {}
features = {}
match_seqs = {}
acc_seqs = {}
pred_seqs = {}
conf_seqs = {}
time_seqs = {}
consensus_seqs = {}
sug_seqs = {}
alg_seqs = {}
alg_matches = create_algs_dict()
matches = {}
matcher_count = 1
matcher_number = len(matchers)+1
df = pd.DataFrame(columns=['alg', 'matcher', 'correspondence', 'conf', 'time', 'con', 'sug', 'alg_val', 'pred', 'real'])
kfold = KFold(folds, True, 1)
row_i = 1
for matcher in matchers:
    matcher_count += 1
#     print('Matcher Number', matcher)
    Hmatcher = HH.HHandler(matcher)
    match = Hmatcher.getMatch()
    match_seqs[matcher], conf_seqs[matcher], time_seqs[matcher] = Hmatcher.getSeqs()
    is_sug = 0
    if matcher in groups['without']:
        is_sug = 1
    sug_seqs[matcher] = len(match_seqs[matcher]) * [is_sug, ]
    quality[matcher] = evaluator.evaluate(match)
    acc_seqs[matcher] = evaluator.getCorrSeq(match_seqs[matcher])
    matches[matcher] = match

i = 1
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
print(st)
matches_train = {}
for alg in list(alg_matches.keys()) + ['all']:
# for alg in ['all']:
    print('Staring', alg, 'Experiment')
    if alg == 'all':
        seq_len = 16
    model = LSTMNet(seq_len, HIDDEN_DIM, target_len, device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for trainset, testset in kfold.split(matchers):
        test = [matchers_ids[m] for m in testset]
        train = [matchers_ids[m] for m in trainset]
        matches_train = {k: matches[k] for k in matches if k in train}
        st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
        print("Starting fold " + str(i) + ' ' + str(st))
        consensus = bulid_consensus(matches_train)
        for e in range(epochs):
            print("Starting epoch " + str(e))
            for matcher in train:
                consensus_seqs[matcher] = bulid_consensus_seq(consensus, match_seqs[matcher])
                alg_seqs[matcher] = algs_seq(match_seqs[matcher], alg_matches, alg)
                X = torch.tensor(list(build_feature_seq([conf_seqs[matcher],
                                                         time_seqs[matcher],
                                                         consensus_seqs[matcher],
                                                         sug_seqs[matcher],
                                                         alg_seqs[matcher]],
                                                         alg == 'all')),
                                 dtype=torch.float)
                Y = torch.tensor(list(acc_seqs[matcher]), dtype=torch.long)
                model.zero_grad()
                Y_hat = model(X)
                loss = loss_function(Y_hat, Y)
                loss.backward()
                optimizer.step()
                for clf_name, clf in classifiers:
                    if len(list(np.unique(Y))) != 1:
                        clf.fit(X=X, y=Y)
        with torch.no_grad():
            for matcher in test:
                consensus_seqs[matcher] = bulid_consensus_seq(consensus, match_seqs[matcher])
                alg_seqs[matcher] = algs_seq(match_seqs[matcher], alg_matches, alg)
                X = torch.tensor(list(build_feature_seq([conf_seqs[matcher],
                                                         time_seqs[matcher],
                                                         consensus_seqs[matcher],
                                                         sug_seqs[matcher],
                                                         alg_seqs[matcher]],
                                                        alg == 'all')),
                                 dtype=torch.float)
                Y = torch.tensor(list(acc_seqs[matcher]), dtype=torch.long)
                Y_hat = model(X)
                pred_seqs[('deep ' + alg, matcher)] = torch.tensor(torch.max(Y_hat, 1)[1], dtype=torch.float).tolist()
                for corr, conf, time, con, sug, alg_val, pred, real in zip(match_seqs[matcher], conf_seqs[matcher],
                                            time_seqs[matcher], consensus_seqs[matcher],
                                            sug_seqs[matcher], alg_seqs[matcher],
                                            pred_seqs[('deep ' + alg, matcher)], acc_seqs[matcher]):
                    df.loc[row_i] = np.array(['deep ' + alg, matcher, corr, conf, time, con, sug, alg_val, pred, real])
                    row_i += 1
                for clf_name, clf in classifiers:
                    Y_hat = clf.predict(X)
                    pred_seqs[(clf_name + ' ' + alg, matcher)] = Y_hat
                    for corr, conf, time, con, sug, alg_val, pred, real in zip(match_seqs[matcher], conf_seqs[matcher],
                                                                               time_seqs[matcher], consensus_seqs[matcher],
                                                                               sug_seqs[matcher], alg_seqs[matcher],
                                                                               pred_seqs[(clf_name + ' ' + alg, matcher)],
                                                                               acc_seqs[matcher]):
                        df.loc[row_i] = np.array([clf_name + ' ' + alg, matcher, corr, conf, time, con, sug, alg_val, pred, real])
                        row_i += 1
    i += 1
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
df.to_csv('res/raw_' + st + '.csv')

matchers = df['matcher'].unique().tolist()
algs = df['alg'].unique().tolist()
all_correct = 65
res = pd.DataFrame(columns=['alg', 'matcher', 'P', 'R'])
row_i = 1
for alg in algs:
    for matcher in matchers:
        sum_correct = len(df[(df['alg'] == alg)
                             & (df['matcher'] == matcher)
                             & (df['pred'] == df['real'])
                             & (df['real'] == 1)])
        sum_answered = len(df[(df['alg'] == alg)
                             & (df['matcher'] == matcher)
                             & (df['pred'] == 1)])
        res.loc[row_i] = np.array([alg,
                                   matcher,
                                   sum_correct/sum_answered if sum_answered > 0 else 0.0,
                                   sum_correct/all_correct])
        row_i += 1

df.to_csv('res/eval_' + st + '.csv')

res[['P','R']]=res[['P','R']].astype(float)
pd.DataFrame(res.groupby('alg')[['P','R']].mean()).to_csv('res/sum_' + st + '.csv')
import pandas as pd
import numpy as np
import config as conf
from scipy import stats
from itertools import permutations, combinations
from sklearn.metrics import accuracy_score


def quality2pandas(quality, export=False):
    global_Calibration = 0.748 - 0.542
    df = pd.DataFrame(columns=['matcher', 'P', 'R', 'Res', 'Cal'])
    i = 1
    for m in quality:
        df.loc[i] = np.array([m, quality[m][0], quality[m][1], quality[m][2], quality[m][3]])
        i += 1
    df['P_bin'] = 0.0
    df.loc[pd.to_numeric(df['P']) > 0.5, 'P_bin'] = 1.0
    df['R_bin'] = 0.0
    df.loc[pd.to_numeric(df['R']) > 0.5, 'R_bin'] = 1.0
    df['Res_bin'] = 0.0
    # THINK ABOUT THE VALUES!
    df.loc[pd.to_numeric(df['Res']).abs() > 0.5, 'Res_bin'] = 1.0
    df['Cal_bin'] = 0.0
    # THINK ABOUT THE VALUES!
    df.loc[pd.to_numeric(df['Cal']).abs() < global_Calibration, 'Cal_bin'] = 1.0
    # Pairs
    bins = ['P_bin', 'R_bin', 'Res_bin', 'Cal_bin']
    for i in range(2, len(bins) + 1):
        for c in combinations(bins, i):
            df[c] = 1.0
            for j in range(i):
                df[c] *= df[c[j]]
    if export:
        df.sort_values(by='matcher', ascending=True).to_csv('./quality.csv', index=False)
    return df


def features2pandas(features, export=False):
    k = list(features.keys())[0]
    cols = ['matcher'] + [str(i) for i in range(len(list(features[k])))]
    df = pd.DataFrame(columns=cols)
    i = 1
    for m in features:
        df.loc[i] = np.array([str(m), ] + list(features[m]))
        i += 1
    if export:
        df.sort_values(by='matcher', ascending=True).to_csv('./features.csv', index=False)
    return df


def eval_model(pred, real):
    return accuracy_score(real, pred)


def summerize_results(res, clfs):
    eval = pd.DataFrame(columns=['Q', 'Clf', 'Acc'])
    i = 1
    for clf, _ in clfs:
        for q in ['P_bin', 'R_bin', 'Res_bin', 'Cal_bin']:
            pred = res[q + '_' + clf].copy()
            real = res[q]
            acc = accuracy_score(real, pred)
            eval.loc[i] = np.array([q, clf, acc])
            i += 1
    return eval


def goodman_kruskal_gamma(m, n):
    """
    compute the Goodman and Kruskal gamma rank correlation coefficient;
    this statistic ignores ties is unsuitable when the number of ties in the
    data is high. it's also slow.
    """
    num = 0
    den = 0
    for (i, j) in permutations(range(len(m)), 2):
        m_dir = m[i] - m[j]
        n_dir = n[i] - n[j]
        sign = m_dir * n_dir
        if sign > 0:
            num += 1
            den += 1
        elif sign < 0:
            num -= 1
            den += 1
    if den == 0:
        return tuple((0.0, 1.0))
    gamma = num / float(den)
    N = min(len(m), len(n))
    t = gamma * np.sqrt((float(den) / ((N)) * (1 - np.power(gamma, 2))))
    df = N ** 2 - 2
    if t >= 0:
        pval = 1 - stats.t.cdf(t, df=df)
    else:
        pval = stats.t.cdf(t, df=df)
    return tuple((gamma, pval))


class Evaluator:

    def __init__(self):
        self.exact = []
        self.loadExact()
        self.correspondences = {}
        self.emptyMatrix = None
        self.createMatrixShell()

    def loadExact(self):
        with open(conf.dir + 'Excel2CIDX.csv') as f:
            for line in f.readlines():
                line_split = line.replace(' ', '').split(',')
                self.exact += [tuple((line_split[1], line_split[0])), ]

    def createMatrixShell(self):
        last = None
        i, j = -1, 0
        with open(conf.dir + 'Matrix.csv') as f:
            for line in f.readlines():
                line_split = line.replace(' ', '').split(',')
                if 'instance' in line_split[0]:
                    continue
                if last != line_split[2]:
                    i += 1
                    j = 0
                    last = line_split[2]
                self.correspondences[tuple((line_split[3], line_split[2]))] = (i, j)
                j += 1
        self.emptyMatrix = np.zeros([i + 1, j + 1], dtype=float)

    def cor2entry(self, cor):
        return self.correspondences[cor]

    def getMatrix4Match(self, match):
        matrix = np.copy(self.emptyMatrix)
        for cor in match:
            matrix[self.cor2entry(cor[0])] = cor[1]
        return matrix

    def getRes(self, match):
        confs = list()
        accs = list()
        for cor in match:
            name = cor[0]
            confs += [cor[1], ]
            if cor[0] in self.exact:
                accs += [1.0, ]
            else:
                accs += [0.0, ]
        return goodman_kruskal_gamma(confs, accs)

    def getCorrSeq(self, match_seq):
        accs = list()
        for cor in match_seq:
            if cor in self.exact:
                accs += [1.0, ]
            else:
                accs += [0.0, ]
        return accs

    def evaluate(self, match, k=None):
        match_set = [c[0] for c in match]
        if len(match_set) < 1:
            return 0.0, 0.0, 0.0, 0.0
        P = float(len(set(self.exact).intersection(set(match_set)))) / len(set(match_set))
        if k:
            R = float(len(set(self.exact).intersection(set(match_set)))) / k
        else:
            R = float(len(set(self.exact).intersection(set(match_set)))) / len(set(self.exact))
        Res = self.getRes(match)
        # ASK RAKEFET ABOUT GAMMA
        Cal = np.array([c[1] for c in match]).mean() - P
        if P == 0.0 or R == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        # F = (2 * P * R) / (P + R)
        res_val = Res[0]
        if Res[1] < 0.1:
            res_val = 0.0
        return P, R, res_val, Cal

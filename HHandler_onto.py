import config as conf
from datetime import datetime
from collections import OrderedDict

class HHandler:

    def __init__(self, matcherId, H = None):
        self.matcherId = matcherId
        self.dir = conf.dir
        if not H:
            self.H = OrderedDict()
            self.loadData()
        else:
            self.H = H

    def loadData(self):
        last_line = ''
        line_split = ''
        last_ai = ''
        last_time = 0.0
        with open(self.dir + 'ontoExperimentData/' + self.matcherId + '/Excel - CIDX/report.log') as f:
            for line in f.readlines():
                last_line_split = line_split
                line_split = line.split('|')
                time = datetime.strptime(line_split[0].split(',')[0], '%Y-%m-%d %H:%M:%S')
                if len(line_split) < 5:
                    continue
                if line_split[4] != 'matched':
                    continue
                if len(last_line_split) < 5 or last_line_split[4] != 'matched':
                    # last_ai = last_line_split[-1].replace('\n', '')
                    last_ai = last_line_split[-1].replace('\n', '').split('.')[-1].replace('"','').replace('@en', '')\
                        .replace(' ', '').lower()
                    last_time = datetime.strptime(last_line_split[0].split(',')[0], '%Y-%m-%d %H:%M:%S')
                # corr = tuple((last_ai, line_split[8]))
                curr_ai = line_split[8].split('.')[-1].replace('"', '').replace('@en', '').replace(' ', '').lower()
                if 'name of an entity' in line_split[8]:
                    curr_ai = line_split[8].split('.')[-2].replace('"', '').replace('@en', '').replace(' ', '').lower()
                elif 'foaf:' in line_split[8]:
                    continue
                # elif '@en' in line_split[-2]:
                #     curr_ai = line_split[-2].split('.')[-1].replace('"', '').replace('@en', '').replace(' ', '').lower()
                # else:
                #     continue
                corr = tuple((last_ai, curr_ai))
                if last_ai == '':
                    print('err1', last_line_split[-1].replace('\n', '').split('.'))
                    continue
                if curr_ai == '':
                    print('err2', line_split[8].replace('\n', '').split('.'))
                    print('err2', line_split[-2].replace('\n', '').split('.'))
                    continue
                elapsed_time = float((time - last_time).seconds)
                if corr not in self.H:
                    self.H[corr] = []
                self.H[corr] += [tuple((float(line_split[9].replace('\n', '')), elapsed_time, time)), ]

    def getMatch(self):
        match = []
        con = 0.0
        for c in self.H:
            con = self.H[c][-1][0]
            if float(con) > 0.0:
                match += [tuple((c, float(con))), ]
        return match

    def getSeqs(self):
        cors = []
        confs = []
        times = []
        for c in self.H:
            con = self.H[c][-1][0]
            time = self.H[c][-1][1]
            if float(con) > 0.0:
                cors += [c, ]
                confs += [con, ]
                times += [time, ]
        return cors, confs, times

    def split2ns(self, n, matcher):
        H_list = list(self.H.items())
        submatcherslist = list(zip(*(H_list[i:] for i in range(n))))
        submatchers = {}
        for i in range(len(submatcherslist)):
            new_id = matcher.zfill(3) + '_' + str(n).zfill(3) + '_' + str(i).zfill(4)
            submatchers[new_id] = HHandler(new_id, OrderedDict(submatcherslist[i]))
        return submatchers

    def extract_behavioural_features(self):
        count_distinct_corr = float(len(self.H.keys()))
        count_general_corr, count_mind_change, sum_conf, max_conf, \
        min_conf, sum_time, max_time, min_time = [0.0, ]*8
        for corr in self.H:
            count_general_corr += len(self.H[corr])
            if len(self.H[corr]) > 1:
                count_mind_change += len(self.H[corr])
            conf = self.H[corr][-1][0]
            time = self.H[corr][-1][1]
            sum_conf += conf
            sum_time += time
            if conf < min_conf:
                min_conf = conf
            if conf > max_conf:
                max_conf = conf
            if time < min_time:
                min_time = time
            if time > max_time:
                max_time = time
        avg_conf = sum_conf/count_general_corr
        avg_time = sum_time / count_general_corr
        return count_distinct_corr, count_general_corr, count_mind_change, avg_conf, max_conf, \
            min_conf, avg_time, max_time, min_time
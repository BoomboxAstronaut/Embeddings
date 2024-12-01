import numpy as np
from pickle import load, dump
from collections import Counter
from tqdm import tqdm

ldct = {
    'alpha': {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'},
    'unifx': {'s_', 'd_', 'r_', 'n_', 't_', 'x_', 'y_', 'a_', 'i_', 'o_', '_a', '_o', '_e', '_i'},
    'unifx_l': {'s_', 'y_', '_a', '_e'},
    'fdbl': {'b', 'c', 'd', 'f', 'g', 'l', 'm', 'n', 'p', 'r', 's', 't'},
    'bdbl': {'b', 'd', 'g', 'm', 'l', 'n', 'p', 'r', 't'},
    'avwls': {'a', 'e', 'i', 'o', 'u', 'y'},
    'bvwls': {'a', 'e', 'i', 'o', 'u'},
    'cvwls': {'a', 'e', 'o', 'i', 'y'},
    'dvwls': {'a', 'e', 'o', 'u'},
    'fvwls': {'a', 'o', 'i', 'u'},
    'uafxs': {'logy_', 'ity_', 'try_', 'cy_', 's_', 'y_'}
}

def surrounds(afx, window=3, merge=False, exact=False):
    left_cnt, right_cnt = Counter(), Counter()
    if not exact:
        afx = afx.strip('_')
        targets = [x.strip('_').split(afx) for x in full_words if afx in x]
    else:
        targets = [x.split(afx) for x in full_words if afx in x]
        targets = [(x[0].strip('_'), x[1].strip('_')) for x in targets]

    if not exact or (exact and afx[-1] == '_'):
        for x in targets:
            idx = min(len(x[0]), window)
            if idx: left_cnt[x[0][-idx:]] += 1
    if not exact or (exact and afx[0] == '_'):
        for x in targets:
            idx = min(len(x[1]), window)
            if idx: right_cnt[x[1][:idx]] += 1

    if merge or exact:
        for x in left_cnt: right_cnt[x] += left_cnt[x]
        return right_cnt
    else:
        return left_cnt, right_cnt

def kld(P, Q=None, dist=ldsts['nd']):
    pcnt = Counter({x: 3 for x in ldct['alpha']})
    for x in P:
        for y in x: pcnt[y] += P[x]  
    psum = sum(x for x in pcnt.values())
    if Q:
        qcnt = Counter({x: 3 for x in ldct['alpha']})
        for x in Q:
            for y in x: qcnt[y] += Q[x]
        for x in ldct['alpha']:
            if pcnt[x] == 3 and qcnt[x] == 3:
                pcnt.pop(x)
                qcnt.pop(x)
        qsum = sum(x for x in qcnt.values())
        return sum([(pcnt[x] / psum) * np.log2((pcnt[x] / psum) / (qcnt[x] / qsum)) for x in pcnt])
    else:
        return sum([(pcnt[x] / psum) * np.log2((pcnt[x] / psum) / dist[x]) for x in pcnt])

def pulld(afx, len_lim=False):
    aln = len(afx)
    sub_set = [x for x in tf2 if afx in x]
    out = []
    for x in sub_set:
        i = 1
        if x[0] == '_':
            while len(x[:-i]) > aln:
                if x[:-i] in sub_set:
                    break
                i += 1
            else: out.append(x)
        else:
            while len(x[i:]) > aln:
                if x[i:] in sub_set:
                    break
                i += 1
            else: out.append(x)
    if not len_lim: return [x for x in out if x != afx]
    else: return [x for x in out if len(x) == len(afx)+1 and x != afx]

def pullu(afx):
    i = 1
    if afx[0] == '_':
        while i < len(afx):
            if afx[:-i] in tf2:
                return afx[:-i]
            i += 1
    else:
        while i < len(afx):
            if afx[i:] in tf2:
                return afx[i:]
            i += 1



def main():
    
    with open(r'D:\dstore\nlp\w2v\fwords', 'rt') as f:
        full_words = Counter({f'_{x[1]}_': int(x[0]) for x in [x.strip().split() for x in f.readlines()]})
    with open(r'D:\dstore\tmp\4', 'rb') as f:
        dsts = load(f)
    with open(r'D:\dstore\tmp\5', 'rb') as f:
        tf2, rntp, drntp = load(f)
    smalls = {x for x in full_words if len(x) > 4 and len(x) < 12}
    bigs = {x for x in full_words if len(x) > 7}

    tf2 = Counter()
    for x in tqdm(smalls):
        x = x.strip('_')
        group = [y for y in bigs if x in y]
        for y in group:
            out = y.split(x)
            if len(out) > 2:
                out.append(f'_{out[1]}')
                out.append(f'{out[1]}_')
                out.pop(1)
            for z in out:
                if z not in ('_', ''):
                    tf2[z] += 1
    tf2 = Counter({x[0]: x[1] for x in tf2.most_common() if x[1] > 2})

    pdst, sdst, ndst = Counter(), Counter(), Counter()
    for x in full_words:
        x = x.strip('_')
        for l in x: ndst[l] += 1
        i = round((len(x)+0.1) / 2)
        for l in x[:i]: pdst[l] += 1
        for l in x[-i:]: sdst[l] += 1
    pdst = Counter({x[0]: x[1] / pdst.total() for x in pdst.most_common()})
    sdst = Counter({x[0]: x[1] / sdst.total() for x in sdst.most_common()})
    ndst = Counter({x[0]: x[1] / ndst.total() for x in ndst.most_common()})

    rntp = {}
    for x in tqdm(tf2):
        hold = []
        if (x[0] == '_' and len(x) > 7) or (x[-1] == '_' and len(x) < 8):
            fd = dsts['sd']
        else:
            fd = dsts['pd']
        for i in range(1, 4):
            o1, o2 = surrounds(x, i)
            hold.append([kld(surrounds(x, i, merge=True)), kld(o2, dist=fd), kld(surrounds(x, i, exact=True), dist=fd)])
        rntp[x] = np.array(hold[::-1]).T

    hold = []
    for i in range(1, 4):
        frel, brel = Counter(), Counter()
        for x in full_words:
            if len(x) > 3+i:
                x = x.strip('_')
                for l in x[-i:]: brel[l] += 1
                for l in x[:i]: frel[l] += 1
        hold.append([kld(frel, dist='pre'), kld(brel, dist='suf')])
    hold = [[x]*3 for x in np.array(hold[::-1]).T]
    frel, brel = np.array(hold[0]), np.array(hold[1])
    long_rel = np.array([rntp[x] for x in tf2 if len(x) > 9]).mean(axis=0)

    drntp = {}
    for x in tqdm(tf2):
        above, below = pulld(x, True), pullu(x)
        
        if above: above = np.array([rntp[y] for y in above]).mean(axis=0)
        else: above = long_rel
        if below: below = rntp[below]
        elif x[0] == '_': below = frel
        else: below = brel
        middle = rntp[x]
        drntp[x] = (above-middle) - (middle-below)

if __name__ == '__main__':
    main()
import numpy as np
import pytest
from pickle import load
from collections import Counter

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

def get_nested(subject, stage, src):
    hln = len(subject) * 3
    hold = {x for x in src if subject in x and len(x) < hln}
    if subject.strip('_') in hold:
        hold.remove(subject.strip('_'))
    return hold

def extract_afx(subject, group, stage, src) -> list:
    if stage == 1:
        hold = []
        group = [f'_{x}_'.split(subject) for x in group]
        for x in group:
            for y in x:
                if len(y) > 2 or y in singles:
                    hold.append(y)
    return hold

def afx_count(inp: Counter, stage: int):
    hold = dict()
    nons = []
    if stage == 1:
        src = inp
    else:
        src = {x.strip('_') for x in inp}
    for subject in tqdm(inp):
        out = get_nested(subject, stage, src)
        if out:
            hold[subject] = out
        else:
            nons.append(subject)
    outp = Counter()
    for subject in hold:
        out = extract_afx(subject, hold[subject], stage, inp)
        if out:
            for x in out:
                outp[x] += 1
    return (Counter({x[0]: x[1] for x in outp.most_common() if x[1] > 2}), nons)

def search(term, corpus, exc=None):
    if not exc:
        return sorted({x for x in corpus if term in x})
    elif isinstance(exc, str):
        return sorted({x for x in corpus if term in x and exc not in x})
    else:
        return sorted({x for x in corpus if term in x and all(y not in x for y in exc)})

def gsub(target: str, afx: str, best=True, amode=0, guard=True, dbg=False):
    if amode == 0:
        if len(target) - len(afx) < 4: return
    else:
        if len(target) - len(afx) < 2: return
    rep = target.replace(afx, '')
    candidates = [target.replace(afx, '_')]
    if afx[0] == '_':
        pre = True
    else:
        pre = False
    if not pre or amode in (1, 2):
        if afx in ldct['uafxs']:

            if afx == 'logy_':
                candidates.append(f'{rep}a_')
                candidates.append(f'{rep}l_')
                candidates.append(f'{rep[:-1]}_')
                candidates.append(f'{rep[:-2]}_')
            elif afx == 'ity_':
                if rep.endswith('abil'):
                    candidates.append(rep.replace('abil', 'able_'))
                if rep.endswith('ibil'):
                    candidates.append(rep.replace('ibil', 'ible_'))
            elif afx == 'try_':
                candidates.append(f'{rep}t_')
            elif afx == 'cy_':
                candidates.append(f'{rep}t_')
                if rep[-1] == 'a': 
                    candidates.append(f'{rep}te_')
            elif afx == 's_':
                if rep[-1] in ['s', 'i', 'u']: return
            elif afx == 'y_':
                if rep[-1] in ldct['bvwls']: return

        if dbg: print(candidates)
        if afx[0] in ldct['bvwls']:
            if afx[0] == rep[-1]:
                return
            dreps = [rep]
            if len(rep) > 4 and rep[-1] == rep[-2] and rep[-1] in ldct['bdbl']:
                dreps.append(rep[:-1])
                candidates.append(f'{rep[:-1]}_')
            elif rep[-1] in ldct['fvwls']:
                candidates.append(f'{rep[:-1]}_')
                candidates.append(f'{rep[:-1]}e_')
                if rep[-1] == 'i':
                    candidates.append(f'{rep[:-1]}y_')
            for drep in dreps:
                candidates.append(f'{drep}e_')
                candidates.append(f'{drep}a_')
                candidates.append(f'{drep}y_')
                if afx[0] == 'e':
                    if drep[-1] == 'v':
                        candidates.append(f'{drep[:-1]}f_')
                    if drep[-1] == 'm':
                        candidates.append(f'{drep[:-1]}_')
                elif afx[0] == 'i':
                    if drep[-1] == 't':
                        candidates.append(f'{drep[:-2]}e_')
                        candidates.append(f'{drep[:-2]}_')
                        if drep[-2] == 'i':
                            candidates.append(f'{drep[:-1]}sh_')
                        elif drep.endswith('ipt'):
                            candidates.append(f'{drep[:-2]}be_')
                        elif drep.endswith('orpt'):
                            candidates.append(f'{drep[:-2]}b_')
                    elif drep[-1] == 's':
                        candidates.append(f'{drep[:-1]}e_')
                        if drep[-2] in ldct['avwls']:
                            candidates.append(f'{drep[:-1]}de_')
                            candidates.append(f'{drep[:-1]}re_')
                        elif drep[-2] == 's' and len(drep) > 2:
                            if drep[-3] in ldct['dvwls']:
                                candidates.append(f'{drep[:-2]}de_')
                            elif drep[-3] == 'i':
                                candidates.append(f'{drep[:-2]}t_')
                        elif drep[-2] == 'r':
                            candidates.append(f'{drep[:-1]}t_')
                        elif drep[-2] == 'n':
                            candidates.append(f'{drep[:-1]}d_')
                elif afx[0] == 'a':
                    if drep.endswith('ti'):
                        candidates.append(f'{drep[:-2]}ce_')

    if dbg: print(candidates)
    if amode == 0: 
        out = sorted([(x, full_words[x]) for x in candidates if (x in full_words and full_words[x] > 4)], key=lambda x: x[1])
    elif amode == 1:
        out = sorted([(x, tf2[x]) for x in candidates if x in tf2], key=lambda x: x[1])
    else:
        out = []
        if pre:
            for x in candidates:
                mafx = f'{x[1:]}_'
                full = f'{x}_'
                if mafx in tf2 and tf2[mafx] > 8:
                    out.append((mafx, tf2[mafx]))
                elif full in full_words and full_words[full] > 256:
                    out.append((full, np.log2(full_words[full])))
        else:
            for x in candidates:
                mafx = f'_{x[:-1]}'
                full = f'_{x}'
                if mafx in tf2 and tf2[mafx] > 8:
                    out.append((mafx, tf2[mafx]))
                elif full in full_words and full_words[full] > 256:
                    out.append((full, np.log2(full_words[full])))
        out = sorted(out, key=lambda x: x[1])
    if out:
        if best: return out[-1][0]
        else: return out

def compress(oafx, window=1, dbg=False, t=full_words):
    cnt = Counter()
    if oafx[0] == '_': pre = True
    else: pre = False
    if len(oafx) == 1:
        for x in [x for x in t if x[0] == '_']:
            cnt[x[1]] += 1
        for x in [x for x in t if x[-1] == '_']:
            cnt[x[-2]] += 1
    else:
        afx = oafx.strip('_')
        if pre:
            group = [x.split(afx)[1].strip('_') for x in t if afx in x]
            for x in group:
                idx = min(len(x), window)
                cnt[x[:idx]] += 1
        else:
            group = [x.split(afx)[0].strip('_') for x in t if afx in x]
            for x in group:
                idx = min(len(x), window)
                cnt[x[-idx:]] += 1
    if '' in cnt: cnt.pop('')
    return Counter({x[0]: x[1] for x in cnt.most_common() if x[1] > 1})

def target_removal(afx, exc1=None, exc2=None, exe=False, dbg=False):
    if exc1 and exc2:
        if isinstance(exc1, str) and isinstance(exc2, str):
            targets = [x for x in tf2 if afx in x and x not in (afx, exc1) and exc2 not in x]
        elif isinstance(exc1, str):
            targets = [x for x in tf2 if afx in x and x not in (afx, exc1) and all(y not in x for y in exc2)]
        elif isinstance(exc2, str):
            targets = [x for x in tf2 if afx in x and x not in (afx, exc1) and exc2 not in x]
        else:
            targets = [x for x in tf2 if afx in x and x not in (afx, *exc1) and all(y not in x for y in exc2)]
    elif exc1:
        if isinstance(exc1, str):
            targets = [x for x in tf2 if afx in x and x not in (afx, exc1)]
        else:
            targets = [x for x in tf2 if afx in x and x not in (afx, *exc1)]
    elif exc2:
        if isinstance(exc2, str):
            targets = [x for x in tf2 if afx in x and x != afx and exc2 not in x]
        else:
            targets = [x for x in tf2 if afx in x and x != afx and all(y not in x for y in exc2)]
    else:
        targets = [x for x in tf2 if afx in x and x != afx]
    rem = []
    if dbg: print(targets)
    for x in targets:
        tmp = gsub(x, afx, amode=1)
        if tmp: rem.append((x, tmp))
    if exe:
        for x in rem:
            tf2[x[1]] += tf2[x[0]]
            tf2.pop(x[0])
    else: return rem

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
            else:
                out.append(x)
        else:
            while len(x[i:]) > aln:
                if x[i:] in sub_set:
                    break
                i += 1
            else:
                out.append(x)
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

def chain(afx):
    out = sorted([x for x in tf2 if afx in x or x in afx], key=lambda x: len(x))[-1]
    return sorted([x for x in tf2 if x in out], key=lambda x: len(x), reverse=True)

def upper(afx, depth=1):
    hold = [afx]
    while depth > 0:
        grp = []
        while hold:
            tmp = pulld(hold.pop(), True)
            if tmp:
                for y in tmp:
                    grp.append(y)
        hold.extend(grp)
        depth -= 1
    if hold:
        hold = np.mean([drntp[x] for x in hold], axis=0)
        return np.array([*[np.mean(x) for x in hold], *[np.mean(x) for x in hold.T]])
    else: return np.array([0]*6)

def remean(rearr):
    return np.array([*[np.mean(y) for y in rearr], *[np.mean(y) for y in rearr.T]])

def chain_scan(target, bridge_coeff=1, dbg=False):
    scores = erw.copy()
    words = chain(target)

    for x in words: scores.append(rntp[x] * wgts)
    if target[0] == '_': scores.append(frw)
    else: scores.append(brw)
    hold = zdre.copy()
    for i in range(1, len(scores)-1): hold.append((scores[i+1]-scores[i])+(scores[i-1]-scores[i]))
    if target[0] == '_': hold.extend([dfrw, dfrw])
    else: hold.extend([dbrw, dbrw])

    out = []
    for i in range(2, len(hold)-2):
        u1, d1, md = hold[i-1].copy(), hold[i+1].copy(), hold[i]
        if dbg: print(words[i-2], (md-u1).mean(), (md-d1).mean())
        if (md-u1).mean() > bridge_coeff or (md-d1).mean() > bridge_coeff:
            u2, d2 = hold[i-2].copy(), hold[i+2].copy()
            u2[u2 > u1] *= 0
            u1[u1 > hold[i-2]] *= 0
            d2[d2 > d1] *= 0
            d1[d1 > hold[i+2]] *= 0
            u1 = u1 + u2
            d1 = d1 + d2
        out.append((md-u1)+(md-d1))

    if dbg:
        for i, x in enumerate(out): print(words[i], '\n', remean(x).mean(), '\n', x)
    return [(words[i], remean(x).mean()) for i, x in enumerate(out)]

def removal1(min_score=-1, min_ratio=0.2):
    for x in ('less_', 'ness_'): target_removal(x, exe=True)
    target_removal('es_', exc1=('es_', 's_'), exc2=('is_', 'us_', 'ss_', 'series_', 'species_'), exe=True)
    target_removal('s_', exc1=('es_', 's_'), exc2=('is_', 'us_', 'ss_', 'series_', 'species_'), exe=True)
    ends = []
    for x in tf2:
        if not pulld(x):
            ends.append(x)
    scores = {x: [] for x in tf2}
    for x in ends:
        for y in chain_scan(x):
            scores[y[0]].append(y[1])
    t = {x[0]: (np.mean(x[1]), np.median(x[1]), len([y for y in x[1] if y > 0]), len(x[1])) for x in scores.items()}
    return [x[0] for x in t.items() if x[1][0] < min_score and x[1][2] / x[1][3] < min_ratio]

with open(r'D:\dstore\nlp\w2v\fwords', 'rt') as f:
    full_words = Counter({f'_{x[1]}_': int(x[0]) for x in [x.strip().split() for x in f.readlines()]})
for x in [x for x in full_words if len(x) < 5]:
    full_words.pop(x)
for x in [x for x in full_words if "'" in x]:
    out = x.split("'")
    if f'{out[0]}_' in full_words:
        full_words[f'{out[0]}_'] += full_words[x]
    full_words.pop(x)
with open(r'D:\dstore\tmp\4', 'rb') as f:
    dsts = load(f)
with open(r'D:\dstore\tmp\5', 'rb') as f:
    tf2, rntp, drntp = load(f)
wgts = np.array([[1, 1, 1.25], [1, 1.25, 1.5], [1.25, 1.5, 1.75]])
frw, brw, erw = dsts['fr']*wgts, dsts['br']*wgts, [dsts['lr']*wgts]
dfrw, dbrw = [], []
for x in [x for x in tf2 if len(x) == 2 and x[0] == '_']: dfrw.append(rntp[x] - frw)
for x in [x for x in tf2 if len(x) == 2 and x[-1] == '_']: dbrw.append(rntp[x] - brw)
dfrw, dbrw = np.mean(dfrw, axis=0)*wgts, np.mean(dbrw, axis=0)*wgts
zdre = [np.array([[0]*3]*3), np.array([[0]*3]*3)]

def main():
    removal1()

if __name__ == '__main__':



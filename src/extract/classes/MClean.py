
import numpy as np
from collections import Counter
from pickle import dump, load
from btk import cprint

class MCleaner:

    def __init__(self, wlst, verif, ldct, roots, wparts, afxscore, fbreaks):
        self.verif = verif
        self.ldct = ldct
        self.wlst = wlst
        self.roots = roots
        self.wparts = wparts
        self.afxscore = afxscore
        self.failed_brk = fbreaks

    def search(self, term: str, corpus: Counter=None, exc: str|tuple[str]=None, pos: bool=True):
        #Returns all items that contain the input affix
        rem_terms = [f'_{term.strip("_")}', f'{term.strip("_")}_', f'_{term.strip("_")}_']
        if not corpus: corpus = self.wlst
        if not exc:
            if pos: res = sorted({x for x in corpus if term in x if '_' in x})
            else: res = sorted({x for x in corpus if term in x})
        elif isinstance(exc, str):
            if pos:  res = sorted({x for x in corpus if term in x and exc not in x and '_' in x})
            else: res = sorted({x for x in corpus if term in x and exc not in x})
        else:
            if pos: res = sorted({x for x in corpus if term in x and all(y not in x for y in exc) and '_' in x})
            else: res = sorted({x for x in corpus if term in x and all(y not in x for y in exc)})
        for x in rem_terms:
            if x in res: res.remove(x)
        return res

    def surrounds(self, afx: str, window: int=3, merge: bool=False, exact: bool=False) -> tuple[Counter] | Counter:
        """
        Counts the letters adjacent to the input affix in all words from the word list.
        By default the input affix will have its positional indicator _ removed.

        Args:
            afx (str): Target affix.
            window (int, optional): Distance from input affix to count. Defaults to 3.
            merge (bool, optional): Combine left and right side counts before returning. Defaults to False.
            exact (bool, optional): Counted words must respect affixes positional indicator. Defaults to False.

        Returns:
            tuple[Counter] | Counter: Counts of letters adjacent to input affix.
        """
        left_cnt, right_cnt = Counter(), Counter()
        if not exact:
            afx = afx.strip('_')
            targets = [x.strip('_').split(afx) for x in self.wlst if afx in x]
        else:
            targets = [x.split(afx) for x in self.wlst if afx in x]
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
        else: return left_cnt, right_cnt

    def kld(self, P: Counter, Q: Counter=None, pfloor: int=0) -> float:
        """
        Kullback Leibler Divergence Calculation

        Args:
            P (Counter): Counts of letters
            Q (Counter, optional): Letter counts or distribution to compare against P. Defaults to letter distribution of entire the word list.
            base_value (int, optional): Base count of all letters. Higher values reduces effect of 0s. Defaults to 3.

        Returns:
            float: Relative Entropy of the two counts / distributions.
        """
        if not pfloor:
            if P.total() < 156: pfloor = 1
            elif P.total() < 312: pfloor = 2
            else: pfloor = 3
        if not Q: Q = self.dsts['nd']

        pcnt = Counter({x: pfloor for x in self.ldct['alpha']})
        for x in P:
            for y in x: pcnt[y] += P[x]  
        psum = sum(x for x in pcnt.values())
        if Q.total() > 1.5:
            qcnt = Counter({x: pfloor for x in self.ldct['alpha']})
            for x in Q:
                for y in x: qcnt[y] += Q[x]
            for x in self.ldct['alpha']:
                if pcnt[x] == pfloor and qcnt[x] == pfloor:
                    pcnt.pop(x)
                    qcnt.pop(x)
            qsum = sum(x for x in qcnt.values())
            return sum([(pcnt[x] / psum) * np.log2((pcnt[x] / psum) / (qcnt[x] / qsum)) for x in pcnt])
        else: return sum([(pcnt[x] / psum) * np.log2((pcnt[x] / psum) / Q[x]) for x in pcnt])

    def gsub(self, target: str, afx: str, amode: int=0, best: bool=True, fltr: bool=True) -> str:
        """
        Remove the affix from a word following english rules, returning the proper root
        Will not work if the remaining word/affix is too short (2 chars for word, 1 char for affix)

        Args:
            target (str): Target word to remove affix from
            afx (str): Affix to remove from target word
            best (bool, optional): Whether to return the best option or all options. Defaults to True.
            amode (int, optional): Verification list. 0 verifies against words list. 1 verifies against affix list. 2 verifies against both with affix and co-affix. Defaults to 0.

        Returns:
            (str): Target with affix removed in its neutral form
        """
        if not amode and len(target) - len(afx) < 3: return
        elif amode and len(target) - len(afx) < 2: return
        rep = target.replace(afx, '')
        candidates = [target.replace(afx, '_')]
        spafx = {'tion_', 'sion_', 'ian_', 'es_', 'cy_', 's_', 'y_'}
        if afx[0] == '_': pre = True
        else: pre = False

        if afx in spafx: #Specific Affix Substitution Rules
            if afx == 'sion_':
                if rep.endswith('is'):
                    candidates.append(f'{rep[:-1]}t_')
                elif rep[-1] in self.ldct['vwl2']:
                    candidates.append(f'{rep}de_')
                    candidates.append(f'{rep}re_')
                elif rep[-1] == 'n': candidates.append(f'{rep[:-1]}d_')
                elif rep[-1] == 'r': candidates.append(f'{rep}t_')
            elif afx == 'tion_':
                if rep.endswith('lu'):
                    candidates.append(f'{rep[:-1]}ve_')
            elif afx == 'ian_':
                if rep.endswith('ar'):
                    candidates.append(f'{rep[:-2]}_')
            elif afx == 'cy_':
                candidates.append(f'{rep}t_')
                candidates.append(f'{rep}ce_')
                if rep[-1] == 'a': candidates.append(f'{rep}te_')
            elif afx == 'es_':
                if rep[-1] == 'v':
                    candidates.append(f'{rep[:-1]}f_')
                    candidates.append(f'{rep[:-1]}fe_')
                elif rep.endswith('ic'):
                    candidates.append(f'{rep[:-2]}ex_')
            elif afx == 's_':
                if rep[-1] in ['s', 'i', 'u']: return
            elif afx == 'y_':
                if rep[-1] in self.ldct['vwl2']: return

        if pre:
            if afx[-1] not in self.ldct['vwl2']: candidates.append(f'_{afx[-1]}{rep}')
            #if rep[0] in self.ldct['vwl2']: candidates.append(f'_{rep[1:]}')
            else:
                if len(rep) > 4 and rep[0] == rep[1] and rep[0] in self.ldct['fdbl']:
                    candidates.append(f'_{rep[1:]}')
        else:
            if len(afx) > 2: candidates.append(f'{rep}e_')

            if afx[0] not in self.ldct['vwl2']:
                candidates.append(f'{rep}{afx[0]}_')
                candidates.append(f'{rep}{afx[0]}e_')
                #v cfx_
                if rep[-1] in self.ldct['vwl2']:
                    candidates.append(f'{rep[:-1]}_')
                    candidates.append(f'{rep[:-1]}e_')
                    if rep[-1] == 'i': candidates.append(f'{rep[:-1]}y_')
                #c cfx_
                else: pass
            else:
                #v vfx_
                if rep[-1] in self.ldct['vwl2']:
                    candidates.append(f'{rep[:-1]}_')
                    candidates.append(f'{rep[:-1]}e_')
                    if rep[-1] == 'i': candidates.append(f'{rep[:-1]}y_')
                #c vfx_
                else:
                    if len(rep) > 4 and rep[-1] == rep[-2] and rep[-1] in self.ldct['bdbl']:
                        candidates.append(f'{rep[:-1]}_')

        if target in candidates: candidates.remove(target)
        if candidates and fltr:
            candidates = set(candidates)
            if amode == 0: out = sorted([(x, 50000) if x in self.roots else (x, self.full_scores[x]) for x in candidates if x in self.full_scores], key=lambda x: np.log(x[1] * (len(x[0])-1)))
            elif amode == 1: out = sorted([(x, self.afx[x]) for x in candidates if x in self.afx], key=lambda x: np.log(x[1] * (len(x[0])-1)))
            elif amode == 2:
                out = []
                if pre:
                    for x in candidates:
                        mafx, full = f'{x[1:]}_', f'{x}_'
                        if mafx in self.afx and self.afx[mafx] > 8: out.append((mafx, self.afx[mafx]))
                        elif full in self.wlst and self.wlst[full] > 256: out.append((full, np.log2(self.wlst[full])))
                else:
                    for x in candidates:
                        mafx, full = f'_{x[:-1]}', f'_{x}'
                        if mafx in self.afx and self.afx[mafx] > 8: out.append((mafx, self.afx[mafx]))
                        elif full in self.wlst and self.wlst[full] > 256: out.append((full, np.log2(self.wlst[full])))
                out = sorted(out, key=lambda x: np.log(x[1] * (len(x[0])-1)))
            if out:
                if best: return out[-1][0]
                else: return out
        else: return candidates

    def find_sub_chain(self, word, afxl=None, sub_depth=0):
        if not afxl: afxl = self.verif
        fout = set()
        #Create list with word and all affixes found in word
        targets = [(word, sorted([x for x in afxl if x in word]), [], [])]
        while targets:
            word, found_afxs, rem, rafxs = targets.pop()
            is_match = False
            while found_afxs:
                #For every affix thats found in a word
                afx = found_afxs.pop()
                if len(afx) / len(word) > 0.6: continue
                sub = self.gsub(word, afx)
                if sub:
                    is_match = True
                    rem.append(word)
                    rafxs.append(afx)
                    #If there are still affixes left add a new group to targets
                    #Allows continuation if word reaches an early dead end
                    if found_afxs and any(y in sub for y in found_afxs): targets.append((sub, found_afxs.copy(), rem.copy(), rafxs.copy()))
                    found_afxs = sorted([x for x in afxl if x in sub])
                    word = sub
                for x in targets[::-1]:
                    if word == x[0] and found_afxs == x[1]: targets.remove(x)
                if len(word) < 7: break
            #Once no more affixes are found in a word add it to the outputs if atleast 1 matcheed
            #If word is still long and no match, use double sub
            if is_match: fout.add((tuple(rafxs), (*rem, word)))
            elif len(word) > 7 and sub_depth:
                dsub = self.g2sub(word, afxl, depth=sub_depth)
                if dsub:
                    for k, ds in enumerate(dsub):
                        dbl_found = sorted([x for x in afxl if x in ds[0]])
                        rcp, rafc = rem.copy(), rafxs.copy()
                        rcp.extend(ds[1])
                        rafc.extend(ds[2])
                        if dbl_found: targets.append((ds[0], dbl_found, rcp, rafc))
                        else: fout.add((tuple(rafc), (*rcp, ds[0])))
        return tuple(fout)

    def h2gsub(self, word, it_mx=2, bridges=False):
        stg = 0
        bkd = {word: {'afx': [], 'reps': [], 'ub': [], 'chk': False, 'bchk': False}}
        while stg <= it_mx:
            new = []
            for w in bkd.items():
                if not w[1]['chk']:
                    for nx in [x for x in self.verif if x in w]:
                        rep = w.replace(nx, '_')
                        w[1]['afx'].append(nx)
                        w[1]['reps'].append(rep)
                        if rep not in bkd: new.append((rep, w[1]['afx'].copy(), w[1]['reps'].copy(), w[1]['ub'].copy()))
                    w[1]['chk'] = True
                elif bridges and not w[1]['bchk']:
                    b1 = [*[(w.replace(f'_{b}', '_'), f'_{b}') for b in self.ldct['bridges'] if w.startswith(f'_{b}')], 
                        *[(w.replace(f'{b}_', '_'), f'{b}_') for b in self.ldct['bridges'] if w.endswith(f'{b}_')]]
                    b2 = [b for b in b1 if any(bz in b[0] for bz in self.verif)]
                    for b in b2:
                        if b[0] not in new and b[0] not in bkd:
                            obr = [q for q in w[1]['ub']]
                            new.append((b[0], [z for z in w[1]['afx']], [z for z in w[1]['reps']], [b[1], *obr]))
                    w[1]['bchk'] = True
            for nw in new:
                if nw[0] not in bkd:
                    bkd[nw[0]] = {'afx': nw[1], 'reps': nw[2], 'ub': nw[3], 'chk': False, 'bchk': False}
            stg += 1
        bkd.pop(word)
        kl = list(bkd.keys())
        for k in kl[::-1]:
            if k not in self.full_scores and k not in self.roots:
                bkd.pop(k)
                kl.remove(k)
        if kl:
            ok = sorted(kl, key=lambda x: len(x))[0]
            if not bridges: return (ok, bkd[ok]['reps'][:-1], bkd[ok]['afx'][:-1])
            else: return (ok, bkd[ok]['reps'][:-1], bkd[ok]['afx'][:-1], bkd[ok]['ub'])

    def g2sub(self, word, afxl=None, depth=1):
        if not afxl: afxl = self.verif
        queue = {(word, x, (), ()) for x in afxl if x in word}
        dupe_key = set()
        out = []
        while depth > 0 and queue:
            newq = set()
            for x in queue:
                step = self.gsub(x[0], x[1], fltr=False)
                if step:
                    for y in step:
                        for z in afxl:
                            if z in y:
                                newq.add((y, z, (*x[2], x[0]), (*x[3], x[1])))
            queue = newq.copy()
            depth -= 1
            for x in queue:
                g = self.gsub(x[0], x[1])
                if g:
                    if (g, tuple(sorted((x[0], x[1], *x[2], *x[3])))) not in dupe_key:
                        out.append((g, (*x[2], x[0]), (*x[3], x[1])))
                        dupe_key.add((g, tuple(sorted((x[0], x[1], *x[2], *x[3])))))
        if out: return out

    def merger(self, results):
        r1o, r2o, c = [], [], 0
        results = tuple(sorted([(x[0], x[1]) for x in results], key=lambda x: len(x[1]), reverse=True))
        while c < len(results[0][1]):
            r1, r2 = [], []
            for x in results:
                if c < len(x[1]):
                    if x[1][c] not in r2: r2.append(x[1][c])
                if c < len(x[1])-1:
                    if x[0][c] not in r1: r1.append(x[0][c])
            if r2:
                if len(r2) > 1: r2o.append(tuple(r2))
                else:  r2o.append(r2[0])
            if r1:
                if len(r1) > 1: r1o.append(tuple(r1))
                else: r1o.append(r1[0])
            c += 1
        return (r1o[::-1], r2o[::-1])

    def packer(self, results, idx, cid=None):
        while cid:
            if cid[0] == '1':
                if len(cid) < 2 or int(cid[1]) >= len(results): return False
                results = results[int(cid[1])]
                cid = cid[2:]
                if cid: cid = f'2{cid}'
            elif cid[0] == '2':
                if len(cid) < 2 or int(cid[1]) > len(idx): return False
                idx = idx[int(cid[1]):]
                cid = cid[2:]
                if cid and isinstance(idx[0], tuple) and cid[0] in ('0', '1'):
                    idx[0] = idx[0][int(cid[0])]
                    cid = ''
            else: cid = ''
        out = {}
        if isinstance(idx[0], tuple):
            print('roots must be a word not a tuple of words')
            return False
        else:
            results = [(x[0][:len(idx)-1], x[1][:len(idx)]) for x in results if idx[0] in x[1]]
            for x in results:
                for j, y in enumerate(x[1][:-1]):
                    if y not in out:
                        out[y] = [idx[0], []]
                        for z in x[0][j:]:
                            out[y][1].append(z)
        return out

    def pretty_printer(self, result):
        if isinstance(result[1][0], tuple): out_str = f'{(12+len(result[1][0][0])+len(result[1][0][1])) * " "}'
        else: out_str = f'{(6+len(result[1][0])) * " "}'
        for i, x in enumerate(result[0]):
            #x is affix, wg is word
            wg = result[1][i+1]
            if isinstance(x, tuple):
                if isinstance(wg, tuple):
                    out_str += ' '
                    if len(x) > 2 or len(wg) > 2:
                        for y in wg:
                            ag = sorted([z for z in x if z in y])
                            if len(ag) == 2: out_str += f'{ag[0]}{(len(y)-len(ag[0])-len(ag[1])) * " "}{ag[1]}    '
                            elif ag[0][0] == '_': out_str += f'{ag[0]}{(len(y)-len(ag[0])) * " "}    '
                            else: out_str += f'{(len(y)-len(ag[0])) * " "}{ag[0]}    '
                    else:
                        sp1 = len(wg[0])-len(x[0])
                        sp2 = len(wg[1])-len(x[1])
                        if x[0][0] == '_': out_str += f'{x[0]}{sp1 * " "}    '
                        else: out_str += f'{sp1 * " "}{x[0]}    '
                        if x[1][0] == '_': out_str += f'{x[1]}{sp2 * " "}    '
                        else: out_str += f'{sp2 * " "}{x[1]}    '
                    out_str += ' '
                else:
                    x = sorted(x)
                    out_str += f'{x[0]}{(len(wg)-(len(x[1])+len(x[0]))) * " "}{x[1]}    '
            else:
                if isinstance(wg, tuple):
                    out_str += ' '
                    if x in wg[0] and x[0] == '_': out_str += f'{x}{(len(wg[0])-len(x)) * " "}{(4+len(wg[1])) * " "}    '
                    elif x in wg[0] and x[-1] == '_': out_str += f'{(len(wg[0])-len(x)) * " "}{x}{(4+len(wg[1])) * " "}    '
                    elif x in wg[1] and x[0] == '_': out_str += f'{(4+len(wg[0])) * " "}{x}{(len(wg[1])-len(x)) * " "}    '
                    elif x in wg[1] and x[-1] == '_': out_str += f'{(4+len(wg[0])) * " "}{(len(wg[1])-len(x)) * " "}{x}    '
                    out_str += ' '
                elif x[0] == '_': out_str += f'{x}{(len(wg)-len(x)) * " "}    '
                else: out_str += f'{(len(wg)-len(x)) * " "}{x}    '
        cprint(((result[1][-1], result[1][0]), (result[1], out_str)), (1, 3))

    def score_eval(self, wrd, pack):
        #Adds scores of removed words to affixes and roots
        print(wrd, pack)
        score = self.wlst[wrd]
        self.wlst[pack[0]] += score
        for fx in pack[1]:
            self.afxscore[fx] += score
        self.wparts[wrd] = ((pack[0],), tuple(pack[1]))

    def aflys(self, afx, exact=False):
        if afx[0] == '_': pre = True
        else: pre = False
        _afx = afx.strip('_')
        afg = self.search(_afx)
        anum = len(afg)
        ara = {}
        print('\n')
        print("{: >40} {: >4}".format('Affix:', afx))
        print("{: >40} {: >4}".format('Matches:', anum))
        print("{: >40} {: >4}".format('Total Words:', len(self.full_scores)))
        print("{: >40} {: >4}".format('Usage:', f'{round((anum / len(self.full_scores))*100, 4)}%'))
        print('\n')
        prx = [self.surrounds(afx, window=i, exact=exact)  for i in range(1, 5)]
        stats = {}
        for i, x in enumerate(prx):
            if not exact:
                for g in [(0, 'l'), (1, 'r')]:
                    stats[f'{i+1}{g[1]}'] = x[g[0]]
                    stats[f'{g[1]}{i+1}crv'] = ((x[g[0]].most_common()[0][1] - x[g[0]].most_common()[-1][1]) * len(x[g[0]])) / 2
                    ara[f'{i}KLD-{g[1]}'] = round(self.kld(x[g[0]]), 4)
            else:
                stats[f'{i+1}'] = x
                stats[f'{i+1}crv'] = ((x.most_common()[0][1] - x.most_common()[-1][1]) * len(x)) / 2
                ara[f'{i}KLD'] = round(self.kld(x), 4)
        alts, c, d, e = [], 0, 0, 0
        if pre: alts.append(afx[:-1])
        else: alts.append(afx[1:])
        for x in self.afx.most_common():
            e += 1
            if x[0] != afx:
                if c < 2 and afx in x[0]:
                    alts.append(x[0])
                    c += 1
                elif d < 2 and _afx in x[0]:
                    alts.append(x[0])
                    d += 1
                if (c >= 2 and d >= 2) or e >= 10000: break
        nafxm = {x: list() for x in (['1', '2', '3', 'd-1', 'd-2', 'd-3'] if exact else ['l1', 'l2', 'l3', 'd-l1', 'd-l2', 'd-l3', 'r1', 'r2', 'r3', 'd-r1', 'd-r2', 'd-r3'])}
        for x in alts:
            for i in range(1, 4):
                altc = self.surrounds(x, window=i, exact=exact)
                if exact:
                    nafxm[f'{i}'].append(round(self.kld(altc), 3))
                    nafxm[f'd-{i}'].append(round(ara[str(i)+"KLD"] - self.kld(altc), 3))
                else:
                    nafxm[f'l{i}'].append(round(self.kld(altc[0]), 3))
                    nafxm[f'r{i}'].append(round(self.kld(altc[1]), 3))
                    nafxm[f'd-l{i}'].append(round(ara[str(i)+"KLD-l"] - self.kld(altc[0]), 3))
                    nafxm[f'd-r{i}'].append(round(ara[str(i)+"KLD-r"] - self.kld(altc[1]), 3))
        if exact:
            pack = [['Window Size:', 'Frags:', 'Total:', 'Max Curve:', 'Curve %:', 'KLD:', 'Afx Coef:']]
            for j in range(4):
                col = []
                col.append(j+1)
                col.append(len(prx[j]))
                col.append(prx[j].total())
                col.append(stats[f"{j+1}crv"])
                col.append(round((1-(prx[j].total() / (stats[f"{j+1}crv"] if stats[f"{j+1}crv"] else 1))) * 100, 3))
                col.append(ara[f'{j}KLD'])
                col.append(round(((1+(prx[j].total() / (stats[f"{j+1}crv"] if stats[f"{j+1}crv"] else 1)))**ara[f'{j}KLD']) * ((ara[f'{j}KLD']**(1+(prx[j].total() / (stats[f"{j+1}crv"] if stats[f"{j+1}crv"] else 1))))), 4))
                pack.append(col)
            cprint(pack, pos=[3, 4], halign='r', col_width=10)
            print('\n')
            print('\n\t\t\tRelative Entropy\t\t\t\t  Diff\n')
            cprint(['Affix/WSize:', 1, 2, 3, 1, 2, 3], pos=[3, 4], halign='r', col_width=10)
            cprint([afx, *[ara[f'{i}KLD'] for i in range(1, 4)]], pos=[3, 4], halign='r', col_width=10)
            pack = [alts, nafxm['1'], nafxm['2'], nafxm['3'], nafxm['d-1'], nafxm['d-2'], nafxm['d-3']]
            cprint(pack, pos=[3, 4], halign='r', col_width=10)
            print('\n')
        else:
            for g in (('Left Window', 0, 'l'), ('Right Window', 1, 'r')):
                print(f'\t\t\t{g[0]}\n')
                pack = [['Window Size:', 'Frags:', 'Total:', 'Max Curve:', 'Curve %:', 'KLD:', 'Afx Coef:']]
                for j in range(4):
                    col = []
                    col.append(j+1)
                    col.append(len(prx[j][g[1]]))
                    col.append(prx[j][g[1]].total())
                    col.append(stats[f"{g[2]}{j+1}crv"])
                    col.append(round((1-(prx[j][g[1]].total() / (stats[f"{g[2]}{j+1}crv"] if stats[f"{g[2]}{j+1}crv"] else 1))) * 100, 3))
                    col.append(ara[f'{j}KLD-{g[2]}'])
                    col.append(round(((1+(prx[j][g[1]].total() / (stats[f"{g[2]}{j+1}crv"] if stats[f"{g[2]}{j+1}crv"] else 1)))**ara[f'{j}KLD-{g[2]}']) * ((ara[f'{j}KLD-{g[2]}']**(1+(prx[j][0].total() / (stats[f"{g[2]}{j+1}crv"] if stats[f"{g[2]}{j+1}crv"] else 1))))), 4))
                    pack.append(col)
                cprint(pack, pos=[3, 4], halign='r', col_width=10)
                print('\n')
                print('\n\t\t\tRelative Entropy\t\t\t\t  Diff\n')
                cprint(['Affix/WSize:', 1, 2, 3, 1, 2, 3], pos=[3, 4], halign='r', col_width=10)
                cprint([afx, *[ara[f'{i}KLD-{g[2]}'] for i in range(1, 4)]], pos=[3, 4], halign='r', col_width=10)
                pack = [alts, nafxm[f'{g[2]}1'], nafxm[f'{g[2]}2'], nafxm[f'{g[2]}3'], nafxm[f'd-{g[2]}1'], nafxm[f'd-{g[2]}2'], nafxm[f'd-{g[2]}3']]
                cprint(pack, pos=[3, 4], halign='r', col_width=10)
                print('\n')
        for x in [x for x in self.wlst.most_common() if afx in x[0]][:30]: print(x)

    def print_status(self, targets, affixes):
        failed, rcn, acn, wcn = [], Counter(), Counter(), Counter()
        for x in targets:
            tmp = self.find_sub_chain(x, affixes)
            if tmp:
                idx = self.merger(tmp)
                if isinstance(idx[1][0], tuple):
                    for cnt in idx[1][0]: rcn[cnt] += 1
                else:
                    rcn[idx[1][0]] += 1
                for cnt in idx[1][:-1]:
                    if isinstance(cnt, tuple):
                        for ccnt in cnt: wcn[ccnt] += 1
                    else: wcn[cnt] += 1
                for cnt in idx[0]:
                    if isinstance(cnt, tuple):
                        for ccnt in cnt: acn[ccnt] += 1
                    else: acn[cnt] += 1
                self.pretty_printer(idx)
                print('\n')
            else:
                cprint((x, 'Failed'), (1, 2))
                failed.append(x)

        cprint([targets[:100], rcn.most_common()[::-1][:100], acn.most_common()[::-1][:100], wcn.most_common()[::-1][:100], failed[::-1][:100]], [1, 3, 5, 7, 9], valign='bottom', halign='m')
        print('\n')
        cprint(['Targets', 'Roots', 'Affixes', 'Intermediate Words', 'Failed'], [1, 3, 5, 7, 9], halign='m')
        return failed

    def load(self, lid: int=0) -> None:
        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{lid}', 'rb') as f:
            self.wlst, self.afx, self.cleared, self.final, self.dsts, self.rntp, self.drntp, self.re_arr, self.full_scores = load(f)

    def save(self, lid: int=0) -> None:
        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{lid}', 'wb') as f:
            dump((self.wlst, self.afx, self.cleared, self.final, self.dsts, self.rntp, self.drntp, self.re_arr, self.full_scores), f)

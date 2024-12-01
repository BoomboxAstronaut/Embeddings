import numpy as np
from pickle import load, dump
from collections import Counter
from tqdm import tqdm

from btk import cprint


class AffixAnalyzer:

    def __init__(self, words, load_id: int=0):
        self.ldct = {
            'alpha': {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'},
            '0afx': {'s_', 'd_', 'r_', 'n_', 't_', 'y_'},
            '1afx': {'s_', 'd_', 'r_', 'n_', 't_', 'y_', '_a', 'a_', '_o', 'o_', 'i_', '_e'},
            '2afx': {
                    "_de", "_di", "_bi", "_co", "_en", "_in", "_re", "_un", 
                    "_ab", "_ad", "_be", "_ec", "_em", "_ex", "_im", "_ob", 
                    "er_", "es_", "ed_", "ic_", "al_", "ry_", "ly_", 
                    "ar_", "cy_", "ee_", "en_", "ia_", "ie_", "or_", "um_",
                    "_up", "up_", "_on", "_by"
            },
            'fdbl': {'b', 'c', 'd', 'f', 'g', 'l', 'm', 'n', 'p', 'r', 's', 't'},
            'bdbl': {'b', 'd', 'g', 'm', 'n', 'p', 'r', 't'},
            'vwl1': {'a', 'e', 'i', 'o', 'u', 'y'},
            'vwl2': {'a', 'e', 'i', 'o', 'u'},
            'vwl3': {'a', 'e', 'o', 'i', 'y'},
            'vwl4': {'a', 'e', 'o', 'u'},
            'vwl5': {'a', 'o', 'i', 'u'},
            'vwl6': {'i', 'u'},
            'vwl7': {'e', 'o', 'a'},
            'readdf': {'_ill', '_app', '_aff', '_irr', '_att', '_agg', '_opp', '_ass', '_all', '_ann', '_eff', '_acc'},
            'bridges': {'emat', 'isat', 'izat', 'ibil', 'ula', 'at', 'an', 'ar', 'ti', 'a', 'e', 'i', 'o', 'u'}
        }
        self.verif = [
            '_electr', '_econom', '_neuro', '_hydro', '_chrom', '_onto', '_onco', '_kine', '_lys', '_eco', '_bio', '_geo',
            '_super', '_supra', '_under', '_trans', '_ortho', '_intra', '_inter', '_vert', '_over', '_fore', '_sup', '_sub', '_out', '_off', '_mid',
            '_multi', '_micro', '_hyper', '_hypo', '_semi', '_poly', '_mono', '_uni', '_iso', '_lat', '_dia',
            '_counter', '_pseudo', '_para', '_meta', '_auto', '_anti', '_pro', '_pre', '_non', '_mis', '_epi', '_sym', '_con', '_com', '_ant', '_ana', '_dis',
            'cide_', 'ment_', 'logy_', 'cian_', 'less_', 'ness_', 'ance_', 'ence_', 'able_', 'ible_', 'ular_',
            'tion_', 'sion_', 'ing_', 'ism_', 'ish_', 'ist_', 'ise_', 'ize_', 'ive_', 'ium_', 'ian_', 'ile_', 
            'ate_', 'ant_', 'ent_', 'est_', 'eum_', 'ean_', 'eur_', 'our_', 'ous_', 'oid_', 'sis_', 'ful_', 
            '_opp', '_irr', '_ill', '_eff', '_att', '_ass', '_app', '_all', '_agg', '_aff', '_acc', '_ad', 
            '_ab', '_an', '_ob', '_ec', '_en', '_ex', '_em', '_in', '_im',
            'es_', 'er_', 'or_', 'ed_', 'ic_', 'al_', 'fy_', 'ty_', 'cy_', 'ly_', 'ry_', 'ia_', 'ie_', 'um_', 'en_', 'ar_', 'ee_',
            '_up', '_on', '_by', '_be', 'up_', '_un', '_re', '_di', '_de', '_co', '_bi', '_e', 'd_', 'r_', 's_', 'y_'
        ]
        self.roots = {
            '_lymph_', '_metre_', '_meter_', '_metry_', '_graph_', '_photo_', '_sume_', '_cede_', '_ceed_', '_ecto_', '_tone_', '_fish_', '_form_', 
            '_ship_', '_man_', '_men_', '_var_', '_max_', '_min_', '_lyr_', '_gress_', '_cess_', '_fess_', '_press_'
        }
        self.cterms = {x for x in self.verif if len(x) > 3}
        self.averif = {*self.ldct['2afx'], *self.ldct['1afx'], *self.verif}
        self.cleared, self.failed_brk, self.final = Counter(), Counter(), Counter()
        self.afxscore, self.wparts = dict(), []
        self.dbg = False
        if load_id:
            self.load(load_id)
        else:
            self.wlst = Counter({f'_{x[1]}_': int(x[0]) for x in words[::-1]})
            self.full_scores = self.wlst.copy()
            self.pre_clean()
            self.create_afx('w', 11, 2)
            self.post_clean('w')
            self.prep_entropy_calc()

    def create_afx(self, method='w', vmax: int=0, vmin: int=0, rmax: int=7) -> None:
        """
        Create list of affixes via 1 of 2 methods.
        w: Window method moves a window of various sizes over all words and counts occurances of affixes
        r: Remainder method for all words, find nested words and removes the inner word from all containing words and counts the remaining affixes

        w ex: '_retracting_' -> ret, etr, tra, rac, act, cti, tin, ing
        r ex: 'firm' | 'reaffirmed', 'confirming' -> reaf, ed, con, ing

        Args:
            vmax (int, optional): Maximum window size for w method || Maximum inner word length for r method. Defaults to 10 for w | 12 for r.
            vmin (int, optional): Minimum window size for w method || Minimum inner word length for r method. Defaults to 2 for w | 4 for r.
            rmin (int, optional): Minimum word length for outer words. r method only. Defaults to 7.
        """
        self.afx = Counter()
        if method == 'w':
            if not vmax: vmax = 10
            if not vmin: vmin = 2

            for word in self.wlst:
                wln = min(len(word), vmax)
                for n in range(vmin, wln-1):
                    for pos in range(wln-n+1):
                        self.afx[word[pos:pos+n]] += self.wlst[word]

        elif method == 'r':
            if not vmax: vmax = 12
            if not vmin: vmin = 4

            smalls = {x for x in self.wlst if len(x) > vmin and len(x) < vmax}
            bigs = {x for x in self.wlst if len(x) > rmax}
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
                            self.afx[z] += 1
        print('Frags Created', len(self.afx))

    def pre_clean(self) -> None:
        #Remove ' words from word list
        for x in ["_ain't_", "_can't_", "_won't_", "_shan't_"]:
            self.final["n't_"] += self.wlst[x]
        for x in ["_i'm_", "_can't_"]:
            self.wlst[f'{x[:-3]}_'] += self.wlst[x]
        for x in ["_ma'am_", "_ain't_", "_i'm_"]:
            self.final[x] += self.wlst[x]
        for x in ["_ain't_", "_can't_", "_won't_", "_shan't_", "_ma'am_", "_i'm_", "_van't_"]:
            self.wlst.pop(x)
        for x in [z for z in self.wlst if "'" in z if any(y in z for y in ["'s_", "'ll_", "'ve_", "n't_", "'re_", "'d_"])]:
            for efx in ["'s_", "'ll_", "'ve_", "n't_", "'re_", "'d_"]:
                if efx in x:
                    self.final[efx] += self.wlst[x]
                    self.wlst[x.replace(efx, '_')] += self.wlst[x]
                    self.wlst.pop(x)
        for x in [z for z in self.wlst if "'" in z]:
            self.wlst.pop(x)

    def post_clean(self, regime: str, cdist: int=0, cmin: int=0) -> None:
        """
        Clean affix list, remove noise and words that will be tokenized as wholes
        
        Args:
            cdist (int, optional): Distance to search from a sorted affix list for nested affix cleaning. Defaults to 2048 for w | 512 for r.
            cmin (int, optional): Minimum occurrences to keep an affix in the list. Defaults to 64 for w | 8 for r.
        """
        if regime == 'w':
            if not cdist: cdist = 2048
            if not cmin: cmin = 32
        elif regime == 'c':
            if not cdist: cdist = 512
            if not cmin: cmin = 8

        for x in self.wlst.most_common():
            if x[1] > 3000000: self.cleared[x[0]] = x[1] #Words occurance > 3M
            elif x[1] > 100000 and len(x[0]) < 6: self.cleared[x[0]] = x[1] #Words, < 4 chars, occurance > 100k
            elif x[1] < 100000: break
        for z in [y for y in self.wlst if len(y) < 5]: #Words 2 letters or less
            if self.wlst[z] > cmin: self.cleared[z] += self.wlst[z] 
            else: self.wlst.pop(z)
        for z in self.cleared: #Remove cleared words
            if z in self.wlst: self.wlst.pop(z)

        #Scan from most common to least, if a nested affix is found within 2048 items, subtract that items value from the current affix
        afidx = [x[0] for x in self.afx.most_common() if '_' in x[0]]
        for i, x in enumerate(afidx):
            group = [y for y in afidx[i+1:i+1+cdist] if x in y]
            if group: self.afx[x] -= self.afx[group[0]]
        self.afx = Counter({x[0]: x[1] for x in self.afx.most_common() if x[1] > cmin})

        for z in [x for x in self.afx if '_' not in x and len(x) < 3]:
            self.afx.pop(z) #Unattached affixes < 3 chars
        for z in [x for x in self.afx if not any(y in x for y in self.ldct['vwl1']) and x not in self.ldct['1afx'] and x not in self.ldct['2afx']]:
            self.afx.pop(z) #Affixes with no vowels
        for z in [x for x in self.afx if len(x) < 4 and '_' in x and x not in self.averif]:
            self.afx.pop(z) #Affixes < 3 chars that arent in pre verified list
        for z in {x[0] for x in [(x, f'_{x}', f'{x}_', f'_{x}_') for x in self.afx if '_' not in x] if (x[1] in self.afx or x[2] in self.afx or x[3] in self.cleared or x[3] in self.final or x[3] in self.wlst)}:
            self.afx.pop(z) #Unattached affixes with that have an attached variant
        for z in tqdm([x for x in self.afx]):
            c = 0
            for y in self.wlst:
                if z in y: c += 1
                if c > 3: break
            else:  self.afx.pop(z)

        for x in ('less_', 'ness_'): self.target_removal(x, exe=True)
        self.target_removal('es_', exc1=('es_', 's_'), exc2=('is_', 'us_', 'ss_', 'series_', 'species_'), exe=True)
        self.target_removal('s_', exc1=('es_', 's_'), exc2=('is_', 'us_', 'ss_', 'series_', 'species_'), exe=True)
        self.target_removal('er_', exc2=('meter_', 'over_', 'under_', 'master_'), exe=True)
        self.target_removal('or_', exc2=('oor_'), exe=True)
        self.target_removal('ed_', exc2=('eed_'), exe=True)
        for x in ('en_', 'ly_', 'ion_', 'ous', 'ing_', 'ity_', 'ize_', 'ise_', 'ive_', 'ist_', 'ism_', 'ory_', 'est_', 'ment_', 'ant_', 'ary_', 'ate_', 'ic_', 'al_'): self.target_removal(x, exe=True)
        self.target_removal('y_', exc2=('ity_', 'ry_', 'ly_', 'ory_', 'ary_'), exe=True)

    def get_compounds(self):
        combos = {}
        for x in self.wlst:
            front, back = [], []
            for y in self.wlst:
                if y != x:
                    f, b = y[1:], y[:-1]
                    if f in x and (len(x) - x.index(f) - len(f)) == 0:
                        front.append(f)
                    if b in x and x.index(b) == 0:
                        back.append(b)
            veri = []
            if front and back:
                for m in front:
                    for n in back:
                        if len(m) + len(n) == len(x):
                            veri.append((m, n))
            if veri:
                combos[x] = tuple(veri)
        return combos

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

    def is_sub(self, orig, rep):
        orig, rep = orig.strip('_'), rep.strip('_')
        rslt = [x.split(rep) for x in self.search(rep) if x != f'_{rep}_']
        orslt = [x.split(orig) for x in self.search(orig) if x != f'_{orig}_']
        if len(rslt) > 3 and len(orslt) > 3:
            if len([x for x in orslt if x in rslt]) > 1: return False
        out = {y for x in rslt for y in x if y != '_' and y in self.verif}
        if (rslt and out) and (len(out) > 7 or len(out) / len(rslt) >= 0.5 or (len(out) / len(rslt) >= 0.15 and len(out) > 2)): return True
        else: return False

    def target_removal(self, afx: str, exc1: str|tuple[str]=None, exc2: str|tuple[str]=None, exe: bool=False) -> list[str] | None:
        """
        Find affixes that contain the input affix and runs the gsub method on them. If the 'exe' paremeter is set to true, found affixes will be removed.

        Args:
            afx (str): Affix to search/remove.
            exc1 (str | tuple[str], optional): Exact affixes to exclude from removal. Defaults to None.
            exc2 (str | tuple[str], optional): Affixes to filter for affixes to exclude from removal. Defaults to None.
            exe (bool, optional): Remove found affixes. Defaults to False.

        Returns:
            list (str): List of substitutions for target affix
        """
        if exc1 and exc2:
            if isinstance(exc1, str) and isinstance(exc2, str): targets = [x for x in self.afx if afx in x and x not in (afx, exc1) and exc2 not in x]
            elif isinstance(exc1, str): targets = [x for x in self.afx if afx in x and x not in (afx, exc1) and all(y not in x for y in exc2)]
            elif isinstance(exc2, str): targets = [x for x in self.afx if afx in x and x not in (afx, *exc1) and exc2 not in x]
            else: targets = [x for x in self.afx if afx in x and x not in (afx, *exc1) and all(y not in x for y in exc2)]
        elif exc1:
            if isinstance(exc1, str): targets = [x for x in self.afx if afx in x and x not in (afx, exc1)]
            else: targets = [x for x in self.afx if afx in x and x not in (afx, *exc1)]
        elif exc2:
            if isinstance(exc2, str): targets = [x for x in self.afx if afx in x and x != afx and exc2 not in x]
            else: targets = [x for x in self.afx if afx in x and x != afx and all(y not in x for y in exc2)]
        else: targets = [x for x in self.afx if afx in x and x != afx]

        if exe:
            for x in targets:
                tmp = self.gsub(x, afx, amode=1)
                if tmp: 
                    self.afx[tmp] += self.afx[x]
                    self.afx.pop(x)
        else: return [res for x in targets if (res := self.gsub(x, afx, amode=1))]

    def pulld(self, afx: str, len_lim: int=0) -> list[str]:
        """
        Finds all child nodes of the input affix

        Args:
            afx (str): Input affix
            len_lim (int, optional): Maximum difference in length between input and output nodes. 0 permits any difference. Defaults to 0.

        Returns:
            list[str]: List of all child nodes
        """
        sub_set = [x for x in self.afx if afx in x and x != afx]
        if '_' not in afx: sub_set = [x for x in sub_set if '_' not in x]
        out, aln = set(), len(afx)
        for x in sub_set:
            i = 1
            if '_' not in x:
                ti, wl = x.index(afx), len(x)
                pf = wl-(ti+aln)
                while i <= max(pf, ti):
                    if x[ti-i:ti+aln] in sub_set and x != x[ti-i:ti+aln]: break
                    elif x[ti:ti+aln+i] in sub_set and x != x[ti:ti+aln+i]: break
                    i += 1
                else: out.add(x)
            elif x[0] == '_':
                while len(x[:-i]) > aln:
                    if x[:-i] in sub_set: break
                    i += 1
                else: out.add(x)
            else:
                while len(x[i:]) > aln:
                    if x[i:] in sub_set: break
                    i += 1
                else: out.add(x)
        if not len_lim: return out
        else: return [x for x in out if len(x) <= aln+len_lim]

    def pullu(self, afx: str) -> str:
        #Return the parent node of the input affix
        i = 1
        if '_' not in afx:
            while i < len(afx):
                hold = []
                if afx[:-i] in self.afx: hold.append(afx[:-1])
                elif afx[i:] in self.afx: hold.append(afx[i:])
                if len(hold) > 1: return hold
                elif hold: return hold[0]
                i += 1
        elif afx[0] == '_':
            while i < len(afx):
                if afx[:-i] in self.afx: return afx[:-i]
                i += 1
        else:
            while i < len(afx):
                if afx[i:] in self.afx: return afx[i:]
                i += 1

    def chain(self, afx: str) -> list[str]:
        #Return all nodes along the longest possible path that contains this affix node
        out = sorted([x for x in self.afx if afx in x or x in afx], key=lambda x: len(x))
        if out:
            out = out[-1]
            if '_' in afx: return sorted([x for x in self.afx if x in out and '_' in x], key=lambda x: len(x), reverse=True)
            else: return sorted([x for x in self.afx if x in out and '_' not in x], key=lambda x: len(x), reverse=True)

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
            for w in bkd:
                if not bkd[w]['chk']:
                    for nx in [x for x in self.verif if x in w]:
                        rep = w.replace(nx, '_')
                        bkd[w]['afx'].append(nx)
                        bkd[w]['reps'].append(rep)
                        if rep not in bkd: new.append((rep, bkd[w]['afx'].copy(), bkd[w]['reps'].copy(), bkd[w]['ub'].copy()))
                    else: bkd[w]['chk'] = True
                elif bridges and not bkd[w]['bchk']:
                    b1 = [*[(w.replace(f'_{b}', '_'), f'_{b}') for b in self.ldct['bridges'] if w.startswith(f'_{b}')], 
                        *[(w.replace(f'{b}_', '_'), f'{b}_') for b in self.ldct['bridges'] if w.endswith(f'{b}_')]]
                    b2 = [b for b in b1 if any(bz in b[0] for bz in self.verif)]
                    for b in b2:
                        if b[0] not in new and b[0] not in bkd:
                            obr = [q for q in bkd[w]['ub']]
                            new.append((b[0], [z for z in bkd[w]['afx']], [z for z in bkd[w]['reps']], [b[1], *obr]))
                    bkd[w]['bchk'] = True
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
                print(f'\n\t\t\tRelative Entropy\t\t\t\t  Diff\n')
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
                idx = merger(tmp)
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
                pretty_printer(idx)
                print('\n')
            else:
                cprint((x, 'Failed'), (1, 2))
                failed.append(x)

        cprint([targets[:100], rcn.most_common()[::-1][:100], acn.most_common()[::-1][:100], wcn.most_common()[::-1][:100], failed[::-1][:100]], [1, 3, 5, 7, 9], valign='bottom', halign='m')
        print('\n')
        cprint(['Targets', 'Roots', 'Affixes', 'Intermediate Words', 'Failed'], [1, 3, 5, 7, 9], halign='m')
        return failed


    def load(self, id: int=0) -> None:
        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{id}', 'rb') as f:
            self.wlst, self.afx, self.cleared, self.final, self.dsts, self.rntp, self.drntp, self.re_arr, self.full_scores = load(f)

    def save(self, id: int=0) -> None:
        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{id}', 'wb') as f:
            dump((self.wlst, self.afx, self.cleared, self.final, self.dsts, self.rntp, self.drntp, self.re_arr, self.full_scores), f)


def pretty_printer(result):
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

def merger(results):
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

def packer(results, idx, cid=None):
    print(results)
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



def main():
    """
    a.wlst: Full word list. Eventually will be only roots
    wrd_q: List of words that pass minimum length and has a common affix
    bdowns: Breakdowns of words
    a.failed_brk: Non english words
    rwrd: Words incorrectly broken down
    roots: roots of words that can no longer be broken down
    """

    with open(r'D:\dstore\nlp\w2v\fwords', 'rt') as f:
        a = AffixAnalyzer([x.strip().split() for x in f.readlines()], 3)

    #ld = input('\t\tEnter "new" or Load ID\n')
    ld = '2'
    if ld == 'new':
        for x in [x for x in a.cleared.most_common() if len(x[0]) > 3 and x[1] < 20500000 and x[1] > 100000]: a.wlst[x[0]] = x[1]
        with open(r'D:\dstore\nlp\w2v\common_neng', 'rt') as f:
            neng = [f'_{x.strip()}_' for x in f.readlines()]
        for x in neng:
            if x in a.wlst: a.wlst.pop(x)
        a.afxscore, a.wparts, a.failed_brk = Counter(), dict(), []
        wrd_q = [x for x in a.wlst if len(x) > 6 and any(y in x for y in a.verif)]
    elif ld:
        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\manu{int(ld)}', 'rb') as f:
            wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk = load(f)
        print(f'\tLoaded {ld}:\n\t\tProgress: {len(wrd_q)} / {len(a.wlst)}\n\t\t{round(100*len(wrd_q)/len(a.wlst), 4)}%')
    else:
        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\manu_auto', 'rb') as f:
            wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk = load(f)
        print(f'Loaded AutoSave:\n\tProgress: {len(wrd_q)} / {len(a.wlst)}\n\t\t{round(100*len(wrd_q)/len(a.wlst), 4)}%')

    oln = len(a.full_scores)
    wrd_q = sorted(wrd_q, key=lambda x: len(x))
    a.verif = sorted(a.verif, key=lambda x: len(x), reverse=True)

    #Menu
    while True:
        #inp0 = input('\tSelect Mode:\n\t\t1: Word by Word\n\t\t2: Word Group\n')
        inp0 = '3'
        match inp0:
            #Word Separation Module
            case '1':
                while wrd_q:
                    word = wrd_q.pop()
                    res = a.find_sub_chain(word)
                    is_parsed = False
                    auto_save_ticker = 0

                    if res:
                        idx = merger(res)
                        print(idx)
                        pretty_printer(idx)
                        inp = input()

                        if inp:
                            match int(inp[0]):
                                case 0:
                                    #Reject cleaner result
                                    a.failed_brk.append(word)
                                case 1:
                                    #Select result to accept
                                    pass
                                case 2:
                                    #Select result group / root word 
                                    s_pack = packer(res, idx[1], inp)
                                    if not s_pack:
                                        print('invalid selection')
                                        wrd_q.append(word)
                                    else: is_parsed = True
                                case 3:
                                    #Substitution fix process, For single affix substitutions
                                    #Input an affix that will split a word into a root and affix that will then be added to the affix list
                                    pass

                                case 4:
                                    #Begin root process
                                    a.roots.append(word)
                                case 5:
                                    #Delete word (Non Words)
                                    a.failed_brk.append(word)
                                    a.wlst.pop(word)
                                case 6:
                                    #Undo last selection
                                    wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk = last

                                case 7:
                                    #Exit
                                    wrd_q.append(word)
                                    with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\manu_auto', 'wb') as f:
                                        dump((wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk), f)
                                    break
                                case 8:
                                    #Status Check
                                    wrd_q.append(word)
                                    print(len(wrd_q), oln, f'   {round(len(wrd_q)/oln, 4) * 100} %')
                                case 9:
                                    #Save
                                    wrd_q.append(word)
                                    print('Save ID?')
                                    select = int(input())
                                    with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\manu{select}', 'wb') as f:
                                        dump((wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk), f)
                                case _: wrd_q.append(word)

                        if not inp:
                            s_pack = packer(res, idx[1])
                            if not s_pack:
                                print('invalid selection')
                                wrd_q.append(word)
                            else: is_parsed = True
                        if is_parsed:
                            auto_save_ticker += 1
                            for i, x in enumerate(s_pack.items()):
                                if i == 0: print(f'accepting {x[1][0]} {[y for y in s_pack]}')
                                a.score_eval(x[0], x[1])
                                if x[0] in wrd_q: wrd_q.remove(x[0])
                            print('\n\n')
                            if auto_save_ticker >= 12:
                                with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\manu_auto', 'wb') as f:
                                    dump((wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk), f)
                                auto_save_ticker = 0
                        last = (wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk)

            #Group Breaker
            case '2':
                while True:
                    for x in wrd_q[::-1][:100][::-1]: print(x)
                    targets, tmp_roots, exclusions, tmp_affixes  = [], [], [], a.verif.copy()

                    while True:
                        action = input('\tInput?\n').split()
                        match action[0]:
                            #Add exclusion filter string
                            case 'rt':
                                core_w = action[1]
                                targets = a.search(core_w, wrd_q)
                                failed = a.print_status(targets, tmp_affixes)
                            case 'aa':
                                tmp_affixes.append(action[1])
                            case 'ra':
                                if action[1] in tmp_affixes:
                                    tmp_affixes.remove(action[1])
                                else: print('invalid input')
                            case 'ar':
                                tmp_roots.append(action[1])
                            case 'rr':
                                if action[1] in tmp_roots:
                                    tmp_roots.remove(action[1])
                                else: print('invalid input')
                            case 'ae':
                                exclusions.append(action[1])
                                targets = a.search(core_w, wrd_q, exc=exclusions)
                            case 're':
                                if action[1] in exclusions:
                                    exclusions.remove(action[1])
                                    targets = a.search(core_w, wrd_q, exc=exclusions)
                                else: print('invalid input')
                            case 'p':
                                failed = a.print_status(targets, tmp_affixes)

                            case 'src':
                                if len(action) > 3:
                                    for x in (a.search(action[2], a.afx, exc=action[3]) if action[1] == 'a' else a.search(action[1], exc=action[3])): print(x)
                                else:
                                    for x in (a.search(action[2], a.afx) if action[1] == 'a' else a.search(action[1])): print(x)
                            case 'top':
                                for x in wrd_q[::-1][:100][::-1]: print(x)
                            case 'info':
                                cprint((targets, tmp_affixes[::-1], tmp_roots[::-1], exclusions[::-1]), (1, 3, 5), valign='bottom')
                                cprint(('Targets', 'Affixes', 'Roots', 'Exclusions'), (1, 3, 5))
                            case 'fail':
                                for x in failed: print(x)

                            case 'lys':
                                a.aflys(action[1], True)
                            case 'dd':
                                for x in targets:
                                    tmp = a.find_sub_chain(x, tmp_affixes)
                                    if tmp:
                                        idx = merger(tmp)
                                        found = False
                                        for q in idx:
                                            for qq in q:
                                                if isinstance(qq, tuple):
                                                    for qqq in qq:
                                                        if action[1] in qqq:
                                                            pretty_printer(idx)
                                                            print('\n')
                                                            found = True
                                                else:
                                                    if action[1] in qq:
                                                        pretty_printer(idx)
                                                        print('\n')
                                                        found = True
                                                if found: break
                                            if found: break
                            case 'try':
                                idx = merger(a.find_sub_chain(action[1], tmp_affixes))
                                if idx: pretty_printer(idx)
                                else: print('\t\tFailed', action[1])
                            case 'rept':
                                tmp = [x for x in wrd_q if action[1] in x]
                                veric = Counter()
                                lab = []
                                if tmp:
                                    for x in tmp:
                                        go = a.gsub(x, action[1])
                                        if go:
                                            veric['pass'] += 1
                                            veric[go] += 1
                                            lab.append(go)
                                        else:
                                            veric['fail'] += 1
                                            lab.append('fail')
                                    cprint([tmp, lab, veric.most_common()[::-1]], [2, 4, 6], valign='bottom')

                            case '?':
                                for x in zip(['aa', 'ra', 'ar', 'rr', 'ae', 're', 'fail', 'info', 'lys', 'save', 'ship', 'exit'], [
                                    'Add an affix to the working list',
                                    'Remove an affix from the working list',
                                    'Add a root to the working list',
                                    'Remove a root from the working list',
                                    'Add an exclusion to the working list',
                                    'Remove an exclusion from the working list',
                                    'Show failed words currently in targets',
                                    'Show current affixes, roots, and exclusions'
                                    'Show analysis data for a specific affix'
                                    'Save a file to an ID',
                                    'Break down words with current affixes, add affixes, remove words, combime points',
                                    'Exit: with autosave']):
                                    cprint((f'{x[0]}:', x[1]), (1, 2))
                            case 'save':
                                with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\manu{action[1]}', 'wb') as f:
                                    dump((wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk), f)
                            case 'ship':
                                for x in tmp_roots:
                                    a.roots.append(x)
                                for x in tmp_affixes:
                                    if x not in a.verif:
                                        a.verif.append(x)
                                targets = a.search(core_w, wrd_q, exc=exclusions)
                                for x in targets:
                                    chain = a.find_sub_chain(x, tmp_affixes)
                                    if chain:
                                        s_pack = packer(chain, merger(chain)[1])
                                        for i, x in enumerate(s_pack.items()):
                                            if x[0] in wrd_q:
                                                if i == 0: print(f'accepting {x[1][0]} {[y for y in s_pack]}')
                                                a.score_eval(x[0], x[1])
                                                wrd_q.remove(x[0])
                                with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\manu_auto', 'wb') as f:
                                    dump((wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk), f)
                                break
                            case 'exit':
                                break

                        if action[0] in ('aa', 'ra', 'rr', 'ar', 'ae', 're'):
                            failed = a.print_status(targets, tmp_affixes)

            #Affix Isolater
            case '3':
                sel = input('\n\t\tpre or suf?\n').split()
                if sel[0] == 'pre':
                    pre = True
                    with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\ppd2', 'rb') as f:
                        tid, tdct = load(f)
                    with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\fsort', 'rt') as f:
                        corpus = [x.strip() for x in f.readlines()]
                    code = 'pafxrts'
                    if len(sel) > 1 and sel[1] == 'l':
                        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\safxrts', 'rb') as f:
                            afx_hold, root_hold, index = load(f)
                        root_hold.extend(afx_hold)
                        ofound = root_hold.copy()
                        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{code}', 'rb') as f:
                            afx_hold, root_hold, index = load(f)
                    else: afx_hold, root_hold, index = [], [], 0
                elif sel[0] == 'suf':
                    pre = False
                    with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\spd2', 'rb') as f:
                        tid, tdct = load(f)
                    with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\bsort', 'rt') as f:
                        corpus = [x.strip() for x in f.readlines()]
                    code = 'safxrts'
                    if len(sel) > 1 and sel[1] == 'l':
                        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\pafxrts', 'rb') as f:
                            afx_hold, root_hold, index = load(f)
                        root_hold.extend(afx_hold)
                        ofound = [x for x in root_hold]
                        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{code}', 'rb') as f:
                            afx_hold, root_hold, index = load(f)
                    else: afx_hold, root_hold, index = [], [], 0

                def agprinter(idx):
                    if not pre: harg = 'r'
                    else: harg = 'l'
                    labels = tid[idx:idx+8]
                    cprint(
                        [tid[idx:idx+33][::-1], [tdct[x][0] for x in tid[idx:idx+33]][::-1], *[corpus[tdct[l][1]:tdct[l][2]] for l in labels]],
                        [2, 5, 9, 14, 19, 24, 29, 34, 39, 44], 
                        halign=harg, 
                        valign='bottom', 
                        col_width=4, 
                        trim=True
                    )
                    print('\n')
                    cprint(
                        [f'{l} {tdct[l][0]}' for l in labels], 
                        [9, 14, 19, 24, 29, 34, 39, 44, 49], 
                        col_width=4, 
                        halign=harg,
                        valign='bottom'
                        )
                    print('\n')

                agprinter(index)
                acnt = 0
                while index < len(tid)-1:

                    action = input(f'\n\t\tIndex: {index}\n\t\tTotal: {len(tid)}\n\t\tAwaiting Input...\n').split()
                    if action:
                        if action[0][0] in ('+', '-'): action = [action[0][0], int(action[0][1:])]
                        match action[0]:
                            case 't':
                                if action[1] not in tid:
                                    print('Affix not Found')
                                else:
                                    subi = tid.index(action[1])
                                    labels = [tid[i] for i in range((subi-3 if subi > 2 else 0), (subi+4 if subi+4 < len(tid) else len(tid)-1))]
                                    if pre:
                                        cprint([corpus[tdct[l][1]:tdct[l][2]] for l in labels], [i for i in range(1, 7)], valign='bottom', col_width=24)
                                    else:
                                        cprint([corpus[tdct[l][1]:tdct[l][2]] for l in labels], [i for i in range(1, 7)], valign='bottom', halign='r', col_width=24)
                                    print('\n')
                                    if pre:
                                        cprint([f'{l} {tdct[l][0]}' for l in labels], [i for i in range(1, 7)], col_width=24)
                                    else:
                                        cprint([f'{l} {tdct[l][0]}' for l in labels], [i for i in range(1, 7)], col_width=24, halign='r')
                                    print('\n')

                            case 'aa':
                                if action[1] in tid:
                                    index = tid.index(action[1]) + 1
                                    if pre: afx_hold.append(f'_{action[1]}')
                                    else: afx_hold.append(f'{action[1]}_')
                                    if index < len(tid)-1: agprinter(index)
                                else:
                                    if pre: afx_hold.append(f'_{action[1]}')
                                    else: afx_hold.append(f'{action[1]}_')
                                acnt += 1

                            case 'ra':
                                if pre:
                                    if f'_{action[1]}' in afx_hold:
                                        afx_hold.remove(f'_{action[1]}')
                                    else: print(f'Invalid Affix {action[1]}')
                                else:
                                    if f'{action[1]}_' in afx_hold:
                                        afx_hold.remove(f'{action[1]}_')
                                    else: print(f'Invalid Affix {action[1]}')

                            case 'ar':
                                if action[1] in tid:
                                    index = tid.index(action[1]) + 1
                                    if index < len(tid)-1: agprinter(index)
                                root_hold.append(action[1])
                                acnt += 1

                            case 'rr':
                                if action[1] in root_hold:
                                    root_hold.remove(action[1])
                                else: print(f'Invalid Root {action[1]}')

                            case '+':
                                index += (action[1] if action[1]+index < len(tid)-1 else 0)
                                agprinter(index)
                            case '-':
                                index -= (action[1] if index - action[1] >= 0 else index)
                                agprinter(index)

                            case 'ls':
                                cprint([tid[index:index+33], [tdct[x][0] for x in tid[index:index+33]], afx_hold[-33:], root_hold[-33:]], [2, 3, 5, 7], valign='bottom')
                                print('\n')
                                cprint(['Targets', 'Affixes', 'Roots'], [2, 5, 7])

                            case 'src':
                                own = [*[x for x in root_hold], [x for x in afx_hold]]
                                if len(action) > 2:
                                    found_wl = a.search(action[1], exc=action[2])
                                    found_afl = a.search(action[1], a.afx, exc=action[2])
                                    found_p = a.search(action[1], own, exc=action[2], pos=False)
                                    found_o = a.search(action[1], ofound, exc=action[2], pos=False)
                                else:
                                    found_wl = a.search(action[1])
                                    found_afl = a.search(action[1], a.afx)
                                    found_p = a.search(action[1], own, pos=False)
                                    found_o = a.search(action[1], ofound, pos=False)
                                cprint([found_wl, found_afl, found_p, found_o], [2, 4, 6, 8], valign='bottom')
                                print('\n')
                                cprint(['Word List', 'Affixes', 'Self Side', 'Alt Side'], [2, 4, 6, 8])

                            case 'lys':
                                a.aflys(action[1], True)

                            case 'save':
                                with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{code}', 'wb') as f:
                                    dump((afx_hold, root_hold, index), f)
 
                    else:
                        index += (8 if index+8 < len(tid)-1 else (len(tid)-2)-index)
                        agprinter(index)

                    if acnt >= 11:
                        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{code}-auto', 'wb') as f:
                            dump((afx_hold, root_hold, index), f)
                        acnt = 0

                with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{code}', 'wb') as f:
                    dump((afx_hold, root_hold, index), f)

            case '7':
                with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\manu_auto', 'wb') as f:
                    dump((wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk), f)

if __name__ == "__main__":
    main()

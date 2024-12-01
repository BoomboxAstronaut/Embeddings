
import numpy as np
from collections import Counter
from tqdm import tqdm
from typing import Container
from btk import lrsort, rrsort
from pickle import dump, load

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
        self.roots = [
            '_lymph_', '_metre_', '_meter_', '_metry_', '_graph_', '_photo_', '_sume_', '_cede_', '_ceed_', '_ecto_', '_tone_', '_fish_', '_form_', 
            '_ship_', '_man_', '_men_', '_var_', '_max_', '_min_', '_lyr_', '_gress_', '_cess_', '_fess_', '_press_'
        ]
        self.cterms = {x for x in self.verif if len(x) > 3}
        self.averif = {*self.ldct['2afx'], *self.ldct['1afx'], *self.verif}
        self.cleared, self.failed_brk, self.final = Counter(), Counter(), Counter()
        self.afxscore, self.wparts = dict(), []
        self.dbg = False
        if load_id:
            self.load(load_id)
            self.bare = {x.strip('_') for x in self.full_scores}
            self.default_search = self.full_scores
        else:
            self.wlst = Counter({f'_{x[1]}_': int(x[0]) for x in words[::-1]})
            self.full_scores = self.wlst.copy()
            self.default_search = self.full_scores
            self.bare = {x.strip('_') for x in self.full_scores}
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

    def _prep_afx(self) -> None:
        with open(r'D:\dstore\nlp\w2v\directions', 'rb') as f:
            directions = load(f)
        self.afx['_a'] += self.wlst["_around_"]
        self.afx["_round"] += self.wlst["_around_"]
        self.afx["_o"] += self.wlst["_over_"]
        self.afx["_ver"] += self.wlst["_over_"]
        for x in directions[0]:
            self.afx[x[:-1]] += self.wlst[x]
        for x in directions[2]:
            self.afx[x[:-1]] += self.wlst[x]
            self.afx[x[1:]] += self.wlst[x]
        for x in directions[1]:
            self.afx[x[1:]] += self.wlst[x]
        for y in directions:
            for x in y:
                if len(x) < 5:
                    self.final[x] += self.wlst[x]
                    self.wlst.pop(x)
                else:
                    self.cleared[x] += self.wlst[x]
                    self.wlst.pop(x)

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

    def search(self, term: str, corpus: Container=None, exc: str|tuple[str]=None, pos: bool=False, sfil=False, svar='i'):
        #Returns all items that contain the input affix
        if not corpus: corpus = self.default_search
        if svar == 's': res = [x.strip() for x in lrsort([x for x in corpus if x.startswith(term)])]
        elif svar == 'e': res = [x.strip() for x in rrsort([x for x in corpus if x.endswith(term)])]
        else: res = sorted([x for x in corpus if term in x])

        if exc: res = [x for x in res if all(y not in x for y in ((exc,) if isinstance(exc, str) else exc))]
        if sfil: res = [x for x in res if x not in {f'_{term.strip("_")}', f'{term.strip("_")}_', f'_{term.strip("_")}_', term}]
        if pos: res = [x for x in res if '_' in x]
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
        spafx = {'sion_', 'ian_', 'es_', 'cy_', 's_', 'y_'}
        rep = target.replace(afx, '')
        candidates = [target.replace(afx, '_')]
        if afx[0] == '_': pre = True
        else: pre = False

        if afx in spafx: #Specific Affix Substitution Rules
            if afx == 'sion_':
                if rep.endswith('is'): candidates.append(f'{rep[:-1]}t_')
                elif rep[-1] == 'n': candidates.append(f'{rep[:-1]}d_')
                elif rep[-1] == 'r': candidates.append(f'{rep}t_')
                elif rep[-1] in self.ldct['vwl2']:
                    candidates.append(f'{rep}de_')
                    candidates.append(f'{rep}re_')
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
            if len(afx) == 2:
                if f'_{rep}' in self.full_scores: return f'_{rep}'
                else: return

            if afx[-1] not in self.ldct['vwl2']: candidates.append(f'_{afx[-1]}{rep}')
            #if rep[0] in self.ldct['vwl2']: candidates.append(f'_{rep[1:]}')
            else:
                if len(rep) > 4 and rep[0] == rep[1] and rep[0] in self.ldct['fdbl']:
                    candidates.append(f'_{rep[1:]}')

        else:
            if len(afx) == 2:
                if f'{rep}_' in self.full_scores: return f'{rep}_'
            elif len(afx) > 2: candidates.append(f'{rep}e_')

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
            return sorted([x for x in self.afx if x in out and '_' not in x], key=lambda x: len(x), reverse=True)

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
        else: return Counter({x[0]: x[1] for x in left_cnt.items() if len(x[0]) == window}), Counter({x[0]: x[1] for x in right_cnt.items() if len(x[0]) == window})

    def remean(self, rearr: np.ndarray) -> np.ndarray:
        #Returns the mean array of the input array for each column and row
        return np.array([*[np.mean(y) for y in rearr], *[np.mean(y) for y in rearr.T]])

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
                #Clean duplicate targets
                for x in targets[::-1]:
                    if word == x[0] and found_afxs == x[1]: targets.remove(x)
                if len(word) < 7: break
            #Once no more affixes are found in a word add it to the outputs if atleast 1 matcheed
            #If word is still long and no match, use double sub
            if is_match: fout.add((tuple(rafxs), (*rem, word)))
            elif len(word) > 7 and dsub:
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

    def e2gsub(self, word, it_mx=2, bridges=False):
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
        kl = [(k, 50000) if k in self.roots else (k, self.full_scores[k]) for k in bkd.keys() if k in self.full_scores]
        if kl:
            ok = sorted(kl, key=lambda x: x[1])[-1][0]
            if not bridges: return (ok, bkd[ok]['reps'][:-1], bkd[ok]['afx'][:-1])
            else: return (ok, bkd[ok]['reps'][:-1], bkd[ok]['afx'][:-1], bkd[ok]['ub'])

    def assign_search_dict(self, words: Container):
        self.default_search = words

    def load(self, lid: int=0) -> None:
        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\saves\\{lid}', 'rb') as f:
            self.wlst, self.afx, self.cleared, self.final, self.dsts, self.rntp, self.drntp, self.re_arr, self.full_scores = load(f)

    def save(self, lid: int=0) -> None:
        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{lid}', 'wb') as f:
            dump((self.wlst, self.afx, self.cleared, self.final, self.dsts, self.rntp, self.drntp, self.re_arr, self.full_scores), f)

def setup():
    with open(r'D:\dstore\nlp\w2v\fwords', 'rt') as f:
        a = AffixAnalyzer([x.strip().split() for x in f.readlines()], 3)
    a.bare = set(sorted([x.split()[0].strip('_') for x in a.wlst], key=lambda x: (len(x), x))[::-1])
    return a
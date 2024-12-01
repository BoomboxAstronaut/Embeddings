
from .Word2 import Word
from pickle import load
from os import rename
from time import time

class Lexicon:
    """
    Class for dissecting a dictionary into elementary components

    bases: Initial list of dictionary words
    media: All words and subcomponents
    files: Files of manually categorized words, exceptions, and decomposition commands
    wfiles: Files for a mostly automated decomposition process
    affixes: Prefixes, suffixes, and infixes
    rgen: Object for interacting with primary files

    """
    dbl = 'bdglmnprt'
    pos = 'nvadreci'
    primes = 'us um os is er a e on as o y io'.split()
    lgs = ('', 'e', 'a', 'o', 'y', 'us', 'is', 'os', 'um', 'on', 'as', 'eo')

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, k):
        if k in self.media: return self.media[k]
        else: raise ValueError(f'{k} not in tracker')

    def __setitem__(self, k, v):
        self.media[k] = v

    def __repr__(self):
        return self.media.__repr__()

    def __init__(self, wlst: set[str], wmode=False):
        self.files = {x[0]: f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\{x[1]}' for x in
            {
                'adjectives': r'v1\adjectives',
                'adverbs': r'v1\adverbs',
                'affixes': r'v0\affixes\_affixes',
                'afx_groups': r'v0\affixes\_afx_groups',
                'afx_prex': r'v0\affixes\_afx_prex',
                'astro': r'v1\astron',
                'b_add': r'v0\roots\_base_add',
                'b_remove': r'v0\roots\_base_remove',
                'bio': r'v1\bio',
                'biota': r'v1\biota',
                'block': r'v0\roots\block',
                'chems0': r'v1\chems0',
                'chems1': r'v1\chems1',
                'conjunctions': r'v1\conjunctions',
                'cpu': r'v1\cpu',
                'determiners': r'v1\determiners',
                'cmls': r'v0\roots\_cmls4',
                'food': r'v1\food',
                'ign_a': r'v0\misc\_adj_ignore',
                'ign_m': r'v0\misc\_member_ignore',
                'ign_pa': r'v0\misc\_past_ignore',
                'ign_pf': r'v0\affixes\_prefix_ignore',
                'ign_pl': r'v0\misc\_plural_ignore',
                'ign_pr': r'v0\misc\_preprog_ignore',
                'ingredients': r'v1\ingredients',
                'interjections': r'v1\interjections',
                'latin': r'v1\latin',
                'latgre': r'v0\misc\latgre',
                'man_1': r'v0\roots\old\_pre_manual',
                'man_2': r'v0\roots\old\_manual',
                'master': r'v1\master_list',
                'measures': r'v1\measures',
                'medicine': r'v1\medic',
                'minerals': r'v0\misc\minerals',
                'nouns': r'v1\nouns',
                'numerals': r'v1\numerals',
                'particles': r'v1\particles',
                'phys': r'v1\phys',
                'pl_es': r'v1\plural_es',
                'pl_ies': r'v1\plural_ies',
                'preposts': r'v1\preposts',
                'pronouns': r'v1\pronouns',
                'propnouns': r'v0\roots\propnouns',
                'rep_a': r'v0\misc\_adj_reps',
                'rep_m': r'v0\misc\_member_reps',
                'rep_pa': r'v0\misc\_past_reps',
                'rep_pf': r'v0\affixes\_prefix_reps',
                'rep_pl': r'v0\misc\_plural_reps',
                'rep_pr': r'v0\misc\_preprog_reps',
                'uniqs': r'v0\misc\uniqs',
                'roots': r'v0\roots\_base',
                'simplex': r'v0\roots\_simplex',
                'syn_1': r'v0\roots\old\_pre_syn',
                'syn_2': r'v0\roots\old\_syn',
                'verbs': r'v1\verbs',
                'wpres': r'v0\affixes\wpres',
                'wsufs': r'v0\affixes\wsufs',
                '_a1': r'v0\affixes\_a_ls',
                '_a2': r'v0\affixes\_a_ls2',
                '_e': r'v0\affixes\_e_ls',
            }.items()
        }
        self.wmode, self.debug = wmode, False
        self.wfiles = ['man_1', 'syn_1', 'rep_pl', 'rep_pr', 'rep_pa', 'rep_a', 'rep_m', 'man_2', 'syn_2']
        self.rgen = {'rls': self.get('roots'), 'rem': self.get('b_remove'), 'add': self.get('b_add'), 'cmls': self.get('cmls')}

        #Roots and bases subsection
        self.basis = set(wlst)
        self.bases = self.basis.copy()
        for x in self.rgen['rem']:
            if x in self.bases: self.bases.remove(x)
        for x in self.rgen['add']:
            self.bases.add(x)

        self.media = {x: Word(x) for x in self.bases}
        self.cdt = {}
        print(f'{len(self.bases)} Words')

        for x in self.get('uniqs'):
            if x not in self.media:
                self.media[x] = Word(x)
            if x not in self.bases:
                self.bases.add(x)
            self[x].tid = 1
        for x in self.get('latgre'):
            if x not in self.media:
                self.media[x] = Word(x)
            if x not in self.bases:
                self.bases.add(x)
            self[x].tid = 4
        for x in self.get('propnouns'):
            if x not in self.media:
                self.media[x] = Word(x)
            self[x].tid = 1
            self[x].exc = True
            if x in self.bases:
                self.bases.remove(x)

        """ for x in self.rgen['rls']:
            if x not in self.media:
                self[x] = Word(x)
                self[x].tid = 4
            else: self[x].tid = 1 """

        #Affixes subsection
        self.cut = 'absc abst acc add acq aff agg all app arr ass att coll corr ecc eff emb emp em eb ed eg ej el ep er ev ill irr imm imb imp occ off opp succ suff sugg summ supp sus syll sys'.split()
        with open(f'{self.files["affixes"]}', 'rt', encoding='utf8') as f:
            self.affixes = {x.strip() for x in f.readlines()}
            for x in self.affixes:
                self[x] = Word(x)
                self[x].tid = 2
            self.prex = sorted([x.strip('_') for x in self.affixes if x[0] == '_'])[::-1]
            self.sufx = sorted([x.strip('_') for x in self.affixes if x[-1] == '_'])[::-1]
        with open(f'{self.files["afx_prex"]}', 'rt', encoding='utf8') as f:
            self.afx_ignore = {}
            for x in f.readlines():
                x = x.strip().split()
                self.afx_ignore[x[0]] = [y.strip('_') for y in x[1:]]
        self.pdc = {'absc': 'ab', 'abst': 'ab', 'ecc': 'ex', 'emb': 'en', 'emp': 'en', 'opp': 'ob', 'occ': 'ob', 'of': 'ob', 'off': 'ob', 'ac': 'ad', 'ag': 'ad', 'al': 'ad', 'diff': 'dis'}
        for x in 'acc aff all agg app arr ass att'.split(): self.pdc[x] = 'ad'
        for x in 'comb comm comp coll corr'.split(): self.pdc[x] = 'con'
        for x in 'imb imm imp irr ill'.split(): self.pdc[x] = 'in'
        for x in 'eff ecc eb ed eg ej el em er ep ev e'.split(): self.pdc[x] = 'ex'
        for x in 'sys syll sym'.split(): self.pdc[x] = 'syn'
        for x in 'succ suff sugg supp summ sus'.split(): self.pdc[x] = 'sub'
        for x in self.pdc.items():
            if len(x[0]) > 1:
                if x[0][-1] == x[0][-2] and f'_{x[0][:-1]}' not in self.media: self.media[f'_{x[0][:-1]}'] = self.media[f'_{x[1]}']
                elif f'_{x[0]}' not in self.media: self.media[f'_{x[0]}'] = self.media[f'_{x[1]}']

        #Command ingest subsection
        for x in self.rgen['cmls']: self.cread(x)
        print(f'{len(self.bases)} Words, {len(self.media)} Fragments')
        

    #Command ingest commands
    def cread(self, command):
        #Process word command        
        for x in command[1:]:
            if x == '|': continue
            if '+' in x:
                x = x.strip('+')
                if x not in self.bases: self.bases.add(x)
            if not x in self.media: self[x] = Word(x)
            if command[0] == 'S': self[x].tid = 3
            else: self[x].tid = 4
            if self.debug: print(f'{x}: Command component word not in component list')
        if command[0] == 'S':
            self.supdate(command[1], command[2:])
        elif command[0] == 'M':
            self.mupdate(command[1], command[2:])
        else: raise ValueError(f'Invalid command {command}')

    def mupdate(self, src: str, sinks: list[str]):
        #Break a source word into composite subwords and update groups
        if sinks[-1] == '|':
            self[src].bcomp = sinks[:-1]
            for x in sinks[:-1]: self[x].acomp.append(src)
            self[src].homos.append(tuple(sinks[:-1]))
        else:
            if self[src].tid not in (0, 2, 4) and self.debug: print(f'Cannot merge {src} code: {self[src].tid}')
            else:
                if self[src].bcomp and not self[src].homos and self.debug: print(f'{src} Command source already decomposed {sinks} | {self[src].bcomp}')
                if sinks[-1] == '-_': self[src].exc = True
                self[src].bcomp = sinks
                self.cdt[src] = ' '.join(sinks)
                for x in sinks: self[x].acomp.append(src)
                if src in self.bases: self.bases.remove(src)

    def supdate(self, sink: str, srcs: list[str]):
        #Merge synonym words and update
        for src in srcs:
            if '+' in src: src = src.strip('+')
            else: self[src].exc = True
            if self.media[src].tid == 1 and self.debug: print(f'Cannot syn merge root {src} into {sink}')
            self[sink].syns.append(src)
            if src in self.bases: self.bases.remove(src)


    #Commands for modifying base word lists
    @property
    def roots(self):
        return [x for x in self.media if self.media[x].tid == 1]

    @property
    def synw(self):
        return sorted([y.strip('+') for x in self.rgen['cmls'] for y in x[2:] if x[0] == 'S'])
    
    @property
    def manw(self):
        return sorted([x[1] for x in self.rgen['cmls'] if x[0] == 'M' and not x[-1] == '|'])
    
    @property
    def ups(self):
        ups = list(self.get('uniqs'))
        ups.extend(self.get('propnouns'))
        return sorted(ups)

    def get(self, group=None, intersects=False):
        #Retrieve manual word file
        if not group:
            for x in sorted(self.files): print(x)
        if group not in self.files: raise ValueError(f'{group} invalid group name')
        with open(self.files[group], 'rt', encoding='utf8') as f:
            if group.startswith('cmls'): return [tuple(x.strip().split()) for x in f.readlines()]
            elif group.startswith('syn'): return [('S', *x.strip().split()) for x in f.readlines()]
            elif group.startswith('man') or group.startswith('rep'): return [('M', *x.strip().split()) for x in f.readlines()]
            elif intersects: return [x.strip() for x in f.readlines() if x.strip() in intersects]
            else: return set([x.strip() for x in f.readlines()])

    def reload(self):
        #Reload base words
        hold = [tuple(x) for x in self.get('cmls')]
        self.rgen['cmls'].extend([x for x in hold if x not in self.rgen['cmls']])
        self.rgen['rem'].extend([x for x in self.get('b_remove') if x not in self.rgen['rem']])
        self.rgen['add'].extend([x for x in self.get('b_add') if x not in self.rgen['add']])
        self.csort()

    def cfind(self, word, src=0, chain=False):
        #Search command list
        if not chain:
            if src == 1: return [x for x in self.rgen['cmls'] if word in x[2:]]
            elif src == 2: return [x for x in self.rgen['cmls'] if any(word in z for z in x[2:])]
            else: return [x for x in self.rgen['cmls'] if f'{word}+' in x or word in x]
        else:
            if src == 1: coms = [x for x in self.rgen['cmls'] if word in x[2:]]
            elif src == 2: coms = [x for x in self.rgen['cmls'] if any(word in z for z in x[2:])]
            else: coms = [x for x in self.rgen['cmls'] if f'{word}+' in x or word in x]
            ncnt = -1
            ccnt = len(coms)
            while ncnt != ccnt:
                ccnt = len(coms)
                coms.extend([y for x in coms for y in self.cfind(x[2], 0) if y[0] == 'M' and y not in coms])
                ncnt = len(coms)
            return coms

    def csort(self, commands):
        #Sort command list
        sct = len(commands)
        elms = {}
        snks = {}
        out = [x for x in commands if x[-1] == '|' or x[0] == 'S']
        incom = [x for x in commands if x[-1] != '|' and x[0] == 'M']
        for x in incom:
            if x[0] == 'S':
                for y in x[2:]:
                    if y.strip('+') in elms: print(f'Conflict {x} | {elms[y.strip("+")]}')
                    elms[y.strip('+')] = x
            else:
                if x[-1] == '|': continue
                if x[1] in elms: print(f'Conflict {x} | {elms[x[1]]}')
                elms[x[1]] = x
                for y in x[2:]:
                    if y not in snks: snks[y] = [x[1]]
                    else: snks[y].append(x[1])
        rmls = []
        for x in snks:
            if x not in elms:
                rmls.append(x)
        for x in rmls: snks.pop(x)
        beg, end = len(incom), 0
        while beg != end:
            rmls = []
            beg = len(incom)
            for x in incom:
                if x[1] not in snks:
                    out.append(x)
                    rmls.append(x)
                    for y in x[2:]:
                        if y in snks:
                            snks[y].remove(x[1])
                            if not snks[y]: snks.pop(y)
            for x in rmls: incom.remove(x)
            end = len(incom)
        if len(out) != sct:
            for x in incom:
                print(x)
            raise ValueError('Incomplete sort recursive')
        else: return out

    def regen(self):
        #Update and reload base files
        if not self.wmode: raise ValueError('Not in working mode')
        kd = {'rls': 'base',
              'rem': 'base_remove', 
              'add': 'base_add', 
              'cmls': 'cmls'}
        self.csort()
        self.rgen['cmls'].extend([tuple(z) for z in self.get('cmls') if tuple(z) not in self.rgen['cmls']])
        self.rgen['rem'].extend([x for x in self.get('b_remove') if x not in self.rgen['rem']])
        self.rgen['add'].extend([x for x in self.get('b_add') if x not in self.rgen['add']])
        for x in kd.values():
            rename(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\roots\\_{x}',
                   f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\roots\\old\\_{x}_{time()}')
        for x in self.rgen.items():
            if x[0] in kd:
                with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\roots\\_{kd[x[0]]}', 'wt') as f:
                    tmp = [f'{" ".join(y)}\n' if not isinstance(y, str) else f'{y}\n' for y in x[1]]
                    tmp[-1] = tmp[-1][:-1]
                    f.writelines(tmp)

    def cd(self, word):
        #Remove word from base files
        if self.cfind(word): raise ValueError(f'{word} in rule {self.cfind(word)}')
        if word in self.rgen['rls']: self.rgen['rls'].remove(word)
        if word in self.rgen['add']: self.rgen['add'].remove(word)
        if word not in self.rgen['rem']: self.rgen['rem'].append(word)

    def aw(self, word, root=False):
        #Add word to base files
        if word in self.rgen['rem']: self.rgen['rem'].remove(word)
        if word not in self.rgen['add'] and word not in self.media: self.rgen['add'].append(word)
        if root and word not in self.rgen['rls']: self.rgen['rls'].append(word)

    def crep(self, ow, nw):
        #Replace word in base / root files
        owp = f'{ow}+'
        nwp = f'{nw}+'
        for i, x in enumerate(self.rgen['cmls']):
            if ow in x or owp in x:
                if ow in x:
                    x = [nw if y == ow else y for y in x]
                if owp in x:
                    x = [nwp if y == owp else y for y in x]
                self.rgen['fml'][i] = tuple(x)

        if ow in self.rgen['add']: self.rgen['add'].remove(ow)
        if nw not in self.rgen['add']: self.rgen['add'].append(nw)
        if nw in self.rgen['rem']: self.rgen['rem'].remove(nw)

        if ow in self.rgen['rls']:
            self.rgen['rls'].remove(ow)
            if nw not in self.rgen['rls']:
                self.rgen['rls'].append(nw)

    def nmcom(self, sink, source, affixes):
        #Add manual command
        if isinstance(affixes, str): affixes = (affixes,)
        self.rgen['cmls'].append(('M', source, (sink, *affixes)))

    def nscom(self, sink, source):
        #Add syn command
        sp = f'{sink}+'
        found = False
        for i, x in enumerate(self.rgen['cmls']):
            if x[0] != 'S': continue
            if sink in x or sp in x:
                self.rgen['cmls'][i] = (*x, source)
                found = True
        if not found: self.rgen['cmls'].append(('S', sink, source))

    def rcom(self, newcom):
        for x in self.rgen['cmls']:
            if x[0] == 'S': continue
            if x[1] == newcom[0]:
                rep = x
                break
        else: return
        self.rgen['cmls'].remove(rep)
        self.rgen['cmls'].append(('M', *newcom))
        self[newcom[0]].bcomp = tuple(newcom)

    def savcom(self):
        rename(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\roots\\_cmls4',
                f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\roots\\old\\_cmls4_{time()}')
        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\roots\\_cmls4',  'wt') as f:
            f.writelines([f'{" ".join(x)}\n' for x in self.rgen['cmls']])


    #Commands for decomposing words with specific affixes, inner level, global rules
    def rep_sfx(self, word: str, al: int, t: str='', r: str='', vd=None) -> None | str:
        if t in ('dbl', 'ck', 'iy', 'fv', 'ctx', 'icex'): alreq = 3
        else: alreq = 2
        alreq += al
        if len(word) < alreq:
            return False

        if t == 'dbl':
            if not vd: vd = {1: 'bdglmnprt'}
            if word[-(al+1)] != word[-(al+2)]: return
        elif t == 'iy':
            if word[-(al+1)] != 'i': return
        elif t == 'ck':
            if word[-(al+1)] != 'k' or word[-(al+2)] != 'c': return
        elif t == 'er':
            if word[-al] != 'r': return
        elif t == 'fv':
            if word[-(al+2):(-al if al else None)] != 've': return
        elif t in ('icix', 'icex'):
            if word[-(al+2):(-al if al else None)] != 'ic': return
        elif t == 'ct':
            if not ((word[-(1+al)] == 'c' and word[-(2+al)] in 'aeiou') or (word[-(1+al)] in 'aeiou' and word[-(2+al)] == 'c')): return
        elif t in ('ctx', 'cte'):
            if word[-(al+2):(-al if al else None)] != 'ct': return
        elif t == 'ntce':
            if word[-(al+3):(-al if al else None)] != 'nce': return
        if vd:
            if isinstance(vd, tuple):
                if word[vd[0]:(vd[0]+len(vd[1]) if vd[0]+len(vd[1]) < 0 else None)] != vd[1]:
                    return False
            elif isinstance(vd, dict):
                for k in vd:
                    if word[-(al+k)] not in vd[k]:
                        return False

        if not t:
            return f'{word[:-al]}{r}'
        elif t == 'e':
            return f'{word[:-al]}e'
        elif t == 'dbl':
            return f'{word[:-(al+1)]}{r}'
        elif t == 'iy':
            return f'{word[:-(al+1)]}y'
        elif t == 'ck':
            return f'{word[:-(al+1)]}'
        elif t == 'er':
            return f'{word[:-al]}er'
        elif t == 'fv':
            if f'{word[:-(al+2)]}f' in self.media: return f'{word[:-(al+2)]}f'
            else: return f'{word[:-(al+2)]}fe'
        elif t == 'icix':
            return f'{word[:-(al+1)]}x'
        elif t == 'icex':
            return f'{word[:-(al+2)]}ex'
        elif t == 'ct':
            if (word[-(1+al)] == 'c' and word[-(2+al)] in 'aeiou'): return f'{word[:-(1+al)]}te'
            else: return f'{word[:-(2+al)]}te'
        elif t == 'ctx':
            return f'{word[:-(al+2)]}x'
        elif t == 'cte':
            return f'{word[:-(al+1)]}e'
        elif t == 'ntce':
            return f'{word[:-(al+2)]}t'
        elif t == 'm':
            for i in range(1, al+1):
                if not word[:-i] in self.media: continue
                return word[:-i]

    def rep_pfx(self, word, pfx) -> str:
        if pfx in self.afx_ignore and any(word.startswith(z) for z in self.afx_ignore[pfx]): return
        elif pfx in self.cut: return word[len(pfx)-1:]
        else: return word[len(pfx):]

    def rep_afx(self, word: str, al: int, t: str='', r: str='', vd: dict[int, str]=None, tx: str='', bridges: dict=None, pref: bool=False, force: bool=False) -> str|None:
        """
        Remove affixes from a word.
        Intended to be highly specific with a bias towards failing so that multiple argument groups can be used simultaneously with only one succeeding.

        Args:
            word (str): Input word
            al (int): affix length
            t (str, optional): Root suffix joining type. Defaults to ''.
            r (str, optional): Suffix replacement string. Defaults to ''.
            vd (dict, optional): Verification dictionary for validating suffix replacement candidates. Letters at the key index must be one of the letters in the value string. Indexing is reversed and starts at 1. Defaults to None.
            tx (str, optional): Root suffix joining super type. Defaults to ''.
            bridges (dict, optional): Additonal letters to be removed/replaced optionally after the affix length has been trimmed. Defaults to None.
            pref (bool, optional): Attempt to remove prefix. Defaults to False.

        Returns:
            str|None: Root word
        """
        if not bridges: bridges = {'': ''}
        else: bridges[''] = ''
        for bk in bridges:
            rep = self.rep_sfx(word, al, t, r, vd)
            if not rep or len(rep)-len(bk) < 3: continue
            if not rep.endswith(bk): continue
            rep = f'{rep[:-len(bk) if bk else None]}{bridges[bk]}'
            if pref and not tx:
                for prefix in sorted([x for x in self.prex if rep.startswith(x) and len(rep)-len(x) > 2]):
                    pfx_rep = self.rep_pfx(rep, prefix)
                    if pfx_rep in self.media: return (pfx_rep, f'{prefix if prefix not in self.pdc else self.pdc[prefix]}')
                    elif prefix == 'ex' and self.check(f's{pfx_rep}'): return (f's{pfx_rep}', f'{prefix if prefix not in self.pdc else self.pdc[prefix]}')
            if pref:
                pfxs = [(self.rep_pfx(rep, z), z) for z in sorted(self.prex) if rep.startswith(z) and len(rep)-len(z) > 2 and self.rep_pfx(rep, z)]
                pfxs.append((rep, ''))
            if tx == 'v':
                #Science latin greek suffix root clipping: um us os is as eo a e o
                for v in self.lgs:
                    if not force and word[-al:] == v: continue
                    if rep[-1] in 'iuoa':
                        if pref:
                            for pgp in pfxs:
                                if f'{pgp[0]}{v}' in self.media: return (f'{pgp[0]}{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                if f'{pgp[0]}s{v}' in self.media: return (f'{pgp[0]}s{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                if rep[-1] == 'u' and f'{pgp[0]}m{v}' in self.media: return (f'{pgp[0]}m{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                if rep[-1] == 'o' and f'{pgp[0]}n{v}' in self.media: return (f'{pgp[0]}n{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                        else:
                            if f'{rep}{v}' in self.media: return f'{rep}{v}'
                            if f'{rep}s{v}' in self.media: return f'{rep}s{v}'
                            if rep[-1] == 'u' and f'{rep}m{v}' in self.media: return f'{rep}m{v}'
                            if rep[-1] == 'o' and f'{rep}n{v}' in self.media: return f'{rep}n{v}'
                    else:
                        if pref:
                            for pgp in pfxs:
                                if f'{pgp[0]}{v}' in self.media: return (f'{pgp[0]}{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                        elif f'{rep}{v}' in self.media: return f'{rep}{v}'
            elif tx == 'vx':
                if rep[-2:] == 'ct':
                    if pref:
                        for pgp in pfxs:
                            for v in self.lgs:
                                if word[-al:] == v: continue
                                if f'{pgp[0][:-2]}{v}' in self.media: return (f'{pgp[0][:-2]}{v}', pgp[1])
                                elif f'{pgp[0][:-2]}s{v}' in self.media: return (f'{pgp[0][:-2]}s{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                elif f'{pgp[0][:-2]}g{v}' in self.media: return (f'{pgp[0][:-2]}g{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                elif f'{pgp[0][:-2]}x{v}' in self.media: return (f'{pgp[0][:-2]}x{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                    else:
                        for v in self.lgs:
                            if word[-al:] == v: continue
                            if f'{rep[:-2]}{v}' in self.media: return f'{rep[:-2]}{v}'
                            elif f'{rep[:-2]}s{v}' in self.media: return f'{rep[:-2]}s{v}'
                            elif f'{rep[:-2]}g{v}' in self.media: return f'{rep[:-2]}g{v}'
                            elif f'{rep[:-2]}x{v}' in self.media: return f'{rep[:-2]}x{v}'
                elif rep[-1] in 'ct':
                    if pref:
                        for pgp in pfxs:
                            for v in self.lgs:
                                if word[-al:] == v: continue
                                if f'{pgp[0][:-1]}s{v}' in self.media: return (f'{pgp[0][:-1]}s{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                elif f'{pgp[0][:-1]}g{v}' in self.media: return (f'{pgp[0][:-1]}g{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                elif f'{pgp[0][:-1]}x{v}' in self.media: return (f'{pgp[0][:-1]}x{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                    else:
                        for v in self.lgs:
                            if word[-al:] == v: continue
                            if f'{rep[:-1]}s{v}' in self.media: return f'{rep}s{v}'
                            elif f'{rep[:-1]}g{v}' in self.media: return f'{rep}g{v}'
                            elif f'{rep[:-1]}x{v}' in self.media: return f'{rep}x{v}'
            elif tx == 'xv':
                #Vowel expansion: proclamation proclaim
                if rep[-1] in 'aeiou':
                    for v in 'aeiou':
                        if pref:
                            for pgp in pfxs:
                                if f'{pgp[0][:-1]}{v}{pgp[0][-1]}' in self.media: return (f'{pgp[0][:-1]}{v}{pgp[0][-1]}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                        elif f'{rep[:-1]}{v}{rep[-1]}' in self.media: return f'{rep[:-1]}{v}{rep[-1]}'
                elif rep[-2] in 'aeiou':
                    for v in 'aeiou':
                        if pref:
                            for pgp in pfxs:
                                if f'{pgp[0][:-2]}{v}{pgp[0][-2:]}' in self.media: return (f'{pgp[0][:-2]}{v}{pgp[0][-2:]}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                        elif f'{rep[:-2]}{v}{rep[-2:]}' in self.media: return f'{rep[:-2]}{v}{rep[-2:]}'
            elif tx == 'vc':
                #ul il er ending rearrangement
                if pref:
                    for pgp in pfxs:
                        if pgp[0][-1] == 'l' and pgp[0][-2] in 'ui' and f'{pgp[0][:-2]}le' in self.media: return (f'{pgp[0][:-2]}le', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                        if pgp[0][-1] == 'r' and pgp[0][-2] not in 'aeiouy' and f'{pgp[0][:-1]}er' in self.media: return (f'{pgp[0][:-1]}er', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                else:
                    if rep[-1] == 'l' and rep[-2] in 'ui' and f'{rep[:-2]}le' in self.media: return f'{rep[:-2]}le'
                    if rep[-1] == 'r' and rep[-2] not in 'aeiouy' and f'{rep[:-1]}er' in self.media: return f'{rep[:-1]}er'
            elif tx == 'lag':
                #Latin Greek
                if pref:
                    for lage in ('us', 'a', 'um', 'is', 'os', 'o', 'e', '', 's', 'x', 'io', 'eo'):
                        if f'{rep}{lage}' in self.media: return (f'{rep}{lage}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                else:
                    for lage in ('us', 'a', 'um', 'is', 'os', 'o', 'e', '', 's', 'x', 'io', 'eo'):
                        if f'{rep}{lage}' in self.media: return f'{rep}{lage}'
            else:
                if pref:
                    for pgp in pfxs:
                        if pgp[0] in self.media: return (pgp[0], pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                elif rep in self.media: return rep

    def rep_pafx(self, word, al):
        if word[al:] in self.media: return word[al:]
        elif word[al:][0] in 'oiau' and word[al+1:] in self.media: return word[al+1:]

    #Commands for decomposing words with specific affixes, outer level, finds and filters word bases, local rules
    def pl_parse(self, pref=False):
        sl = len(self.bases)
        with open(f'{self.files["ign_pl"]}', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]
        if not pref and not self.wmode:
            with open(f'{self.files["pl_ies"]}', 'rt', encoding='utf8') as f:
                pies = {x.strip() for x in f.readlines()}
                for x in [x for x in self.bases if x.endswith('ies')]:
                    if x in pies:
                        y = self.rep_afx(x, 2, t='iy')
                        if y:
                            self.mupdate(x, (y, 's_'))
                            self[y].pos = 'noun'
            with open(f'{self.files["pl_es"]}', 'rt', encoding='utf8') as f:
                pes = {x.strip() for x in f.readlines()}
                for x in [x for x in self.bases if x.endswith('es')]:
                    if x in pes:
                        y = self.rep_afx(x, 2)
                        if y:
                            self.mupdate(x, (y, 's_'))
                            self[y].pos = 'noun'
            with open(f'{self.files["rep_pl"]}', 'rt', encoding='utf8') as f:
                for x in [x.strip().split() for x in f.readlines()]:
                    self.mupdate(x[1], (x[0], x[2]))
                    self[x[0]].pos = 'noun'

        targets = [x for x in self.bases if len(x) > 3 and x.endswith('s') and x not in igls]
        for x in targets:
            mx = False

            if x.endswith('s') and x[-2] != 's' and (mx := self.rep_afx(x, 1, pref=pref)): pass
            elif x.endswith('es') and (mx := self.rep_afx(x, 2, vd={1: 'shoxz'}, pref=pref)): pass
            elif x.endswith('es') and (mx := self.rep_afx(x, 2, t='dbl', pref=pref)): pass
            elif x.endswith('ies') and (mx := self.rep_afx(x, 2, t='iy', pref=pref)): pass
            elif x.endswith('ves') and (mx := self.rep_afx(x, 1, t='fv', pref=pref)): pass
            elif x.endswith('ices') and (mx := self.rep_afx(x, 2, t='icix', pref=pref)): pass
            elif x.endswith('ices') and (mx := self.rep_afx(x, 2, t='icex', pref=pref)): pass
            elif x.endswith('es') and (mx := self.rep_afx(x, 2, r='is', pref=pref)): pass

            if mx:
                if pref: self.mupdate(x, (mx[0], mx[1], 's_'))
                else: self.mupdate(x, (mx, 's_'))
        print(f'{sl - len(self.bases)} items combined for plurals\n{len(self.bases)} remaining')

    def prpt_parse(self, pref=False):
        sl = len(self.bases)
        with open(f'{self.files["ign_pr"]}', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]
        if not pref and not self.wmode:
            with open(f'{self.files["rep_pr"]}', 'rt', encoding='utf8') as f:
                for x in [x.strip().split() for x in f.readlines()]:
                    self.mupdate(x[1], (x[0], x[2]))
                    self[x[0]].pos = 'verb'

        targets = [x for x in self.bases if len(x) > 5 and x.endswith('ing') and x not in igls]
        for x in targets:
            mx = False

            if (mx := self.rep_afx(x, 3, t='dbl', pref=pref)): pass
            elif (mx := self.rep_afx(x, 3, r='e', pref=pref)): pass
            elif (mx := self.rep_afx(x, 3, pref=pref)): pass
            elif (mx := self.rep_afx(x, 3, t='ck', pref=pref)): pass
            if mx:
                if pref:
                    self.mupdate(x, (mx[0], mx[1], 'ing_'))
                    self[mx[0]].pos = 'verb'
                else:
                    self.mupdate(x, (mx, 'ing_'))
                    self[mx].pos = 'verb'
        print(f'{sl - len(self.bases)} items combined for present progressives\n{len(self.bases)} remaining')

    def pt_parse(self, pref=False):
        sl = len(self.bases)
        with open(f'{self.files["ign_pa"]}', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]
        if not pref and not self.wmode:
            with open(f'{self.files["rep_pa"]}', 'rt', encoding='utf8') as f:
                for x in [x.strip().split() for x in f.readlines()]:
                    self.mupdate(x[1], (x[0], x[2]))
        
        targets = [x for x in self.bases if len(x) > 4 and x.endswith('ed') and x not in igls]
        if pref: targets = [x for x in targets if not any(x.endswith(y) for y in igls)]
        for x in targets:
            mx = False
            if x.endswith('ied') and (mx := self.rep_afx(x, 2, t='iy', pref=pref)): afx = 'ed_'
            elif (mx := self.rep_afx(x, 2, pref=pref)): afx = 'ed_'
            elif (mx := self.rep_afx(x, 2, t='e', pref=pref)): afx = 'ed_'
            elif (mx := self.rep_afx(x, 2, t='dbl', pref=pref)): afx = 'ed_'
            elif (mx := self.rep_afx(x, 2, t='ck', pref=pref)): afx = 'ed_'
            if mx:
                if pref: self.mupdate(x, (mx[0], mx[1], afx))
                else: self.mupdate(x, (mx, afx))
        print(f'{sl - len(self.bases)} items combined for past \n{len(self.bases)} remaining')

    def adjv_parse(self, pref=False):
        #er/est | y
        sl = len(self.bases)
        with open(f'{self.files["ign_a"]}', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]
        if not pref:
            if not self.wmode:
                with open(f'{self.files["rep_a"]}', 'rt', encoding='utf8') as f:
                    for x in [x.strip().split() for x in f.readlines()]:
                        self.mupdate(x[1], (x[0], x[2]))

            sgrp = [(x, f'{x[:-2]}r') for x in [x for x in self.bases if x.endswith('est')] if f'{x[:-2]}r' in self.bases and x not in igls]
            al = 3
            for x in sgrp:
                pack = []
                for k, y in enumerate(x):
                    if (mx := self.rep_afx(y, al-k)): pack.append(mx)
                    if (mx := self.rep_afx(y, al-k, r='e')): pack.append(mx)
                    if (mx := self.rep_afx(y, al-k, t='iy')): pack.append(mx)
                    if (mx := self.rep_afx(y, al-k, t='dbl')): pack.append(mx)

                pack = [y for y in pack if y not in igls]
                if not pack or len(pack) < 2: continue
                self.mupdate(x[0], (pack[0], 'est_' if x[0].endswith('est') else 'er_'))
                self.mupdate(x[1], (pack[0], 'est_' if x[1].endswith('est') else 'er_'))

        targets = [x for x in self.bases if (len(x) > 4 and x.endswith('y')) and x not in igls]
        for x in targets:
            mx = False
            afx = f'{x[-2:]}_'

            if x.endswith('ly') and (mx := self.rep_afx(x, 2, vd={1: 'bcdefghklmnprstwordy'}, pref=pref)): pass
            elif x.endswith('ry') and (mx := self.rep_afx(x, 2, vd={1: 'cdeklnt'}, pref=pref)): pass
            elif x.endswith('ty') and (mx := self.rep_afx(x, 2, vd={1: 'elx'}, pref=pref)): afx = f'ity_'
            elif x.endswith('bility') and (mx := self.rep_afx(x, 5, r='le', pref=pref)): afx = f'ity_'
            elif x.endswith('cy') and (mx := self.rep_afx(x, 2, r='te', vd={1: 'a'}, pref=pref)): pass
            elif x.endswith('cy') and (mx := self.rep_afx(x, 2, r='t', vd={1: 'n'}, pref=pref)): pass
            elif (mx := self.rep_afx(x, 1, vd={1: 'dfghklmnprstwordz'}, pref=pref)): afx = 'y_'
            elif x.endswith('ary') and (mx := self.rep_afx(x, 3, r='e', vd={1: 'bcdgklmnprstvz'}, pref=pref)): afx = f'{x[-3:]}_'
            elif any(x.endswith(y) for y in ('ily', 'ory')) and (mx := self.rep_afx(x, 3, r='e', vd={1: 'bcdgklmnprstvz'}, pref=pref)): pass
            elif any(x.endswith(y) for y in ('ity', 'ify')) and (mx := self.rep_afx(x, 3, r='e', vd={1: 'bcdgklmnprstvz'}, pref=pref)): afx = f'{x[-3:]}_'
            elif (mx := self.rep_afx(x, 1, r='e', vd={1: 'bcdgklmnprstvz'}, pref=pref)): afx = 'y_'
            elif (mx := self.rep_afx(x, 1, t='dbl', vd={1: 'bdglmnpt'}, pref=pref)): afx = 'y_'
            elif (mx := self.rep_afx(x, 2, vd={0: 'r', 1: 'r', 2: 'u'}, pref=pref)): pass
            elif x.endswith('ically') and (mx := self.rep_afx(x, 4, pref=pref)): pass
            elif (x.endswith('arily') or x.endswith('sily')) and (mx := self.rep_afx(x, 3, r='y', pref=pref)): pass
            elif x.endswith('llary') and (mx := self.rep_afx(x, 4, pref=pref)): afx = f'{x[-3:]}_'
            elif x.endswith('ary') and (mx := self.rep_afx(x, 3, vd={1: 'bdmnrt'}, pref=pref)): afx = f'{x[-3:]}_'
            elif x.endswith('ily') and (mx := self.rep_afx(x, 3, vd={1: 'dhkmpt'}, pref=pref)): pass
            elif x.endswith('ily') and (mx := self.rep_afx(x, 3, t='dbl', vd={1: 'ndp'}, pref=pref)): pass
            elif x.endswith('ity') and (mx := self.rep_afx(x, 3, vd={1: 'cdelmnrtx'}, pref=pref)): afx = f'{x[-3:]}_'
            elif x.endswith('ity') and (mx := self.rep_afx(x, 3, t='dbl', vd={1: 'lp'}, pref=pref)): afx = f'{x[-3:]}_'
            elif x.endswith('ory') and (mx := self.rep_afx(x, 3, vd={1: 'st'}, pref=pref)): pass
            elif x.endswith('ery') and (mx := self.rep_afx(x, 3, t='dbl', vd={1: 'bgln'}, pref=pref)): pass
            elif any(x.endswith(y) for y in ('ily', 'ory')) and (mx := self.rep_afx(x, 3)): pass
            elif any(x.endswith(y) for y in ('ity', 'ify', 'ary')) and (mx := self.rep_afx(x, 3)): afx = f'{x[-3:]}_'

            if mx:
                if pref: self.mupdate(x, (mx[0], mx[1], afx))
                else: self.mupdate(x, (mx, afx))

        print(f'{sl - len(self.bases)} items combined for adjectives \n{len(self.bases)} remaining')

    def en_parse(self, pref=False):
        sl = len(self.bases)
        igls = 'listen albumen haven molten pollen dozen graben midden garden baleen spleen careen lichen maiden heathen token salen ramen somen rumen semen linen aspen siren warren paten marten solen'.split()

        targets = [x for x in self.bases if len(x) > 4 and x.endswith('en') and x not in igls and not any(x.endswith(y) for y in ('screen', 'gen', 'teen', 'seen'))]
        al = 2
        for x in targets:
            mx = False

            if (mx := self.rep_afx(x, al, t='dbl', pref=pref)): pass
            elif (mx := self.rep_afx(x, al, pref=pref)): pass
            elif (mx := self.rep_afx(x, al, t='e', pref=pref)): pass
            elif (mx := self.rep_afx(x, al, t='dbl', r='e', pref=pref)): pass
            if mx:
                if pref: self.mupdate(x, (mx[0], mx[1], 'en_'))
                else: self.mupdate(x, (mx, 'en_'))
        print(f'{sl - len(self.bases)} items combined for en suffix\n{len(self.bases)} remaining')

    def mbr_parse(self, pref=False):
        sl = len(self.bases)
        with open(f'{self.files["ign_m"]}', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]
        fls = 'meter water power flower polar'.split()
        if not pref and not self.wmode:
            with open(f'{self.files["rep_m"]}', 'rt', encoding='utf8') as f:
                for x in [x.strip().split() for x in f.readlines()]:
                    self.mupdate(x[1], (x[0], x[2]))

        al = 3
        targets = [x for x in self.bases if len(x) > 5 and (x.endswith('ian') or x.endswith('ist')) and not x in igls]
        for x in targets:
            mx = False
            afx = f'{x[-3:]}_'

            if (mx := self.rep_afx(x, 3, pref=pref)): pass
            elif (mx := self.rep_afx(x, al, r='y', pref=pref)): pass
            elif x.endswith('scientist') and (mx := self.rep_afx(x, len('scientist'), r='science', pref=pref)): afx = 'ist_'
            elif x.endswith('tarian') and (mx := self.rep_afx(x, 5, r='y', pref=pref)): afx = 'ian_'
            elif x.endswith('ician') and (mx := self.rep_afx(x, 3, pref=pref)): afx = 'cian_'
            elif x.endswith('ician') and (mx := self.rep_afx(x, 5, pref=pref)): afx = 'cian_'
            elif x.endswith('ian') and (mx := self.rep_afx(x, al, t='m', pref=pref)): afx = 'ian_'
            elif (mx := self.rep_afx(x, al, pref=pref)): pass
            elif (mx := self.rep_afx(x, al, t='e', pref=pref)): pass
            elif (mx := self.rep_afx(x, al, t='dbl', pref=pref)): pass
            elif (mx := self.rep_afx(x, al, r='ic', pref=pref)): pass
            elif (mx := self.rep_afx(x, 1, r='m', pref=pref)): pass

            if mx:
                if pref: self.mupdate(x, (mx[0], mx[1], afx))
                else: self.mupdate(x, (mx, afx))

        targets = [x for x in self.bases if len(x) > 4 and x[-2:] in ('er', 'ee', 'or') and not x in igls and all(y not in x for y in fls)]
        for x in targets:
            if x.endswith('ster'): al = 4
            else: al = 2
            mx = False

            if (mx := self.rep_afx(x, al, t='m', pref=pref)): pass
            elif (mx := self.rep_afx(x, al, t='e', pref=pref)): pass
            elif x.endswith('ier') and (mx := self.rep_afx(x, al, t='iy', pref=pref)): pass
            elif x.endswith('ier') and (mx := self.rep_afx(x, al, t='e', pref=pref)): pass
            elif x.endswith('eer') and (mx := self.rep_afx(x, al+1, pref=pref)): pass
            elif (mx := self.rep_afx(x, al, t='dbl', pref=pref)): pass
            elif x.endswith('ster') and (mx := self.rep_afx(x, al, pref=pref)): pass

            if mx:
                if pref: self.mupdate(x, (mx[0], mx[1], f'{x[-al:]}_'))
                else: self.mupdate(x, (mx, f'{x[-al:]}_'))

        targets = [x for x in self.bases if len(x) > 6 and x.endswith('ling') and not x in igls]
        for x in targets:
            mx = False
            if (mx := self.rep_afx(x, 4, pref=pref)):
                if pref: self.mupdate(x, (mx[0], mx[1], 'ling_'))
                else: self.mupdate(x, (mx, 'ling_'))

        print(f'{sl - len(self.bases)} items combined for membership \n{len(self.bases)} remaining')

    def sfx1_parse(self, pref=False):
        sl = len(self.bases)
        igls = 'magister manoeuvre parable liable capable arable sister amiable gullible malleable tangible tanginess semen talisman shamaness'.split()

        al = 4
        targets = [x for x in self.bases if len(x) > 4 and x not in igls and any(x.endswith(t) for t in ('less', 'ness', 'able', 'ible'))]
        if pref: targets = [x for x in targets if not x in igls]
        for x in targets:
            mx = False

            if (mx := self.rep_afx(x, al, pref=pref)): pass
            elif (x.endswith('able') or x.endswith('ible')) and (mx := self.rep_afx(x, al, r='e', pref=pref)): pass
            elif (mx := self.rep_afx(x, al, t='iy', pref=pref)): pass

            if mx:
                if x.endswith('ness'): afx = 'ness_'
                elif x.endswith('ible'): afx = 'able'
                else: afx = x[-4:]
                if pref: self.mupdate(x, (mx[0], mx[1], afx))
                else: self.mupdate(x, (mx, afx))

        for tgt in (('woman', 'women'), ('man', 'men')):
            al = len(tgt[0])
            targets = [x for x in self.bases if len(x) > 5 and x not in igls and (x.endswith(tgt[0]) or x.endswith(tgt[1]))]
            if pref: targets = [x for x in targets if not any(x.endswith(y) for y in igls)]
            for x in targets:
                mx = False
                if (mx := self.rep_afx(x, al, pref=pref)): self.mupdate(x, (mx, tgt[0]))
        print(f'{sl - len(self.bases)} items combined for sfx1 \n{len(self.bases)} remaining')

    def pre_parse(self):
        sl = len(self.bases)
        with open(f'{self.files["ign_pf"]}', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]

        for pfx in 'absc abst ab\
                    ad acc aff agg all app arr ass att\
                    con comb comm comp coll corr\
                    anti ana an de dis mis non ill irr im imb imm imp un\
                    pre pro for para per re\
                    cata dys eu sub ante pre epi poly\
                    ex ec eb ed eff eg el ej ep er ev\
                    en emb emp in\
                    ob occ off opp\
                    dia peri ambi pan\
                    allo apo tele\
                    syn sus sym sys succ suff sugg summ supp syll iso meta auto'.split():
            if pfx in self.pdc: pfl = len(self.pdc[pfx])
            else: pfl = len(pfx)
            for word in sorted([x for x in self.bases if x.startswith(pfx) and len(x)-pfl > 2 and not any(x.startswith(y) for y in igls)], key=lambda x: 1/len(x)):
                mx = self.rep_pfx(word, pfx)
                if mx in self.media and len(mx) > 2:
                    if pfx in self.pdc: pfx = self.pdc[pfx]
                    print(f'M {mx} {word} _{pfx}')
                    self.mupdate(word, (mx, f'_{pfx}'))
                    mx = False
        print(f'{sl - len(self.bases)} items combined for prefixes\n{len(self.bases)} remaining')

    """ def full_break(self):
        fails = []
        oc2 = {}
        agroup = [x for x in self.achems if x not in self.cdt]
        bgroup = sorted(self.t1, key=lambda x: 1/len(x))
        for x in self.chems:
            hold = [(x[len(y):], [self.t1[y]] if ' ' not in self.t1[y] else self.t1[y].split()) for y in bgroup if x.startswith(y)]
            cands = []
            while hold:
                wr, wls = hold.pop()
                lwr = len(wr)
                for y in [y for y in self.cdt if wr.startswith(y)]:
                    if len(y) == lwr:
                        cands.append(tuple([*wls, *self.cdt[wr].split()]) if ' ' in self.cdt[wr] else tuple([*wls, self.cdt[wr]])) 
                    else:
                        hold.append((wr[len(y):], [*wls, *self.cdt[y].split()]) if ' ' in self.cdt[y] else (wr[len(y):], [*wls, self.cdt[y]]))

                for y in [y for y in agroup if wr.startswith(y)]:
                    if len(y) == lwr: cands.append(tuple([*wls, y]))
                    else: hold.append((wr[len(y):], [*wls, y]))

                for i, pos in enumerate((self.t1, self.t2, self.t3)):
                    for y in [y for y in pos if wr.startswith(y)]:
                        if lwr == len(y) and i != 2: continue
                        if i == 2 and y in self.t5 and len(y) < lwr: continue
                        if i == 0 and y in self.t4: continue
                        if len(y) == lwr: cands.append(tuple([*wls, *pos[y].split()]) if ' ' in pos[y] else tuple([*wls, pos[y]]))
                        else: hold.append((wr[len(y):], [*wls, *pos[y].split()]) if ' ' in pos[y] else (wr[len(y):], [*wls, pos[y]]))
            if not cands: fails.append(x)
            else: oc2[x] = tuple(set(cands))
        return oc2, fails """

    #Global tests
    def parse_wfile(self, fid, mode=0):
        parts, consume = set(), set()
        with open(fid, 'rt') as f:
            if mode == 0:
                for x in [x.strip().split() for x in f.readlines()]:
                    if x[-1] != '|':
                        consume.add(x[1])
                        x.pop()
                    parts.add(x[0])
                    for y in x[2:]: parts.add(y)
            elif mode == 1:
                with open(fid, 'rt') as f:
                    for x in [x.strip().split() for x in f.readlines()]:
                        parts.add(x[0])
                        for y in x[1:]: consume.add(y)
        return parts, consume

    def file_check(self):
        #Use for deconf
        roots = self.get('roots')
        rwords = self.get('b_remove')
        awords = self.get('b_add')
        for x in [x for x in rwords if x in roots]: print(f'{x} root also in rlist')
        for x in [x for x in awords if x in rwords]: print(f'{x} added word in rlist')
        mafxs = {x.strip(): 0 for x in self.get('affixes')}
        parts, consume = set(), set()
        hist = {}
        read = set()
        for file in self.wfiles:
            if file.startswith('syn'): fgrp = 1
            else: fgrp = 0
            npts, ncns = self.parse_wfile(self.files[file], fgrp)
            for y in ncns:
                if y[-1] == '+':
                    y = y[:-1]
                    read.add(y)
                if y in consume and hist[y] != file and y not in read: print(f'{y} target_from {file} reduced_in {hist[y]}')
                elif y in consume and y in read: read.remove(y)
                else: consume.add(y)
                if y in hist:
                    if isinstance(hist[y], str): hist[y] = [hist[y], file]
                    elif isinstance(hist[y], list): hist[y].append(file)
                else: hist[y] = file
                if '_' in y:
                    if y not in mafxs:
                        print(f'{y} new source affix')
                        mafxs[y] = 1
                    else: mafxs[y] += 1
            for y in npts:
                if y in consume and hist[y] != file and y not in read: print(f'{y} component_from {file} reduced_in {hist[y]}')
                elif y in consume and y in read: read.remove(y)
                else: parts.add(y)
                if '_' in y:
                    if y not in mafxs:
                        print(f'{y} new component affix ')
                        mafxs[y] = 1
                    else: mafxs[y] += 1

    def nfcheck2(self):
        roots = self.get('roots')
        rwords = self.get('b_remove')
        awords = self.get('b_add')
        for x in [x for x in rwords if x in roots]: print(f'{x} root also in rlist')
        for x in [x for x in awords if x in rwords]: print(f'{x} added word in rlist')
        mafxs = {x.strip(): 0 for x in self.get('affixes')}
        hist = {}
        for x in self.get('cmls'):
            if x[1] in hist: print(f'{x[1]} in {hist[x[1]]} already consumed')
            if x[0] == 'S':
                for y in x[2:]:
                    if y[-1] == '+':
                        if y[:-1] in hist: print(f'conflict {y[:-1]} in {x}\n with {hist[y[:-1]]}')
                    else:
                        if '_' in y:
                            if y not in mafxs: print(f'New affix {y}')
                            mafxs[y] += 1
                            continue
                        if y in hist: print(f'conflict {y} in {x}\n with {hist[y]}')
                        else: hist[y] = x
            elif x[0] == 'M':
                if x[-1] == '|':
                    if x[2] in hist: print(f'conflict {x[2]} in {x}\n with {hist[x[2]]}')
                    x = x[:-1]
                    sid = 3
                else: sid = 2
                for y in x[sid:]:
                    if '_' in y:
                        if y not in mafxs: print(f'New affix {y}')
                        mafxs[y] += 1
                        continue
                    if y in hist: print(f'conflict {y} in {x}\n with {hist[y]}')
                    else: hist[y] = x
            else: print(f'Invalid command {x}')

    def nfcheck(self):
        roots = self.get('roots')
        rwords = self.get('b_remove')
        awords = self.get('b_add')
        for x in [x for x in rwords if x in roots]: print(f'{x} root also in rlist')
        for x in [x for x in awords if x in rwords]: print(f'{x} added word in rlist')
        mafxs = {x.strip(): 0 for x in self.get('affixes')}
        hist = {}
        syndex = {}
        for x in self.get('cmls'):
            if '_' in x[1]:
                if x[1] not in mafxs: print(f'New affix {x[1]}')
                mafxs[x[1]] += 1
            elif x[1] in hist: print(f'{x[1]} in {hist[x[1]]} already consumed')

            if x[0] == 'S':
                for y in x[1:]:
                    if y[-1] == '+': y = y[:-1]

                    if not y in syndex: syndex[y] = [x]
                    else:
                        print(f'{x} syn_conflict {syndex[y]}')
                        syndex[y].append(x)

            for i, y in enumerate(x[2:]):
                if '|' in y: continue
                if '_' in y:
                    if y not in mafxs: print(f'New affix {y}')
                    mafxs[y] += 1
                    continue
                if (i == 0 and x[-1] == '|'): add = False
                elif x[0] == 'M' and i > 0: add = False
                elif y[-1] == '+':
                    add = False
                    y = y[:-1]
                else: add = True
                
                if y in hist:
                    print(f'conflict {y} ~ {x}\nwith\t\t {hist[y]}')
                    if add:
                        if isinstance(hist[y][0], str): hist[y] = [hist[y], x]
                        else: hist[y].append(x)
                if add: hist[y] = x

    def tests(self, fine=False, plain=False):
        built = self.basis
        rck = set()
        tmp = {}
        ups = list(self.get('uniqs'))
        ups.extend(self.get('propnouns'))

        for x in self.rgen['cmls']:
            if x[-1] == '|': continue
            if x[0] == 'S':
                for y in x[2:]:
                    y = y.strip('+')
                    if y not in tmp: tmp[y] = ' '.join(x)
                    elif isinstance(tmp[y], list): tmp[y].append(' '.join(x))
                    else: tmp[y] = [tmp[y], ' '.join(x)]
                    rck.add(y)
                rck.add(x[1])
            else:
                if x[1] not in tmp: tmp[x[1]] = ' '.join(x)
                elif isinstance(tmp[x[1]], list): tmp[x[1]].append(' '.join(x))
                else: tmp[x[1]] = [tmp[x[1]], ' '.join(x)]
                if x[1] in ups: print(f'Unique word reduced: {" ".join(x)}')
                for y in x[1:]: rck.add(y)
        for x in tmp.items():
            if isinstance(x[1], list):
                print(f'Duplicate command source word:')
                for y in x[1]: print(y)
        for x in self.rgen['cmls']:
            for y in x[1:]:
                if y in self.rgen['rem']:
                    print(y if plain else f'Command component in remove | {y} | {" ".join(x)}')

        if plain: print("Add word already in basis")
        ad = self.rgen['add']
        for x in ad:
            if x in self.basis:
                print(x if plain else f'Add word already in basis | {x}')
                built.add(x)
        """ if plain: print("Root word not in basis")
        rt = self.rgen['rls']
        for x in rt:
            if x not in built:
                if fine: print(x if plain else f'Root word not in basis | {x}')
                built.add(x) """
        if plain: print("PNoun word not in basis")
        pn = self.get('propnouns')
        for x in pn:
            if x not in built:
                if fine: print(x if plain else f'PNoun word not in basis | {x}')
                built.add(x)
        if plain: print("Latgre word not in basis")
        lg = self.get('latgre')
        for x in lg:
            if x not in built:
                if fine: print(x if plain else f'Latgre word not in basis | {x}')
                built.add(x)
        if plain: print("Unique word not in basis")
        uniq = self.get('uniqs')
        for x in uniq:
            if x not in built:
                if fine: print(x if plain else f'Unique word not in basis | {x}')
                built.add(x)
        if plain: print("Remove words utilized")
        for x in self.rgen['rem']:
            if x in rck: print(f'Removed word used | {x}')
            if x not in built: continue
            if x in ad: print(x if plain else f'Add word in remove | {x}')
            #if x in rt: print(x if plain else f'Root word in remove | {x}')
            if x in pn: print(x if plain else f'PNoun word in remove | {x}')
            if x in lg: print(x if plain else f'Latgre word in remove | {x}')
            if x in uniq: print(x if plain else f'Unique word in remove | {x}')

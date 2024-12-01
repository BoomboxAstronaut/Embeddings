
from .Word import Word
from pickle import load
from os import rename
from time import time
from collections import Counter

class Lexicon:
    #('', ''), ('', ''), 
    dbl = 'bdglmnprt'
    pos = 'nvadreci'

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, k):
        if k in self.bt:
            return self.media[self.bt[k][0]]
        else: raise ValueError(f'{k} not in tracker')

    def __setitem__(self, k, v):
        self.media[k] = v
        self.bt[k] = [k]

    def __repr__(self):
        return self.bases.__repr__()

    def __init__(self, wlst, wmode=False):
        self.wmode, self.block = wmode, False
        self.bases = set(sorted(wlst, key=lambda x: (len(x), x))[::-1])
        self.media = {x: Word(x) for x in wlst}
        self.bt = {x: [x] for x in wlst}
        self.at = {}
        self.cut = 'absc abst acc acq aff agg app arr ass att coll corr eff emb emp em eb ed eg ej el ep er ev ill irr imm imb imp occ opp succ suff sugg summ supp sus syll sys'.split()
        self.wfiles = ['man_1', 'syn_1', 'rep_pl', 'rep_pr', 'rep_pa', 'rep_a', 'rep_m', 'man_2', 'syn_2']
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
                'cmls': r'v0\roots\_cmls',
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
                'man_1': r'v0\roots\old\_pre_manual',
                'man_2': r'v0\roots\old\_manual',
                'master': r'v1\master_list',
                'measures': r'v1\measures',
                'medicine': r'v1\medic',
                'nouns': r'v1\nouns',
                'numerals': r'v1\numerals',
                'particles': r'v1\particles',
                'phys': r'v1\phys',
                'pl_es': r'v1\plural_es',
                'pl_ies': r'v1\plural_ies',
                'preposts': r'v1\preposts',
                'pronouns': r'v1\pronouns',
                'rep_a': r'v0\misc\_adj_reps',
                'rep_m': r'v0\misc\_member_reps',
                'rep_pa': r'v0\misc\_past_reps',
                'rep_pf': r'v0\affixes\_prefix_reps',
                'rep_pl': r'v0\misc\_plural_reps',
                'rep_pr': r'v0\misc\_preprog_reps',
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
        self.cands = {}

        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\nests2', 'rb') as f:
            self.fnest, self.bnest = load(f)
        with open(f'{self.files["affixes"]}', 'rt', encoding='utf8') as f:
            self.affixes = {x.strip() for x in f.readlines()}
            for x in self.affixes:
                self[x] = Word(x)
                self[x].nmut = 2
            self.prex = sorted([x.strip('_') for x in self.affixes if x[0] == '_'])[::-1]
            self.sufx = sorted([x.strip('_') for x in self.affixes if x[-1] == '_'])[::-1]
        with open(f'{self.files["afx_prex"]}', 'rt', encoding='utf8') as f:
            self.afx_ignore = {}
            for x in f.readlines():
                x = x.strip().split()
                self.afx_ignore[x[0]] = [y.strip('_') for y in x[1:]]
        self.pdc = {'absc': 'ab', 'abst': 'ab', 'emb': 'en', 'emp': 'en', 'opp': 'ob', 'occ': 'ob', 'of': 'ob'}
        for x in 'acc aff agg app arr ass att'.split(): self.pdc[x] = 'ad'
        for x in 'comb comm comp coll corr'.split(): self.pdc[x] = 'con'
        for x in 'imb imm imp irr ill'.split(): self.pdc[x] = 'in'
        for x in 'eff ecc eb ed eg ej el em er ep ev e'.split(): self.pdc[x] = 'ex'
        for x in 'sys syll sym'.split(): self.pdc[x] = 'syn'
        for x in 'succ suff sugg supp summ sus'.split(): self.pdc[x] = 'sub'
        rts = self.get('roots')
        for x in rts:
            if x not in self.media: self[x] = Word(x)
            self.media[x].nmut = 1

        rms = [z for z in self.get('b_remove') if z not in rts]
        ras = [z for z in self.get('b_add') if z not in rms]
        self.rgen = {'rls': rts, 'rem': rms, 'add': ras}
        self.bls = self.get('block')
        self.kes, self.ves, self.ckls = {}, {}, {}
        print(f'{len(self.bases)} Words')

        if self.wmode:
            self.rgen['cmls'] = [tuple(x) for x in self.get('cmls')]
            for x in self.rgen['cmls']: self.cread(x)
        else:
            self.manual(f'{self.files["man_1"]}')
            self.sync(f'{self.files["syn_1"]}')
            fulmod = []
            for x in self.wfiles: fulmod.extend([tuple(y) for y in self.get(x)])
            self.rgen['cmls'] = fulmod


    def check(self, word):
        if self.block and word in self.bls: return False
        if word in self.bt: return True

    def update(self, sink: str, src: str, afx=None, cat='', remove=True, dbg=False):
        if src not in self.media:
            raise ValueError(f'Cannot merge {src} into {sink}. Source {src} merged into {self.bt[src][0]}')
        if (not afx or afx[-1] != '|') and (self.media[src].nmut not in (0, 2, 4) or (self.media[src].nmut == 2 and self.media[sink].nmut not in (1, 2))):
            #if command has no affixes or command is a homonym: and src is a root/immutable, or src is a afx when sink is not a root/afx 
            raise ValueError(f'Cannot merge {src} into {sink} code: {self.media[src].nmut}')
        if (afx and afx[-1] != '|') or not afx:
            if isinstance(afx, (tuple, list, set)): self.ckls[src] = (sink, src, *afx)
            elif isinstance(afx, str): self.ckls[src] = (sink, src, afx)
            else: self.ckls[src] = (sink, src, 'S')
            if src in self.kes and self.kes[src] != sink: raise ValueError(f'Conf: {src} -> {sink} !|! {src} -> {self.kes[src]}')
            self.kes[src] = sink
            self.ves[src] = sink
        sink = self[sink].w
        if dbg and len(sink) < 4: print(f'M {sink} {src} {afx if afx[1:] not in self.pdc else self.pdc[afx[1:]]}')
        if not afx or afx[-1] != '|':
            self.media[sink].mods.extend(self.media[src].mods)
            self.media[sink].alias.extend(self.media[src].alias)
            self.media[sink].forms.extend(self.media[src].forms)
            self.media[sink].all.extend(self.media[src].all)
            self.media[sink].all.append(src)
        if isinstance(afx, (tuple, list, set)):
            for x in afx:
                if x and x != '|': self[x].mods.append(sink)
        elif isinstance(afx, str): self[afx].mods.append(sink)
        if not afx or afx[-1] != '|':
            if remove: self.media.pop(src)
            if src in self.bases: self.bases.remove(src)
            self.update_track(sink, src)
        if cat: self.media[sink][cat].append(src)

    def update_track(self, sink, src):
        self.bt[sink].extend([x for x in self.bt[src] if x not in self.bt[sink]])
        pack = self.bt[sink]
        for x in pack:
            self.bt[x] = self.bt[sink]

    def ls(self, target):
        if target in self.media: print(self.media[target])
        elif target in self.at: print('DROPPED:', self.at[target])
        else: print('Not Found')

    def get(self, group=None, intersects=False):
        if not group:
            for x in sorted(self.files): print(x)
        if group not in self.files: raise ValueError(f'{group} invalid group name')
        with open(self.files[group], 'rt', encoding='utf8') as f:
            if group.startswith('cmls'): return [x.strip().split() for x in f.readlines()]
            elif group.startswith('syn'): return [('S', *x.strip().split()) for x in f.readlines()]
            elif group.startswith('man') or group.startswith('rep'): return [('M', *x.strip().split()) for x in f.readlines()]
            elif intersects: return [x.strip() for x in f.readlines() if x.strip() in intersects]
            else: return [x.strip() for x in f.readlines()]

    def drop(self, k):
        if k not in self.bt: raise ValueError(f'{k} drop target not found')
        root = self.bt[k][0]
        if k in self.bases: self.bases.remove(k)
        for x in [y for y in self.bt[k] if y in self.bt]: self.at[x] = self.bt.pop(x)
        if k in self.media: self.at[root] = self.media.pop(k)

    def reload(self):
        hold = [tuple(x) for x in self.get('cmls')]
        self.rgen['cmls'].extend([x for x in hold if x not in self.rgen['cmls']])
        self.rgen['rem'].extend([x for x in self.get('b_remove') if x not in self.rgen['rem']])
        self.rgen['add'].extend([x for x in self.get('b_add') if x not in self.rgen['add']])
        self.csort()


    def cread(self, command, debug=False):
        sink = command[1]
        if sink not in self.media and self.check(sink):
            if debug: print(f'{sink} reduced into {self.bt[sink][0]}')
        if not self.check(sink):
            if debug: print(f'Creating transitive sink {sink}')
            self[sink] = Word(sink)
            self[sink].nmut = 4
        if command[0] == 'S':
            for y in command[2:]:
                if '+' in y or '_' in y:
                    rem = False
                    y = y.strip('+')
                else: rem = True
                if not self.check(y):
                    if debug: print(f'Creating transitive source {y} {[p for p in self.rgen["cmls"] if y in p or y+"+" in p]}')
                    self[y] = Word(y)
                self.update(sink, y, cat='alias', remove=rem)
                if rem:
                    self.bt.pop(y)
                    self.at[y] = sink
                elif '_' in y: self.media[y].nmut = 2
                else: self.media[y].nmut = 3
        elif command[0] == 'M':
            src = command[2]
            if not self.check(src):
                if debug: print(f'Creating transitive source {src} {[p for p in self.rgen["cmls"] if src in p or src+"+" in p]}')
                self[src] = Word(src)
                self[src].nmut = 4
            self.update(sink, src, command[3:], cat='forms')
        else: raise ValueError(f'Invalid command {command}')

    def cfind(self, word, src=0, chain=False):
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

    def ucodir(self, reps=5):
        for _ in range(reps):
            for x in self.ves.items():
                if x[1] in self.kes:
                    self.ves[x[0]] = self.kes[x[1]]

    def recon(self, word):
        targ = word
        forma = ['ROOT']
        ipoint = 1
        jpoint = 0
        while targ in self.kes:
            for y in self.ckls[targ][2:]:
                if y[0] == '_':
                    forma.insert(jpoint, y)
                    ipoint += 1
                    jpoint += 1
                elif y[-1] == '_': forma.insert(ipoint, y)
                else:
                    forma.insert(jpoint, y)
                    ipoint += 1
                    jpoint += 1
            targ = self.ckls[targ][0]
        forma[forma.index('ROOT')] = targ
        if any(z in self.kes for z in forma):
            found = True
            while found:
                for x in forma:
                    if x in self.kes:
                        rep = self.recon(x)
                        kpoint = forma.index(x)
                        forma.pop(kpoint)
                        while rep:
                            forma.insert(kpoint, rep.pop())
                        break
                else: found = False
        return forma

    def csort(self, commands=None):
        if not commands:
            commands = self.rgen['cmls'][::-1]
            decom = True
        else: decom = False
        rads = [x for x in commands if x[-1] == '|']
        rad_src = {x[2] for x in rads}
        pre_rads = [x for x in commands if x[1] in rad_src]
        syns = [x for x in commands if x[0] == 'S']
        sources = [[y for y in x[2:] if not '+' in y] if x[0] == 'S' else ([] if '|' in x else [x[2]]) for x in commands]
        for i, x in enumerate(commands):
            if '_' in x[2] or '|' in x: continue
            if x[0] == 'M':
                for y in sources[i+1:]:
                    if x[2] in y: raise ValueError(f'CONFLICT {x} {y}')
            elif x[0] == 'S':
                for z in sources[i]:
                    for y in sources[i+1:]:
                        if z in y: raise ValueError(f'CONFLICT {x} {y}')
        hold, lasthold = [], []
        while True:
            sources = [[y for y in x[2:] if not '+' in y] if x[0] == 'S' else ([] if '|' in x else [x[2]]) for x in commands]
            for i, x in enumerate(commands):
                if '_' in x[1] or '|' in x: continue
                for y in sources[i+1:]:
                    if not y or x[1] not in y: continue
                    hold.append(x)
                    break
            if not hold: break
            elif hold == lasthold: raise ValueError(f'CONFLICT 2 {hold}')
            else: lasthold = hold.copy()
            for x in hold: commands.remove(x)
            commands.extend(hold)
            hold = []
        if decom: self.rgen['cmls'] = commands[::-1]
        else: return commands[::-1]

    def regen(self):
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
        if self.cfind(word): raise ValueError(f'{word} in rule {self.cfind(word)}')
        if word in self.rgen['rls']: self.rgen['rls'].remove(word)
        if word in self.rgen['add']: self.rgen['add'].remove(word)
        if word not in self.rgen['rem']: self.rgen['rem'].append(word)

    def aw(self, word, root=False):
        if word in self.rgen['rem']: self.rgen['rem'].remove(word)
        if word not in self.rgen['add'] and word not in self.bt: self.rgen['add'].append(word)
        if root and word not in self.rgen['rls']: self.rgen['rls'].append(word)

    def crep(self, ow, nw):
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
        if isinstance(affixes, str): affixes = (affixes,)
        self.rgen['cmls'].append(('M', sink, source, *affixes))

    def nscom(self, sink, source):
        sp = f'{sink}+'
        found = False
        for i, x in enumerate(self.rgen['cmls']):
            if x[0] != 'S': continue
            if sink in x or sp in x:
                self.rgen['cmls'][i] = (*x, source)
                found = True
        if not found: self.rgen['cmls'].append(('S', sink, source))


    @property
    def roots(self):
        return [x for x in self.media if self.media[x].nmut == 1]

    def rep_sfx(self, word, al, t='', r='', vd='') -> None | str:
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
            if self.check(f'{word[:-(al+2)]}f'): return f'{word[:-(al+2)]}f'
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
                if not self.check(word[:-i]): continue
                return word[:-i]

    def rep_pfx(self, word, pfx) -> str:
        if pfx in self.afx_ignore and any(word.startswith(z) for z in self.afx_ignore[pfx]): return
        elif pfx in self.cut: return word[len(pfx)-1:]
        else: return word[len(pfx):]

    def rep_afx(self, word, al, t='', r='', vd='', tx='', bridges=dict(), pref=False):
        bridges[''] = ''
        for bk in bridges:
            rep = self.rep_sfx(word, al, t, r, vd)
            if not rep or len(rep)-len(bk) < 3: continue
            if not rep.endswith(bk): continue
            rep = f'{rep[:-len(bk) if bk else None]}{bridges[bk]}'
            if pref and not tx:
                for prefix in sorted([x for x in self.prex if rep.startswith(x) and len(rep)-len(x) > 2]):
                    pfx_rep = self.rep_pfx(rep, prefix)
                    if self.check(pfx_rep): return (pfx_rep, f'{prefix if prefix not in self.pdc else self.pdc[prefix]}')
                    elif prefix == 'ex' and self.check(f's{pfx_rep}'): return (f's{pfx_rep}', f'{prefix if prefix not in self.pdc else self.pdc[prefix]}')

            if pref:
                pfxs = [(self.rep_pfx(rep, z), z) for z in sorted(self.prex) if rep.startswith(z) and len(rep)-len(z) > 2 and self.rep_pfx(rep, z)]
                pfxs.append((rep, ''))
            if tx == 'v':
                for v in ('', 'y', 'e', 'us', 'is', 'os', 'a', 'um', 'on', 'io', 'o'):
                    if word[-al:] == v: continue
                    if rep[-1] in 'iuo':
                        if not pref:
                            if f'{rep}{v}' in self.bt: return f'{rep}{v}'
                            if f'{rep}s{v}' in self.bt: return f'{rep}s{v}'
                            if rep[-1] == 'u' and f'{rep}m{v}' in self.bt: return f'{rep}m{v}'
                            if rep[-1] == 'o' and f'{rep}n{v}' in self.bt: return f'{rep}n{v}'
                        else:
                            for pgp in pfxs:
                                if f'{pgp[0]}{v}' in self.bt: return (f'{pgp[0]}{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                if f'{pgp[0]}s{v}' in self.bt: return (f'{pgp[0]}s{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                if rep[-1] == 'u' and f'{pgp[0]}m{v}' in self.bt: return (f'{pgp[0]}m{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                if rep[-1] == 'o' and f'{pgp[0]}n{v}' in self.bt: return (f'{pgp[0]}n{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                    else:
                        if not pref:
                            if f'{rep}{v}' in self.bt: return f'{rep}{v}'
                        else:
                            for pgp in pfxs:
                                if f'{pgp[0]}{v}' in self.bt: return (f'{pgp[0]}{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
            elif tx == 'vx':
                if rep[-2:] == 'ct':
                    if not pref:
                        for v in ('', 'y', 'e', 'us', 'is', 'os', 'a', 'um', 'on', 'io', 'o'):
                            if word[-al:] == v: continue
                            if f'{rep[:-2]}{v}' in self.bt: return f'{rep[:-2]}{v}'
                            elif f'{rep[:-2]}s{v}' in self.bt: return f'{rep[:-2]}s{v}'
                            elif f'{rep[:-2]}g{v}' in self.bt: return f'{rep[:-2]}g{v}'
                            elif f'{rep[:-2]}x{v}' in self.bt: return f'{rep[:-2]}x{v}'
                    else:
                        for pgp in pfxs:
                            for v in ('', 'y', 'e', 'us', 'is', 'os', 'a', 'um', 'on', 'io', 'o'):
                                if word[-al:] == v: continue
                                if f'{pgp[0][:-2]}{v}' in self.bt: return (f'{pgp[0][:-2]}{v}', pgp[1])
                                elif f'{pgp[0][:-2]}s{v}' in self.bt: return (f'{pgp[0][:-2]}s{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                elif f'{pgp[0][:-2]}g{v}' in self.bt: return (f'{pgp[0][:-2]}g{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                elif f'{pgp[0][:-2]}x{v}' in self.bt: return (f'{pgp[0][:-2]}x{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                elif rep[-1] in 'ct':
                    if not pref:
                        for v in ('', 'y', 'e', 'us', 'is', 'os', 'a', 'um', 'on', 'io', 'o'):
                            if word[-al:] == v: continue
                            if f'{rep[:-1]}s{v}' in self.bt: return f'{rep}s{v}'
                            elif f'{rep[:-1]}g{v}' in self.bt: return f'{rep}g{v}'
                            elif f'{rep[:-1]}x{v}' in self.bt: return f'{rep}x{v}'
                    else:
                        for pgp in pfxs:
                            for v in ('', 'y', 'e', 'us', 'is', 'os', 'a', 'um', 'on', 'io', 'o'):
                                if word[-al:] == v: continue
                                if f'{pgp[0][:-1]}s{v}' in self.bt: return (f'{pgp[0][:-1]}s{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                elif f'{pgp[0][:-1]}g{v}' in self.bt: return (f'{pgp[0][:-1]}g{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                                elif f'{pgp[0][:-1]}x{v}' in self.bt: return (f'{pgp[0][:-1]}x{v}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
            elif tx == 'xv':
                #Vowel expansion: proclamation proclaim
                if rep[-1] in 'aeiou':
                    for v in 'aeiou':
                        if not pref:
                            if f'{rep[:-1]}{v}{rep[-1]}' in self.bt: return f'{rep[:-1]}{v}{rep[-1]}'
                        else:
                            for pgp in pfxs:
                                if f'{pgp[0][:-1]}{v}{pgp[0][-1]}' in self.bt: return (f'{pgp[0][:-1]}{v}{pgp[0][-1]}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                elif rep[-2] in 'aeiou':
                    for v in 'aeiou':
                        if not pref:
                            if f'{rep[:-2]}{v}{rep[-2:]}' in self.bt: return f'{rep[:-2]}{v}{rep[-2:]}'
                        else:
                            for pgp in pfxs:
                                if f'{pgp[0][:-2]}{v}{pgp[0][-2:]}' in self.bt: return (f'{pgp[0][:-2]}{v}{pgp[0][-2:]}', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
            elif tx == 'vc':
                #ul il er ending rearrangement
                if not pref:
                    if rep[-1] == 'l' and rep[-2] in 'ui' and f'{rep[:-2]}le' in self.bt: return f'{rep[:-2]}le'
                    if rep[-1] == 'r' and rep[-2] not in 'aeiouy' and f'{rep[:-1]}er' in self.bt: return f'{rep[:-1]}er'
                else:
                    for pgp in pfxs:
                        if pgp[0][-1] == 'l' and pgp[0][-2] in 'ui' and f'{pgp[0][:-2]}le' in self.bt: return (f'{pgp[0][:-2]}le', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
                        if pgp[0][-1] == 'r' and pgp[0][-2] not in 'aeiouy' and f'{pgp[0][:-1]}er' in self.bt: return (f'{pgp[0][:-1]}er', pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])
            else:
                if not pref:
                    if self.check(rep): return rep
                else:
                    for pgp in pfxs:
                        if self.check(pgp[0]): return (pgp[0], pgp[1] if pgp[1] not in self.pdc else self.pdc[pgp[1]])

    def rep_pafx(self, word, al):
        if self.check(word[al:]): return word[al:]
        elif word[al:][0] in 'oiau' and self.check(word[al+1:]): return word[al+1:]


    def phx(self, word, v=0):
        last = ''
        if not v:
            while word and word[-1] not in 'aeiouy':
                last = f'{word[-1]}{last}'
                word = word[:-1]
        else:
            while word and word[-1] in 'aeiouy':
                last = f'{word[-1]}{last}'
                word = word[:-1]
        return last

    def phon(self, word, pc=1):
        wl = len(word)
        key = 0
        for i in range(1, wl-2):
            x = word[-(i+2):-i]
            if x[0] not in 'aeiouy' and x[1] in 'aeiouy':
                key = i+1
                pc -= 1
            if not pc: break
        return word[-key:]

    def make_cands(self):
        cnt = Counter()
        for x in [x[1] for x in self.rgen['cmls'] if x[0] != 'S']:
            cnt[x] += 1
        cands = [x[0] for x in cnt.most_common() if len(x[0]) > 3 and not x[0].endswith('ion') and (x[1] >= 7 or (x[1] >= 3 and any(x[0].endswith(z) for z in ('os', 'is', 'us', 'as', 'um', 'a', 'er', 'or', 'on')))) and (not self.check(x[0]) or any(x[0].endswith(z) for z in ('os', 'is', 'us', 'as', 'um', 'a', 'er', 'or', 'on')))]
        cands.extend([x for x in self.roots if len(x) > 4 and not x.endswith('ica') and len(x) < 9 and x not in cands and self.phon(x) in ('os', 'is', 'us', 'as', 'um', 'a', 'or', 'on')])
        for x in 'meter equis price termus border dica'.split():
            cands.append(x)
        cands = set(cands)
        for x in 'allos alumnus album amazon bachelor bellum filia canvas areola deacon cavus frenum cover equus mora poena velum water zircon reason season sermon terra thalamus thallos thrombus thymus titulus vanadis bolos callus canton capon carina colon confer consider contra cotton cretus decis decor defer demon director divus dominus endos equitum florida fluoros formosa formula gallon gener heron hirsutus homos horos humerus humor humus ignis imperium indica inter janitor legitimus magnesia marathon maritimus mason media member meris millus minister mitos monos montanus moron motor munera nanos neighbor newton nitron nodosus obscurus oleis over paleos palmatus para parameter paris parvus pastus pater penis planum porta pubis rector refer religious retis sacer septum seros serum sinus sister sopor tailor talon tenis tennis tenus teras terminus termis terror testis tomos transfer tribos tris uranus valgus valor venus viola waris weapon'.split():
            cands.remove(x)
        
        for x in cands.copy():
            if any(x.endswith(z) for z in ('os', 'is', 'us', 'as', 'um', 'or', 'er', 'on')):
                if self.check(x[:-2]): cands.remove(x)
            elif self.check(x[:-1]): cands.remove(x)

        outp = {}
        for x in cands:
            if x[-2:] in ('os', 'is', 'us', 'as', 'um', 'or', 'er', 'on'):
                if x[:-2] in outp: print('failed', x[:-2], x, outp[x[:-2]])
                outp[x[:-2]] = x
            if x[-2:] == 'is':
                if f'{x[:-2]}e' in outp: print('failed', f'{x[:-2]}e', x, outp[f'{x[:-2]}e'])
                outp[f'{x[:-2]}e'] = x
            if x[:-1] in outp: print('failed', x[:-1], x, outp[x[:-1]])
            outp[x[:-1]] = x
        return outp


    def rep_comp(self, word):
        for y in self.bnest[word]:
            front = word[:len(word)-len(y)]
            if len(y) < 3 and y not in self.sufx: continue
            if len(front) < 3 and y not in self.sufx: continue
            if front in self.fnest[word]:
                return (front, y)
            elif f'{front}{y[0]}' in self.fnest[word]:
                return (f'{front}{y[0]}', y)
            elif len(front) > 1 and front[-1] in 'aoiu' and front[:-1] in self.fnest[word]:
                return (f'{front[:-1]}', y)
            elif len(front) > 1:
                for l in 'eoaiu':
                    if f'{front}{y[0]}{l}' != word and f'{front}{y[0]}{l}' in self.fnest[word]:
                        return (f'{front}{y[0]}{l}', y)
                    elif front[-1] in 'aeiou' and f'{front[:-1]}{l}' in self.fnest[word]:
                        return (f'{front[:-1]}{l}', y)
                    elif f'{front}{l}' in self.fnest[word]:
                        return (f'{front}{l}', y)

    def unisplit(self, word, crp=None, ends=None):
        if not crp: crp = self.bt
        if not ends: ends = crp
        out = [(x, word[len(x):]) for x in crp if word.startswith(x) and word[len(x):] in ends]
        if out: return out


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
                            self.update(y, x, 's_', 'forms')
                            self[y].noun = True
            with open(f'{self.files["pl_es"]}', 'rt', encoding='utf8') as f:
                pes = {x.strip() for x in f.readlines()}
                for x in [x for x in self.bases if x.endswith('es')]:
                    if x in pes:
                        y = self.rep_afx(x, 2)
                        if y:
                            self.update(y, x, 's_', 'forms')
                            self[y].noun = True
            with open(f'{self.files["rep_pl"]}', 'rt', encoding='utf8') as f:
                for x in [x.strip().split() for x in f.readlines()]:
                    self.update(x[0], x[1], x[2], 'forms')
                    self[x[0]].noun = True

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
                if pref: self.update(mx[0], x, (mx[1], 's_'), 'forms')
                else: self.update(mx, x, 's_', 'forms')
        print(f'{sl - len(self.bases)} items combined for plurals\n{len(self.bases)} remaining')

    def prpt_parse(self, pref=False):
        sl = len(self.bases)
        with open(f'{self.files["ign_pr"]}', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]
        if not pref and not self.wmode:
            with open(f'{self.files["rep_pr"]}', 'rt', encoding='utf8') as f:
                for x in [x.strip().split() for x in f.readlines()]:
                    self.update(x[0], x[1], x[2], 'forms')
                    self[x[0]].verb = True

        targets = [x for x in self.bases if len(x) > 5 and x.endswith('ing') and x not in igls]
        for x in targets:
            mx = False

            if (mx := self.rep_afx(x, 3, t='dbl', pref=pref)): pass
            elif (mx := self.rep_afx(x, 3, r='e', pref=pref)): pass
            elif (mx := self.rep_afx(x, 3, pref=pref)): pass
            elif (mx := self.rep_afx(x, 3, t='ck', pref=pref)): pass
            if mx:
                if pref:
                    self.update(mx[0], x, (mx[1], 'ing_'), 'forms')
                    self[mx[0]].verb = True
                else:
                    self.update(mx, x, 'ing_', 'forms')
                    self[mx].verb = True
        print(f'{sl - len(self.bases)} items combined for present progressives\n{len(self.bases)} remaining')

    def pt_parse(self, pref=False):
        sl = len(self.bases)
        with open(f'{self.files["ign_pa"]}', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]
        if not pref and not self.wmode:
            with open(f'{self.files["rep_pa"]}', 'rt', encoding='utf8') as f:
                for x in [x.strip().split() for x in f.readlines()]:
                    self.update(x[0], x[1], x[2], 'forms')
        
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
                if pref: self.update(mx[0], x, (mx[1], afx), 'forms')
                else: self.update(mx, x, afx, 'forms')
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
                        self.update(x[0], x[1], x[2], 'forms')

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
                self.update(pack[0], x[0], ('est_' if x[0].endswith('est') else 'er_'), 'forms')
                self.update(pack[0], x[1], ('est_' if x[1].endswith('est') else 'er_'), 'forms')

        targets = [x for x in self.bases if (len(x) > 4 and x.endswith('y')) and x not in igls]
        for x in targets:
            mx = False
            afx = f'{x[-2:]}_'

            if x.endswith('ly') and (mx := self.rep_afx(x, 2, vd={1: 'bcdefghklmnprstwordy'}, pref=pref)): pass
            elif x.endswith('ry') and (mx := self.rep_afx(x, 2, vd={1: 'cdeklnt'}, pref=pref)): pass
            elif x.endswith('ty') and (mx := self.rep_afx(x, 2, vd={1: 'elx'}, pref=pref)): pass
            elif x.endswith('bility') and (mx := self.rep_afx(x, 5, r='le', pref=pref)): pass
            elif x.endswith('cy') and (mx := self.rep_afx(x, 2, r='te', vd={1: 'a'}, pref=pref)): pass
            elif x.endswith('cy') and (mx := self.rep_afx(x, 2, r='t', vd={1: 'n'}, pref=pref)): pass
            elif (mx := self.rep_afx(x, 1, vd={1: 'dfghklmnprstwordz'}, pref=pref)): afx = 'y_'
            elif any(x.endswith(y) for y in ('ily', 'ary', 'ory', 'ity', 'ify')) and (mx := self.rep_afx(x, 3, r='e', vd={1: 'bcdgklmnprstvz'}, pref=pref)): pass
            elif (mx := self.rep_afx(x, 1, r='e', vd={1: 'bcdgklmnprstvz'}, pref=pref)): afx = 'y_'
            elif (mx := self.rep_afx(x, 1, t='dbl', vd={1: 'bdglmnpt'}, pref=pref)): afx = 'y_'
            elif (mx := self.rep_afx(x, 2, vd={0: 'r', 1: 'r', 2: 'u'}, pref=pref)): pass
            elif x.endswith('ically') and (mx := self.rep_afx(x, 4, pref=pref)): pass
            elif (x.endswith('arily') or x.endswith('sily')) and (mx := self.rep_afx(x, 3, r='y', pref=pref)): pass
            elif x.endswith('llary') and (mx := self.rep_afx(x, 4, pref=pref)): pass
            elif x.endswith('ary') and (mx := self.rep_afx(x, 3, vd={1: 'bdmnrt'}, pref=pref)): pass
            elif x.endswith('ily') and (mx := self.rep_afx(x, 3, vd={1: 'dhkmpt'}, pref=pref)): pass
            elif x.endswith('ily') and (mx := self.rep_afx(x, 3, t='dbl', vd={1: 'ndp'}, pref=pref)): pass
            elif x.endswith('ity') and (mx := self.rep_afx(x, 3, vd={1: 'cdelmnrtx'}, pref=pref)): pass
            elif x.endswith('ity') and (mx := self.rep_afx(x, 3, t='dbl', vd={1: 'lp'}, pref=pref)): pass
            elif x.endswith('ory') and (mx := self.rep_afx(x, 3, vd={1: 'st'}, pref=pref)): pass
            elif x.endswith('ery') and (mx := self.rep_afx(x, 3, t='dbl', vd={1: 'bgln'}, pref=pref)): pass

            if mx:
                if pref: self.update(mx[0], x, (mx[1], afx), 'forms')
                else: self.update(mx, x, afx, 'forms')

        print(f'{sl - len(self.bases)} items combined for adjectives \n{len(self.bases)} remaining')

    def en_parse(self, pref=False):
        sl = len(self.bases)
        igls = 'listen albumen haven molten pollen dozen graben midden garden baleen spleen careen lichen heathen token salen ramen somen rumen semen linen aspen siren warren paten marten solen'.split()
        """ for x in [x.strip().split() for x in f.readlines()]:
            self.update(x[0], x[1], x[2], 'forms') """

        targets = [x for x in self.bases if len(x) > 4 and x.endswith('en') and x not in igls and not any(x.endswith(y) for y in ('screen', 'gen', 'teen', 'seen'))]
        al = 2
        for x in targets:
            mx = False

            if (mx := self.rep_afx(x, al, t='dbl', pref=pref)): pass
            elif (mx := self.rep_afx(x, al, pref=pref)): pass
            elif (mx := self.rep_afx(x, al, t='e', pref=pref)): pass
            elif (mx := self.rep_afx(x, al, t='dbl', r='e', pref=pref)): pass
            if mx:
                if pref: self.update(mx[0], x, (mx[1], 'en_'), 'forms')
                else: self.update(mx, x, 'en_', 'forms')
        print(f'{sl - len(self.bases)} items combined for en suffix\n{len(self.bases)} remaining')

    def mbr_parse(self, pref=False):
        sl = len(self.bases)
        with open(f'{self.files["ign_m"]}', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]
        fls = 'meter water power flower polar'.split()
        if not pref and not self.wmode:
            with open(f'{self.files["rep_m"]}', 'rt', encoding='utf8') as f:
                for x in [x.strip().split() for x in f.readlines()]:
                    self.update(x[0], x[1], x[2], 'forms')

        al = 3
        targets = [x for x in self.bases if len(x) > 5 and (x.endswith('ian') or x.endswith('ist')) and not x in igls]
        for x in targets:
            mx = False

            if (mx := self.rep_afx(x, 1, r='m', pref=pref)): afx = 'ist_'
            elif (mx := self.rep_afx(x, al, r='y', pref=pref)): afx = f'{x[-3:]}_'
            elif x.endswith('scientist') and (mx := self.rep_afx(x, len('scientist'), r='science', pref=pref)): afx = 'ist_'
            elif x.endswith('tarian') and (mx := self.rep_afx(x, 5, r='y', pref=pref)): afx = 'ian_'
            elif x.endswith('ician') and (mx := self.rep_afx(x, 3, r='s', pref=pref)): afx = 'cian_'
            elif x.endswith('ician') and (mx := self.rep_afx(x, 5, pref=pref)): afx = 'cian_'
            elif x.endswith('ian') and (mx := self.rep_afx(x, al, t='m', pref=pref)): afx = 'ian_'
            elif (mx := self.rep_afx(x, al, pref=pref)): afx = f'{x[-3:]}_'
            elif (mx := self.rep_afx(x, al, t='e', pref=pref)): afx = f'{x[-3:]}_'
            elif (mx := self.rep_afx(x, al, t='dbl', pref=pref)): afx = f'{x[-3:]}_'
            elif (mx := self.rep_afx(x, al, r='ic', pref=pref)): afx = f'{x[-3:]}_'

            if mx:
                if pref: self.update(mx[0], x, (mx[1], afx), 'forms')
                else: self.update(mx, x, afx, 'forms')

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
                if pref: self.update(mx[0], x, (mx[1], f'{x[-al:]}_'), 'forms')
                else: self.update(mx, x, f'{x[-al:]}_', 'forms')

        targets = [x for x in self.bases if len(x) > 6 and x.endswith('ling') and not x in igls]
        for x in targets:
            mx = False
            if (mx := self.rep_afx(x, 4, pref=pref)):
                if pref: self.update(mx[0], x, (mx[1], 'ling_'), 'forms')
                else: self.update(mx, x, 'ling_', 'forms')

        print(f'{sl - len(self.bases)} items combined for membership \n{len(self.bases)} remaining')

    def sfx1_parse(self, pref=False):
        sl = len(self.bases)
        igls = 'magister manoeuvre parable liable capable arable sister amiable gullible malleable tangible tanginess semen talisman'.split()

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
                if pref: self.update(mx[0], x, (mx[1], afx), 'forms')
                else: self.update(mx, x, afx, 'forms')

        for tgt in (('woman', 'women'), ('man', 'men')):
            al = len(tgt[0])
            targets = [x for x in self.bases if len(x) > 5 and x not in igls and (x.endswith(tgt[0]) or x.endswith(tgt[1]))]
            if pref: targets = [x for x in targets if not any(x.endswith(y) for y in igls)]
            for x in targets:
                mx = False
                if (mx := self.rep_afx(x, al, pref=pref)): self.update(tgt[0], x, afx=mx, cat='forms')
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
                if self.check(mx) and len(mx) > 2:
                    if pfx in self.pdc: pfx = self.pdc[pfx]
                    print(f'M {mx} {word} _{pfx}')
                    self.update(mx, word, f'_{pfx}', cat='forms')
                    mx = False
        print(f'{sl - len(self.bases)} items combined for prefixes\n{len(self.bases)} remaining')


    def v2d_parse(self, pref=False):
        sl = len(self.bases)
        with open(f'{self.wdir}v0\\misc\\_verb_adv_reps', 'rt', encoding='utf8') as f:
            rpls = [x.strip().split() for x in f.readlines()]
        with open(f'{self.wdir}v0\\misc\\_verb_adv_ignore', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]
        if not pref:
            for x in rpls:
                self.update(x[0], x[1], x[2:], 'forms')
        fls = 'active drive ceptive ceive dive hive'.split()

        al = 3
        targets = [x for x in self.bases if len(x) > 3+al and x.endswith('ive') and (y not in x for y in fls) and x not in igls]
        if pref: targets = [x for x in targets if not any(x.endswith(y) for y in igls)]
        for x in targets:
            mx = False

            if (mx := self.rep_afx(x, al, pref=pref)): pass
            elif (mx := self.rep_afx(x, al, pref=pref)): pass
            elif (mx := self.rep_afx(x, al+1, r='de', vd={0: 's', 1: 'aeiou'}, pref=pref)): pass
            elif (mx := self.rep_afx(x, al+2, r='e', vd=(-5, 'it'), pref=pref)): pass
            elif (mx := self.rep_afx(x, al+2, vd=(-5, 'it'), pref=pref)): pass
            elif (mx := self.rep_afx(x, al+1, r='d', vd={0: 'sn'}, pref=pref)): pass
            elif (mx := self.rep_afx(x, al, pref=pref)): pass
            elif (mx := self.rep_afx(x, al, r='e', pref=pref)): pass
            elif (mx := self.rep_afx(x, al+2, r='e', vd=(-5, 'at'), pref=pref)): pass
            elif (mx := self.rep_afx(x, al+2, vd=(-5, 'at'), pref=pref)): pass
            elif (mx := self.rep_afx(x, al+2, r='y', vd=(-5, 'at'), pref=pref)): pass

            if mx:
                if pref: self.update(mx[0], x, (mx[1], 'ive_'), 'forms')
                else: self.update(self[mx].w, x, 'ive_', 'forms')
        print(f'{sl - len(self.bases)} items combined for plurals\n{len(self.bases)} remaining')

    def unq_parse(self, pref=False):
        #ed
        sl = len(self.bases)
        with open(f'{self.wdir}v0\\a', 'rt', encoding='utf8') as f:
            rpls = [x.strip().split() for x in f.readlines()]
        with open(f'{self.wdir}v0\\a', 'rt', encoding='utf8') as f:
            igls = [x.strip() for x in f.readlines()]
        if not pref:
            for x in rpls:
                self.update(x[0], x[1], cat='forms')
        fls = ''.split()

        al = 2
        for x in [x for x in self.bases if len(x) > 3+al and x.endswith('') and (y not in x for y in fls) and x not in igls]:
            mx = False
            
            if x.endswith('') and (mx := self.rep_afx(x, al, pref=pref)): pass
            elif x.endswith('') and (mx := self.rep_afx(x, al, pref=pref)): pass

            if mx: self.update(self[mx], x, x[-2:], 'forms')
                
        print(f'{sl - len(self.bases)} items combined for plurals\n{len(self.bases)} remaining')


    def sync(self, target=None, debug=False):
        with open(target, 'rt', encoding='utf8') as f:
            for x in [x.strip().split() for x in f.readlines()]:
                if x[0] not in self.media and self.check(x[0]):
                    if debug: print(f'{x[0]} reduced into {self.bt[x[0]][0]}')
                    continue
                elif self.check(x[0]):
                    if debug: print(f'Creating transitive sink {x[0]}')
                    self[x[0]] = Word(x[0])
                    self.media[x[0]].nmut = 4
                for y in x[1:]:
                    if '+' in y or '_' in y:
                        rem = False
                        y = y.strip('+')
                    else: rem = True
                    if not self.check(y):
                        if debug: print(f'Creating transitive source {y}')
                        self[y] = Word(y)
                    self.update(x[0], y, cat='alias', remove=rem)
                    if rem:
                        self.bt.pop(y)
                        self.at[y] = x[0]
                    elif '_' in y: self.media[y].nmut = 2
                    else: self.media[y].nmut = 3

    def manual(self, target=None, debug=False):
        with open(target, 'rt', encoding='utf8') as f:
            manuals = [x.strip().split() for x in f.readlines()]
            while manuals:
                man_t = {y for x in manuals for i, y in enumerate(x) if i != 1}
                man_v = [x for x in manuals if x[1] in man_t]
                for x in manuals:
                    if x in man_v: continue
                    sink = x[0]
                    src = x[1]
                    if x[-1] == '|':
                        readd = True
                        x.pop()
                    else: readd = False
                    if not self.check(sink):
                        if debug: print(f'Creating transitive sink {sink}')
                        self[sink] = Word(sink)
                        self[sink].nmut = 4
                    if not self.check(src):
                        if debug: print(f'Creating transitive source {src}')
                        self[src] = Word(src)
                        self[src].nmut = 4
                    self.update(sink, src, x[2:], cat='forms')
                    if readd:
                        self[src] = Word(src)
                        self.bases.add(src)
                manuals = man_v

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

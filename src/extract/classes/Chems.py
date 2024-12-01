from pickle import dump, load
from collections import Counter

class Chems:

    def __init__(self):
        self.veri = set('actinium aluminum americium antimony argon arsenic astatine barium berkelium beryllium bismuth bohrium boron bromine cadmium calcium californium carbon cerium cesium chlorine chromium cobalt copernicium copper cuprum curium darmstadtium dubnium dysprosium einsteinium erbium europium fermium flerovium fluorine francium gadolinium gallium germanium gold aurum hafnium hassium helium holmium hydrogen indium iodine iridium iron krypton lanthanum lawrencium lead plumbum lithium livermorium lutetium magnesium manganese meitnerium mendelevium mercury molybdenum neodymium neon neptunium nickel niobium nitrogen nobelium osmium oxygen palladium phosphorus platinum plutonium polonium potassium praseodymium promethium protactinium radium radon rhenium rhodium roentgenium rubidium ruthenium rutherfordium samarium scandium seaborgium selenium silicon silver argentum sodium strontium sulfur tantalum technetium tellurium terbium thallium thorium thulium tin titanium tungsten uranium vanadium xenon ytterbium yttrium zinc zirconium'.split())
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\pchems', 'rt', encoding='utf8') as f:
            self.chems = sorted([x.strip() for x in f.readlines()])
            self.achems = [x for x in self.chems]
        self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7 = self.manual_afx_read()
        self.cdct = {}
        for x in self.chems:
            for z in (self.t1, self.t2, self.t3):
                if x in z:
                    self.veri.add(x)
                    self.cdct[x] = z[x]
        for x in self.veri:
            if x in self.chems: self.chems.remove(x)
        for i, x in enumerate('methane ethane propane butane pentane hexane heptane octane nonane decane undecane dodecane tridecane tetradecane pentadecane hexadecane heptadecane octadecane nonadecane eicosane icosane'.split()):
            self.t6[x[:-1]] = x
            if i < 4: self.t6[x[:-2]] = x


    def manual_decomp(self):
        vtmp = []
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\oc_forms', 'rt', encoding='utf8') as f:
            for x in f.readlines():
                x = x.strip().split()
                self.cdct[x[1]] = ' '.join(x[2:])
                vtmp.append(x[1])
                for z in (self.t1, self.t2, self.t3):
                    if x[1] in z and ' '.join(x[2:]) != z[x[1]]:
                        print(f'{x[1]} Conflict {" ".join(x[2:])} ||| {z[x[1]]}')
        for x in vtmp:
            if x in self.chems:
                self.chems.remove(x)
                self.veri.add(x)
            elif x in self.achems: continue
            else: print(f'Add {x} to pchems')

    def manual_afx_read(self):
        pre, mid, suf = {}, {}, {}
        opre, osuf = {}, {}
        decon = {}
        ekeys = set()
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\bpack\\trial', 'rt', encoding='utf8') as f:
            rtest = [x.strip() for x in f.readlines()]
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\base_chems', 'rt', encoding='utf8') as f:
            rtest.extend([x.strip() for x in f.readlines()])
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\hydroma', 'rt', encoding='utf8') as f:
            rtest.extend([x.strip() for x in f.readlines()])
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\biota', 'rt', encoding='utf8') as f:
            rtest.extend([x.strip() for x in f.readlines()])
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\medica', 'rt', encoding='utf8') as f:
            rtest.extend([x.strip() for x in f.readlines()])
        rtest = [x.split() for x in rtest]
        rtest = [(x[0], x[1], x[2], ' '.join(x[3:])) for x in rtest]
        for line in rtest:
            rchop = True if line[2][0] == '-' else False
            exchop = True if line[2][-1] == 'x' else False
            lspl = True if '(' in line[0] else False
            lsplx = True if '|' in line[0] else False
            if lspl and not lsplx:
                keys = [f'{line[0][:line[0].index("(")]}{line[0][line[0].index(")")+1:]}', 
                        f'{line[0][:line[0].index("(")]}{line[0][line[0].index("(")+1:line[0].index(")")]}{line[0][line[0].index(")")+1:]}']
            elif lspl and lsplx:
                si = line[0].index("(")
                sii = line[0].index("(")+1
                see = line[0].index("|")
                se = line[0].index(")")+1

                keys = []
                while len(keys) < line[0].count('|')+1:
                    keys.append(f'{line[0][:si]}{line[0][sii:see]}{line[0][se:]}')
                    sii = see + 1
                    if '|' in line[0][see + 1:]: see = line[0][see + 1:].index('|') + see + 1
                    else: see = se - 1
            else: keys = [line[0]]

            lval = line[2]
            if rchop: lval = lval[1:]
            if exchop: lval = lval[:-1]
            lval = int(lval)
            if line[1] == 'e' and len(keys) == 1:
                osuf[keys[0]] = line[3]
            if line[1] == 'f' and len(keys) == 1:
                opre[keys[0]] = line[3]
            if line[1] == 'fme':
                sk = sorted(keys, key=lambda x: len(x))
                if len(keys) > 1:
                    if len(sk[0]) > len(sk[1]): ekeys.add(sk[0])
                    else:
                        if sk[0][-1] == 'e': ekeys.add(sk[0])
                        elif sk[1][-1] == 'e': ekeys.add(sk[1])
                else: ekeys.add(sk[0])

            for key in keys:
                for target in ('f', 'm', 'e'):
                    if target not in line[1]: continue
                    if target == 'f': tdct = pre
                    if target == 'm': tdct = mid
                    if target == 'e': tdct = suf

                    if not (key in tdct and line[3] != tdct[key]):
                        if not (key[-1] == 'e' and 'e' in line[1] and target in 'fm'):
                            tdct[key] = line[3]
                    else: print(f'Conflict {key} {tdct[key]} with {line}')

                    if exchop:
                        if rchop:
                            if not (key[lval:] in tdct and line[3] != tdct[key[lval:]]): tdct[key[lval:]] = line[3]
                            else: print(f'Conflict {key[lval:]} {tdct[key[lval:]]} with {line}')
                        elif target != 'e':
                            if not (key[:-lval] in tdct and line[3] != tdct[key[:-lval]]): tdct[key[:-lval]] = line[3]
                            else: print(f'Conflict {key[:-lval]} {tdct[key[:-lval]]} with {line}')
                            if len(keys) > 1 and key[-1] == 'o':
                                if key == keys[0]: decon[key[:-lval]] = keys[-1]
                                else: decon[key[:-lval]] = keys[0]
                            else: decon[key[:-lval]] = key
                    elif lval:
                        lcnt = lval
                        while lcnt:
                            if rchop:
                                if not (key[lcnt:] in tdct and line[3] != tdct[key[lcnt:]]): tdct[key[lcnt:]] = line[3]
                                else: print(f'Conflict {key[lcnt:]} {tdct[key[lcnt:]]} with {line}')
                            elif target != 'e':
                                if not (key[:-lcnt] in tdct and line[3] != tdct[key[:-lcnt]]): tdct[key[:-lcnt]] = line[3]
                                else: print(f'Conflict {key[:-lcnt]} {tdct[key[:-lcnt]]} with {line}')
                                if len(keys) > 1 and key[-1] == 'o':
                                    if key == keys[0]: decon[key[:-lcnt]] = keys[-1]
                                    else: decon[key[:-lcnt]] = keys[0]
                                else: decon[key[:-lcnt]] = key
                            lcnt -= 1
        return pre, mid, suf, opre, osuf, decon, ekeys

    def decomp(self, roots, pres, sufs, targ, dbl='', spc=(), dexc=(), keep=False):
        opd = {}
        tried = set()
        if isinstance(roots, dict):
            rwords = [y for x in roots.values() for y in x]
        else: rwords = roots
        if keep: found = [x for x in rwords]
        else: found = [x for x in rwords if x in targ]
        for r in rwords:
            for p in pres:
                rep = f'{p[:-1] if p in spc and r[0] == p[-1] else p}{r}'
                if rep in tried: continue
                if rep in targ:
                    opd[rep] = (f'_{p}', r)
                    found.append(rep)
                tried.add(rep)
                if 'p' in dbl and p not in dexc:
                    for pp in pres:
                        if pp == p or pp in dexc: continue
                        rep2 = f'{pp}{p[:-1] if p in spc and r[0] == p[-1] else p}{r}'
                        if rep2 in tried: continue
                        if rep2 in targ:
                            opd[rep2] = (f'_{pp}', f'_{p}', r)
                            found.append(rep2)
                        tried.add(rep2)
            for s in sufs:
                rep = f'{r}{s}'
                if rep in tried: continue
                if rep in targ:
                    opd[rep] = (r, f'{s}_')
                    found.append(rep)
                tried.add(rep)
                if 's' in dbl and s not in dexc:
                    for ss in sufs:
                        if ss == s or ss in dexc: continue
                        rep2 = f'{r}{s[:-1] if s[-1] == "e" else s}{ss}'
                        if rep2 in tried: continue
                        if rep2 in targ:
                            opd[rep2] = (r, f'{s}_', f'{ss}_')
                            found.append(rep2)
                        tried.add(rep2)
                for p in pres:
                    rep = f'{p[:-1] if p in spc and r[0] == p[-1] else p}{r}{s}'
                    if rep in tried: continue
                    if rep in targ:
                        opd[rep] = (f'_{p}', r, f'{s}_')
                        found.append(rep)
                    tried.add(rep)
                    if 's' in dbl and s not in dexc:
                        for ss in sufs:
                            if ss == s or ss in dexc: continue
                            rep2 = f'{p[:-1] if p in spc and r[0] == p[-1] else p}{r}{s[:-1] if s[-1] == "e" else s}{ss}'
                            if rep2 in tried: continue
                            if rep2 in targ:
                                opd[rep2] = (f'_{p}', r, f'{s}_', f'{ss}_')
                                found.append(rep2)
                            tried.add(rep2)
                    if 'p' in dbl and p not in dexc:
                        for pp in pres:
                            if pp == p or pp in dexc: continue
                            rep2 = f'{pp}{p[:-1] if p in spc and r[0] == p[-1] else p}{r}{s}'
                            if rep2 in tried: continue
                            if rep2 in targ:
                                opd[rep2] = (f'_{pp}', f'_{p}', r, f'{s}_')
                                found.append(rep2)
                            tried.add(rep2)
                            if 's' in dbl and s not in dexc:
                                for ss in sufs:
                                    if s == ss or ss in dexc: continue
                                    rep3 = f'{pp}{p[:-1] if p in spc and r[0] == p[-1] else p}{r}{s[:-1] if s[-1] == "e" else s}{ss}'
                                    if rep3 in tried: continue
                                    if rep3 in targ:
                                        opd[rep3] = (f'_{pp}', f'_{p}', r, f'{s}_', f'{ss}_')
                                        found.append(rep3)
                                    tried.add(rep3)
        if isinstance(roots, dict):
            fout = {x: {} for x in roots}
            rkey = {y: x[0] for x in roots.items() for y in x[1]}
            for x in opd.items():
                for y in x[1]:
                    if '_' in y: continue
                    fout[rkey[y]][x[0]] = ' '.join([rkey[z] if '_' not in z else z for z in x[1]])
            return fout, sorted(found)
        else: return opd, sorted(found)

    def inorg_afx(self):
        r1 = {}
        for x in 'aluminium aluminium alumin,antimony antimon stib,arsenic arseno arsen,astatine astat,barium bar,bismuth,boron boro bor,bromine brom,cadmium cadmo,caesium caes,calcium calc,carbon carb,cerium cer,chlorine chlor,chromium chrom,cobalt cobal,copper cupr,europium europ,fluorine fluor,gallium gall,germanium germ,gold aur,hafnium hafn,hydrogen hydro hydr,indium ind,iodine iod,iron ferro ferr,lead plumb,lithium lith,lutetium lute,magnesium magnes,manganese mangan,mercury mercur,molybdenum molybden molybd,neodymium neodym,niobium niob,nitrogen nitr,oxygen oxy ox,phosphorus phosphor phosph,platinum platin,polonium polon,potassium kal,rhodium rhod,rubidium rubi,ruthenium ruthen,selenium seleno selen,silicon silic,silver argentum,sodium sod hal,sulfur sulphur sulf sulph thion thio,tantalum tantal,technetium technet,tellurium tellur,thallium,tin stann,titanium titan,tungsten tungst,uranium uran,vanadium vanad vand,xenon xen,zinc,zirconium zircon'.split(','):
            r1[x.split()[0]] = x.split()
        p1 = 'ortho,meta,para,thio,hypo,per,mono,di,bi,tri,tetra,tetr,penta,pent,hexa,hex,hepta,hept,octa,oct,nona,non,deca,dec'.split(',')
        s1 = 'ide ite ate ic ous ol one oic oate onium yl oyl oxy'.split()
        return p1, r1, s1

    def org_afx(self):
        r2 = {}
        for x in 'alkane alk,methane meth,ethane eth,propane prop,butane but,pentane pent,hexane hex,heptane hept,octane oct,nonane non,decane dec,undecane undec,dodecane dodec,tridecane tridec,tetradecane tetradec,pentadecane pentadec,hexadecane hexadec,heptadecane heptadec,octadecane octadec,nonadecane nonadec,eicosane eicos'.split(','):
            r2[x.split()[0]] = x.split()
        p2 = 'fluoro bromo chloro iodo cyclo iso neo cis trans adi di atri tri tetra oxo oxa oxy'.split()
        s2 = set('ane ene yne yl ide ite ate ic ous ol oic oate onium yl oyl'.split())
        return p2, r2, s2


    def inorg_break(self):
        p1, r1, s1 = self.inorg_afx()
        ioc, v1 = self.decomp(r1, p1, s1, self.chems, dbl='ps', spc=('mono',), dexc=('deca', 'dec'), keep=True)
        self.update(ioc, v1)

    def org_break(self):
        p2, r2, s2 = self.org_afx()
        oc, v2 = self.decomp(r2, p2, s2, self.chems, dbl='sp')
        self.update(oc, v2)

    def full_break(self, replace=True):
        fails = []
        oc2 = {}
        agroup = [x for x in self.achems if x not in self.cdct]
        bgroup = sorted(self.t1, key=lambda x: 1/len(x))
        for x in self.chems:
            if replace: hold = [(x[len(y):], [self.t1[y]] if ' ' not in self.t1[y] else self.t1[y].split()) for y in bgroup if x.startswith(y)]
            else: hold = [(x[len(y):], [y]) for y in bgroup if x.startswith(y)]
            cands = []
            while hold:
                wr, wls = hold.pop()
                lwr = len(wr)
                for y in [y for y in self.cdct if wr.startswith(y)]:
                    if len(y) == lwr:
                        if replace: cands.append(tuple([*wls, *self.cdct[wr].split()]) if ' ' in self.cdct[wr] else tuple([*wls, self.cdct[wr]])) 
                        else: cands.append(tuple([*wls, wr]))
                    else:
                        if replace: hold.append((wr[len(y):], [*wls, *self.cdct[y].split()]) if ' ' in self.cdct[y] else (wr[len(y):], [*wls, self.cdct[y]]))
                        else: hold.append((wr[len(y):], [*wls, y]))
                for y in [y for y in agroup if wr.startswith(y)]:
                    if len(y) == lwr: cands.append(tuple([*wls, y]))
                    else: hold.append((wr[len(y):], [*wls, y]))

                for i, pos in enumerate((self.t1, self.t2, self.t3)):
                    for y in [y for y in pos if wr.startswith(y)]:
                        if lwr == len(y) and i != 2: continue
                        if i == 2 and y in self.t5 and len(y) < lwr: continue
                        if i == 0 and y in self.t4: continue
                        if replace:
                            if len(y) == lwr: cands.append(tuple([*wls, *pos[y].split()]) if ' ' in pos[y] else tuple([*wls, pos[y]]))
                            else: hold.append((wr[len(y):], [*wls, *pos[y].split()]) if ' ' in pos[y] else (wr[len(y):], [*wls, pos[y]]))
                        else:
                            if len(y) == lwr: cands.append(tuple([*wls, y]))
                            else: hold.append((wr[len(y):], [*wls, y]))
            if not cands: fails.append(x)
            else: oc2[x] = tuple(set(cands))
        return oc2, fails


    def coded_repair(self):
        for x in self.cdct:
            if x.endswith('tion') and self.cdct[x].endswith(' ate_ ion'):
                self.cdct[x] = f'{y[:-9]} tion_'
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\fulcom', 'rb') as f:
            comls = load(f)
        man_words = {x[2] for x in comls}
        for x in comls:
            if x[0] != 'S': continue
            for y in x[2:]:
                man_words.add(y.strip('+'))
        funds = 'cation anion zwitterion proton neutron fermion phonon hadron meson boson photon gluon tachyon lepton baryon nucleon'.split()
        tmp = {x: y for x in funds for y in comls if x == y[2]}
        for x in tmp.items():
            if x[1][-1][0] == '_': self.cdct[x[0]] = f'{x[1][-1]} {x[1][1]}'
            else: self.cdct[x[0]] = f'{x[1][1]} {x[1][-1]}'
            self.veri.add(x[0])
            if x[0] in self.chems:
                self.chems.remove(x[0])

    def mend_cdct(self, targ):
        ereps = {'_fluoro': 'fluorine', '_bromo': 'bromine', '_chloro': 'chlorine', '_iodo': 'iodine', '_cyclo': 'cycle', '_tetra': 'tetras', 'oxy_': 'oxygen', 'ase_': 'enzyme'}
        ignr = {x[0] for x in self.t1.items() if x[0] in x[1].split() and len(x[1].split()) > 1}
        tmp = []
        for x in targ.items():
            if len(x[1]) == 1:
                self.cdct[x[0]] = ' '.join(x[1][0])
                self.chems.remove(x[0])
                self.veri.add(x[0])
                tmp.append(x[0])
            else:
                repack = set()
                for y in x[1]:
                    if any(z in y for z in ereps): repack.add(tuple([q if q not in ereps else ereps[q] for q in y]))
                    else: repack.add(y)
                targ[x[0]] = tuple(repack)
        for x in tmp: targ.pop(x)

        for pack in targ.items():
            if len(pack[1]) == 1: continue
            ogpack = []
            for x in pack[1]:
                hold = [y for y in x]
                found = True
                i = 0
                while found:
                    hold2 = []
                    i += 1
                    for k, y in enumerate(hold):
                        if y in ignr:
                            hold2.append(y)
                            continue
                        found = False
                        for i, affixes in enumerate((self.t1, self.t2, self.t3)):
                            if y in affixes and not ((' ' in affixes[y] and y in affixes[y].split()) or y == affixes[y]):
                                if y in self.t4 and k != 0: continue
                                if y in self.t5 and k != len(hold) - 1: continue
                                hold2.extend(affixes[y].split()) if ' ' in affixes[y] else hold2.append(affixes[y])
                                found = True
                                break
                        if not found and y in self.cdct and y not in self.cdct[y]:
                            hold2.extend([*self.cdct[y].split()]) if ' ' in self.cdct[y] else hold2.append(self.cdct[y])
                            found = True
                        if not found: hold2.append(y)
                    if i > 10:
                        print(pack)
                        print(hold)
                        print(hold2)
                        break
                    hold = hold2
                    hold2 = list()
                if tuple(hold) not in ogpack: ogpack.append(tuple(hold))
            targ[pack[0]] = tuple(ogpack)
        return targ

    def update(self, cgroups, vers):
        for x in cgroups.values():
            for y in x.items():
                if y[0] not in self.cdct:
                    self.cdct[y[0]] = y[1]
                elif y[0] in self.cdct and self.cdct[y[0]] == y[1]: continue
                else: print(f'Conflict {y[0]}: {y[1]} || {self.cdct[y[0]]}')
        for z in vers:
            self.veri.add(z)
            if z in self.chems:
                self.chems.remove(z)


    def print_rxsuf(self):
        for x in sorted(self.achems):
            if x.endswith('mab'): print(self.mab_sfx(x)) if self.mab_sfx(x) else None
            elif x.endswith('stat'): print(self.statin_sfx(x)) if self.statin_sfx(x) else None
            elif x.endswith('statin'): print(self.statin_sfx(x)) if self.statin_sfx(x) else None
            elif x.endswith('stim'): print(self.stim_sfx(x)) if self.stim_sfx(x) else None
            elif x.endswith('tant'): print(self.tant_sfx(x)) if self.tant_sfx(x) else None
            elif x.endswith('tide'): print(self.tide_sfx(x)) if self.tide_sfx(x) else None
            elif x.endswith('tinib'): print(self.tinib_sfx(x)) if self.tinib_sfx(x) else None
            elif x.endswith('gene'): print(self.gene_sfx(x)) if self.gene_sfx(x) else None
            elif x.endswith('vec'): print(self.vec_sfx(x)) if self.vec_sfx(x) else None
            elif x.endswith('vir'): print(self.vir_sfx(x)) if self.vir_sfx(x) else None
            elif x.endswith('ase'): print(self.ase_sfx(x)) if self.ase_sfx(x) else None
            elif x.endswith('ast'): print(self.ast_sfx(x)) if self.ast_sfx(x) else None

    def dump_chems(self):
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\tmpchems', 'wb') as f:
            dump((self.cdct, self.oc2, self.veri, self.chems, self.fails), f)

    def load_chems(self):
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\tmpchems', 'rb') as f:
            self.cdct, self.oc2, self.veri, self.chems, self.fails = load(f)


    def deconfor(self):
        tmp = {}
        for z in self.t6.items():
            tmp[z[1]] = [x[0] for x in self.t6.items() if x[1] == z[1]]
            if len(z[1]) - len(sorted(tmp[z[1]], key=lambda m: len(m))[0]) == 1: tmp.pop(z[1])
        tmp = [sorted((*x[1], x[0]), key=lambda m: len(m)) for x in tmp.items()]
        new = {}
        for x in tmp:
            for i in range(0, len(x)-1):
                sword = x[i]
                for j in range(i+1, len(x)):
                    if j-i < 1: continue
                    pack = set()
                    efx = x[j][len(sword):]
                    for pre in (self.t1, self.t2):
                        for suf in (self.t2, self.t3):
                            if sword in pre:
                                if efx in suf:
                                    pack.add(f'{pre[sword]} {suf[efx]}')
                    if pack:
                        bundle = []
                        for pac in pack:
                            reap = []
                            for mq in pac.split():
                                if mq not in self.cdct: reap.append(mq)
                                elif ' ' in self.cdct[mq]: reap.extend(self.cdct[mq].split())
                                else: reap.append(self.cdct[mq])
                            if ' '.join(reap) not in pack: bundle.append(' '.join(reap))
                        for pac in bundle: pack.add(pac)
                        if x[j] in new: new[x[j]] = tuple(set([*new[x[j]], *pack]))
                        else: new[x[j]] = tuple(set(pack))
        return new

    def coc(self, show=False):
        tmp = []
        for x in self.oc2.items():
            if len(x[1]) == 1:
                self.veri.add(x[0])
                self.cdct[x[0]] = ' '.join(x[1][0])
                self.chems.remove(x[0])
                tmp.append(x[0])
                if show: print(f'{x[0]}: {" ".join(x[1][0])}')
        for x in tmp: self.oc2.pop(x)

    def scrub1(self, show=False):
        tmp = set('acyl alcohol aldehyde alkali alkane allium alum aluminium amide amine ammonia amyl anthrax antimony argentum arsenic astatine bakelite barium benzene boron bromine cadmium caesium calcium carbo carbon carboxyl cerium chitin chlorine chromium cobalt cyanide cyano epoxy ester ether europium fluorine freon gallium gasoline germanium guano hafnium hydrogen hydroxyl imide imine iodine ion iridium ketone lanthanum lithium lutetium magnesium manganese mercury naphtha neodymium niobium nitrogen nylon opium oxygen phosphorus platinum polonium prene rhodium rubidium ruthenium selenium silicon sodium sulfur technetium tellurium titanium ununennium uranium urea vanadium zirconium'.split())
        for x in tmp:
            if x in self.chems: self.chems.remove(x)
            if x in self.fails: self.fails.remove(x)
            if x in self.oc2: self.oc2.pop(x)
            self.veri.add(x)
            if show: print(x)
        mods = {'oic_': 'carboxyl ic_', 'oate_': 'carboxyl ate_', 'oyl_': 'ketone', 'one_': 'ketone'}
        for x in self.oc2.items():
            repack = []
            for y in x[1]:
                hold = []
                for z in y:
                    if z in mods:
                        if ' ' in mods[z]:
                            hold.extend(mods[z].split())
                        else:
                            hold.append(mods[z])
                    else:
                        hold.append(z)
                repack.append(tuple(hold))
            self.oc2[x[0]] = tuple(repack)

    def scrub2(self, test=False):
        new = self.deconfor()
        filt1 = ('_meta _allo', 'ite_ hydroxyl', 'benzene ic_', 'carbon xylon ic_', 'ide_ ine_', 'benzoic', 'alum ine_', 'benzoate', 'benzene ate_', '_dia hexadecan', 'iridium ine_', 'hydroxyl ide_',
                'hydroxyl ine_', 'pentas ring ine_', 'aldehyde ene_', 'propan ane_ ion', 'hexas ring ane_', 'carboxyl carboxyl ic_', 'carboxyl carboxyl ate_', 'carboxyl yl_ ic_', 'carboxyl yl_ ate_',
                'carbon xylon', 'hydrogen xylon', 'pentas ring idine_', 'hydroxyl idine_', 'ketone ium_', 'cortex ster oid_ oid_', 'cortex ic_ oid_', '_per idine_', 'alum ane_', 'mycos in_', 'alkane yl_ ene_ hydroxyl')
        filt2 = {' id_': ' oid_', ' protein': ' protos ein_', 'carbon oxygen': 'carboxyl', 'hydrogen oxygen': 'hydroxyl', 'decan ane_ oic_': 'hexan ane_ oic_',
                'nitrogen ine_': 'nitrogen hexas ring', 'oxygen ine_': 'oxygen hexas ring', 'oxygen ane_': 'oxygen satis hexas ring', 'sulfur ane_': 'sulfur satis hexas ring',
                'sulfur in_': 'sulfur hexas ring', 'hydroxyl ane_': 'satis pentas ring', 'malus ic_ ketone': 'malon', 'cytos ine_': 'cytos ribos ose_ ine_', 'imide nitrogen pentas ring': '_di nitrogen pentas ring',
                'silicon yl_': 'silicon oxygen yl_', 'nitrogen ic_': 'nitrogen carboxyl ic_', 'state in_': 'enzyme inhibitor', 'line ene_': 'ane_ line ene_', 'urea ketone': 'urea aldehyde carboxyl',
                'boron ic_': 'boron _di hydroxyl', 'ane_ _di ene_': '_di ene_', 'ketone ine_': 'nonas ring', 'as _di ene_': 'an ane_ _di ene_', 'pentas ring ide_': 'satis pentas ring',
                'sulfur ketone': 'sulfur _di oxygen', 'ster hydroxyl': 'ster oid_ hydroxyl', 'cycle ine_': 'protein synthesis inhibitor', 'ster enzyme': 'ester enzyme', 'aldehyde ane_ ine_': 'aldehyde ine_',
                'ster ketone': 'ster oid_ hormos ketone', 'alkane yl_ ene_ carboxyl ic_': 'ene_ carboxyl ic_'}
        rls = []
        for x in self.oc2.items():
            new_pack = []
            repack = [' '.join(z) for z in x[1]]
            if test: print('\n', x[0])
            for zx in repack:
                found = False
                for y in new.items():
                    if not y[0] in x[0]: continue
                    for m in y[1]:
                        if m in zx: found = True
                if any(p in zx for p in filt1): found = True
                if any(p in zx and any(filt2[p] in xz and zx != xz and zx.index(p) == xz.index(filt2[p]) for xz in repack) for p in filt2): found = True
                if test:
                    if found: print(f'  -----  {zx}')
                    else: print(zx)
                if not found: new_pack.append(tuple(zx.split()))
            if test: continue
            self.oc2[x[0]] = tuple(new_pack)
            if not new_pack: rls.append(x[0])
        for x in rls:
            self.fails.append(x)
            self.oc2.pop(x)

    def scrub3(self):
        dcc = {'_tetra': 'tetras', '_hypo': 'hypo', '_chloro': 'chlorine yl_', '_ortho': 'ortho', '_penta': 'pentas', '_cyclo': 'cycle', 'oic_': 'hydroxyl ic_', '_thio': 'sulfur',
            '_hexa': 'hexas', '_fluoro': 'fluorine yl_', 'kine_': 'kines in_', 'oate_': 'hydroxyl ate_', '_hepta': 'heptas', '_octa': 'octas', 'oxy_': 'oxygen', 'oside_': 'glycos ose_ ide_',
            'ulose_': 'ule_ ose_', '_cis': 'cis', '_deca': 'decas', '_tetr': 'tetras', '_pent': 'pentas', '_hex': 'hexas', '_hept': 'heptas', '_oct': 'octas', '_iodo': 'iodine yl_',
            '_bromo': 'bromine yl_', '_neo': 'neo', 'allos': '_allo', 'anti': '_anti', 'poly': '_poly', 'ase_': 'enzyme', 'ole_': 'oleum', 'one_': 'ketone', 'ol_': 'hydroxyl'}
        for x in self.cdct.items():
            if ' ' not in x[1]: continue
            if any(z in x[1] for z in dcc):
                self.cdct[x[0]] = ' '.join([z if z not in dcc else dcc[z] for z in x[1].split()])



    def trans(self, ioc):
        hold = {}
        for x in ioc.items():
            hold1 = set()
            for y in x[1]:
                hold2 = []
                for z in y:
                    if z in self.cdct:
                        if ' ' in self.cdct[z]: hold2.extend(self.cdct[z].split())
                        else: hold2.append(self.cdct[z])
                    else: hold2.append(z)
                hold1.add(tuple(hold2))
            hold[x[0]] = tuple(hold1)
        return hold

    def conf_sufs(self):
        fdct = {'mab': self.mab_sfx, 'statin': self.statin_sfx, 'tinib': self.tinib_sfx, 'vir': self.vir_sfx, 'ast': self.ast_sfx}
        for x in 'mab statin tinib vir ast'.split():
            print(f':: {x}')
            for y in bsearch(x):
                out = fdct[x](y)
                if y in self.cdct:
                    print(f'        {self.cdct[y]}')
                else:
                    print(f'{out[-1].split()[-1]} {out[0]} {y[:-1]} {out[2]}')

    def ctrans(self, idct, show=False):
        ncdt = {}
        for x in idct.items():
            hold = []
            for y in x[1].split():
                if y in idct:
                    if ' ' in idct[y]:
                        hold.extend(idct[y].split())
                    else:
                        hold.append(idct[y])
                else:
                    hold.append(y)
            if show and x[1] != ' '.join(hold): print(f'{x[0]} ::: {x[1]} |-| {" ".join(hold)}')
            ncdt[x[0]] = ' '.join(hold)
        return ncdt

    def best_e(self, word):
        hold = [(x[len(x):], [self.t1[x]]) for x in self.t1 if word.startswith(x)]
        out = set()
        while hold:
            wr, wls = hold.pop()
            f = False
            for x in self.t1:
                if wr.startswith(x):
                    hold.append((wr[len(x):], [*wls, self.t1[x]]))
                    f = True
            for x in self.t2:
                if wr.startswith(x):
                    hold.append((wr[len(x):], [*wls, self.t2[x]]))
                    f = True
            if not f: out.add(tuple(wls))
        return out

    def trans1(self, entry, tdc):
        hold = []
        if ' ' in entry:
            entry = entry.split()
        else: entry = (entry,)
        while True:
            hold = []
            for x in entry:
                if x in tdc:
                    if ' ' in tdc[x]:
                        hold.extend(tdc[x].split())
                    else: hold.append(tdc[x])
                else: hold.append(x)
            if ' '.join(hold) == ' '.join(entry):
                return ' '.join(hold)
            else: entry = hold

    def medlim(self, idct):
        igls = 'mab statin tinib vir ast'.split()
        rems = [x for x in self.achems if not any(x.endswith(z) for z in igls)]
        ends = []
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\medica', 'rt', encoding='utf8') as f:
            for x in sorted([x.strip().split() for x in f.readlines()], key=lambda z: 1/len(z[0])):
                found = False
                excs = []
                tout = trans1(' '.join(x[3:]), ct3)
                if x[0] in igls: continue
                if 'f' in x[1]:
                    for y in self.achems:
                        if y.startswith(x[0]) and x[0] != y:
                            if not found:
                                found = True
                                print(f'F :{x[0]}:  {tout}')
                            print(f'{x[-1]} {y} {tout}')
                            if y in idct and tout in idct[y]: print(f'{x[-1]} {y} {idct[y]}')
                            elif y in idct and idct[y].endswith(x[0]): print(f'{x[-1]} {y} {idct[y][:-len(x[0])]} {tout}')
                            elif y in idct: excs.append(f'X-: {y} {idct[y]}')
                            else: print(f'{x[-1]} {y} {y[:-1]} {tout}')
                if 'e' in x[1]:
                    for y in rems[::-1]:
                        if y.endswith(x[0]) and x[0] != y:

                            if y in idct and tout in idct[y]: pass  # print(f'{x[-1]} {y} {idct[y]}')
                            #elif y in idct and idct[y].endswith(x[0]): print(f'{x[-1]} {y} {idct[y][:-len(x[0])]} {tout}')
                            elif y in idct: excs.append(f'X-: {y} {idct[y]}  |  {tout}')
                            else:
                                if not found:
                                    found = True
                                    print(f' :{x[0]}:  {tout}')
                                print(f'{x[-1]} {y} {y[:-1]} {tout}')
                            rems.remove(y)
                if found:
                    if excs:
                        for z in excs: print(z)
                    print('\n')
                    ends.append(x[0])
        return ends

    def conf_test(self, bat=''):
        if not bat or '1' in bat:
            print('T1')
            cprint([rrsort(fails), (zc := [tx[0] if (tx := sorted([z for z in self.t3 if x.endswith(z)], key=lambda q: 1/len(q))) else "---" for x in rrsort(fails)]), [z[:len(z)-len(zc[j])] for j, z in enumerate(rrsort(fails))]], [0, 2])
        if not bat or '2' in bat:
            print('T2: Main and Sub value dictionary collisions')
            for x in [x for x in self.cdct.items()]:
                if x[0] in self.t1:
                    if ' '.join(x[1]) != self.t1[x[0]]:
                        print(x[0], x[1], self.t1[x[0]])
                elif x[0] in self.t2:
                    if ' '.join(x[1]) != self.t2[x[0]]:
                        print(x[0], x[1], self.t2[x[0]])
                elif x[0] in self.t3:
                    if ' '.join(x[1]) != self.t3[x[0]]:
                        print(x[0], x[1], self.t3[x[0]])
        cnt = Counter()
        check = set()
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\oc_forms', 'rt', encoding='utf8') as f:
            oforms = [x.strip().split() for x in f.readlines()]
        if not bat or '3' in bat:
            print('T3: Double Main dictionary entries')
            for x in oforms:
                if x[1] not in check: check.add(x[1])
                else: print(x[1])
                for y in x[2:]:
                    cnt[y] += 1
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\bpack\\trial', 'rt', encoding='utf8') as f:
            rtest = [x.strip() for x in f.readlines()]
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\base_chems', 'rt', encoding='utf8') as f:
            rtest.extend([x.strip() for x in f.readlines()])
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\hydroma', 'rt', encoding='utf8') as f:
            rtest.extend([x.strip() for x in f.readlines()])
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\biota', 'rt', encoding='utf8') as f:
            rtest.extend([x.strip() for x in f.readlines()])
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\medica', 'rt', encoding='utf8') as f:
            rtest.extend([x.strip() for x in f.readlines()])
        if not bat or '4' in bat:
            print('T4: Double Sub dictionary entries')
            for x in [x.split() for x in rtest]:
                if x[0] not in check: check.add(x[0])
                else: print(x[0])
                for y in x[3:]:
                    cnt[y] += 1
        if not bat or '5' in bat:
            print('T5: Check for missing subcomponents')
            for x in cnt.most_common():
                if x[0] not in self.achems and '_' not in x[0] and x[0] not in self.t1 and x[0] not in self.t3 and x[0] not in self.t2:
                    print(x)
        if not bat or '6' in bat:
            print('T6: Main and Sub dictionary key collisions')
            of2 = [x[1] for x in oforms]
            rt2 = [x[0] for x in rtest]
            for x in of2:
                if x in rt2:
                    print(x)
        if not bat or '7' in bat:
            print('T7: Medication Check')
            with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\misc\\cpack\\medica', 'rt', encoding='utf8') as f:
                rems = [x for x in self.achems]
                for x in sorted([x.strip().split() for x in f.readlines()], key=lambda z: 1/len(z[0])):
                    found = False
                    if 'f' in x[1]:
                        for y in self.achems:
                            if y.startswith(x[0]):
                                if not found:
                                    found = True
                                    print(f'{x[0]}: {" ".join(x[3:])}')
                                print(f'    {y}')
                                if y in self.cdct:
                                    print(f'        {self.cdct[y]}')
                    if 'e' in x[1]:
                        for y in rems[::-1]:
                            if y.endswith(x[0]):
                                if not found:
                                    found = True
                                    print(f'{x[0]}: {" ".join(x[3:])}')
                                print(f'    {y}')
                                if y in self.cdct:
                                    print(f'        {self.cdct[y]}')
                                rems.remove(y)
        if not bat or '8' in bat:
            print('T8: Null dictionary keys')
            for x in self.t1.items():
                if not any(z.startswith(x[0]) for z in self.achems):
                    box = []
                    for y in self.t1.items():
                        if x[1] == y[1] and x[0] != y[0]:
                            box.append(y[0])
                    if box:
                        for z in box:
                            if any(q.startswith(z) for q in self.achems):
                                break
                        else: print(f'F: {x[0]} - {x[1]}')
                    else: print(f'F: {x[0]} - {x[1]}')
            for x in self.t2.items():
                if not (any(x[0] in z and not z.endswith(x[0]) for z in self.achems) if x[0] not in self.t3 else any(x[0] in z for z in self.achems)):
                    box = []
                    for y in self.t2.items():
                        if x[1] == y[1] and x[0] != y[0]:
                            box.append(y[0])
                    if box:
                        for z in box:
                            if (any(z in q and not q.endswith(z) for q in self.achems) if z not in self.t3 else any(z in q for q in self.achems)):
                                break
                        else: print(f'M: {x[0]} - {x[1]}')
                    else: print(f'M: {x[0]} - {x[1]}')
            for x in self.t3.items():
                if not (any(z.endswith(x[0]) for z in self.achems) if x[0] not in self.t1 or x[0] in self.t2 else any(x[0] in z for z in self.achems)):
                    box = []
                    for y in self.t3.items():
                        if x[1] == y[1] and x[0] != y[0]:
                            box.append(y[0])
                    if box:
                        for z in box:
                            if (any(q.endswith(z) for q in self.achems) if not z in self.t1 or z not in self.t2 else any(z in q for q in self.achems)):
                                break
                        else: print(f'E: {x[0]} - {x[1]}')
                    else: print(f'E: {x[0]} - {x[1]}')

    def show_dconflicts(self):
        for x in self.cdct.items():
            if x[0] == x[1]: continue
            if ' ' not in x[1]: continue
            last = []
            hold = x[1]
            cnt = 0
            while last != hold:
                last = hold
                hold = []
                for y in last.split():
                    if y in self.cdct:
                        if ' ' in self.cdct[y]: hold.extend(self.cdct[y].split())
                        else: hold.append(self.cdct[y])
                    else: hold.append(y)
                hold = ' '.join(hold)
                cnt += 1
                if cnt > 10:
                    print(last)
                    print(hold)
                    break
            if hold != x[1]:
                if len(hold) < 90:
                    print(f'{x[0]}: {x[1]} -- {hold}')
                else:
                    print(f'!= {x[0]}: {x[1]} -- {hold}')

    def show_bconflicts(self):
        for x in sorted(self.oc2.items(), key=lambda x: x[0]):
            print(x[0])
            for y in x[1]:
                print(f'-{" ".join(y)}')
                print(f'{y[0]} {x[0]} {" ".join(y)}')
            print('\n')

    def dbreak(self, skip=False):
        if not skip:
            self.inorg_break()
            self.org_break()
            self.manual_decomp()
            self.coded_repair()
            self.oc, self.fails = self.full_break()
            self.oc2 = self.mend_cdct(self.oc)
            self.dump_chems()
        else:
            self.load_chems()
        self.scrub1()
        self.scrub2()
        self.oc2 = self.trans(self.oc2)
        self.coc()
        self.oc2 = self.trans(self.oc2)
        self.coc()
        self.cdct = self.ctrans(self.cdct)
        self.cdct = self.ctrans(self.cdct)
        self.cdct = self.ctrans(self.cdct)
        self.scrub3()
        print(len(self.achems), len(self.chems), len(self.cdct), len(self.veri), len(self.oc2), len(self.fails))



    def mab_sfx(self, word):
        if not word.endswith('mab'): return
        ends = {'mab': 'mono clonal antibody', 'zumab': 'humanized mono clonal antibody', 'omab': 'mouse mono clonal antibody', 'xizumab': 'chimera humanized mono clonal antibody', 'ximab': 'chimera mono clonal antibody', 'umab': 'human mono clonal antibody'}
        mids = {
            'ami': 'amyloidosis',
            'am': 'amyloidosis',
            'b': 'bacterial',
            'ba': 'bacterial',
            'bac': 'bacterial',
            'c': 'cardiovascular',
            'ci': 'cardiovascular',
            'cir': 'cardiovascular',
            'f': 'fungal',
            'fu': 'fungal',
            'fung': 'fungal',
            'gr': 'muscle growth',
            'gro': 'muscle growth',
            'gros': 'muscle growth',
            'k': 'interleukin',
            'ki': 'interleukin',
            'kin': 'interleukin',
            'l': 'immune modulating',
            'li': 'immune modulating',
            'lim': 'immune modulating',
            'n': 'neural',
            'ne': 'neural',
            'ner': 'neural',
            'o': 'bone',
            's': 'bone',
            'so': 'bone',
            'os': 'bone',
            't': 'tumor',
            'tu': 'tumor',
            'tum': 'tumor',
            'ta': 'tumor',
            'co': 'tumor',
            'col': 'tumor',
            'go': 'tumor',
            'got': 'tumor',
            'gov': 'tumor',
            'ma': 'tumor',
            'mar': 'tumor',
            'me': 'tumor',
            'mel': 'tumor',
            'pr': 'tumor',
            'pro': 'tumor',
            'tox': 'toxin',
            'toxa': 'toxin',
            'vet': 'veterinary',
            'v': 'viral',
            'vi': 'viral',
            'vir': 'viral',
            'le': 'lesion',
            'les': 'lesion',
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')

    def statin_sfx(self, word):
        ends = {'stat': 'inhibitor', 'statin': 'inhibitor'}
        if not any(word.endswith(z) for z in ends): return
        mids = {
            'ca': 'dopamine beta hydroxyl enzyme',
            'du': 'hypoxia inducible factor prolyl hydroxyl enzyme',
            'ele': 'elastic enzyme',
            'gace': 'gamma _se crete enzyme',
            'ino': 'histone deactylase',
            'li': 'gastrointestinal lipo enzyme',
            'ma': 'matrix metal protein enzyme',
            'mo': 'proteolytic enzyme',
            're': 'aldose reduce ezyme',
            'va': 'antihyper lipidaemia _co enzyme reductase'
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        if not any([tword.endswith(z) for z in mids]): return (word, tword, ends[end])
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')

    def stim_sfx(self, word):
        ends = {'stim': 'colony stimulating factor'}
        if not any(word.endswith(z) for z in ends): return
        mids = {
            'di': '_di',
            'gramo': 'granulocyte macrophage',
            'gra': 'granulocyte',
            'mo': 'macrophage',
            'ple': 'interleukin derivative'
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        if not any([tword.endswith(z) for z in mids]): return (word, tword, ends[end])
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')

    def tant_sfx(self, word):
        ends = {'tant': 'receptor antagonist'}
        if not any(word.endswith(z) for z in ends): return
        mids = {
            'pi': 'NK1 neurokinin',
            'du': 'NK2 neurokinin',
            'ne': 'NK3 neurokinin',
            'ner': 'neurotensin'
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        if not any([tword.endswith(z) for z in mids]): return (word, tword, 'neurokinin tachykinin antagonist')
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')

    def tide_sfx(self, word):
        ends = {'tide': 'peptide'}
        if not any(word.endswith(z) for z in ends): return
        mids = {
            'glu': 'glucagon like',
            'mo': 'immunization agent',
            'reo': 'somastatin modulator',
            'ri': 'natriuretic',
            'ac': 'synthetic poly'
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        if not any([tword.endswith(z) for z in mids]): return (word, tword, ends[end])
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')

    def tinib_sfx(self, word):
        ends = {'tinib': 'tyrosine kinase inhibitor'}
        if not any(word.endswith(z) for z in ends): return
        mids = {
            'bru': 'agammaglobulinaemia Bruton',
            'ci': 'Janus',
            'me': 'mitogen kinase'
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        if not any([tword.endswith(z) for z in mids]): return (word, tword, ends[end])
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')

    def gene_sfx(self, word):
        ends = {'gene': 'genetic therapeutic'}
        if not any(word.endswith(z) for z in ends): return
        mids = {
            'cima': 'cytosine deaminase',
            'ermin': 'growth factor',
            'kin': 'interleukin',
            'lim': 'immunomodulator',
            'lip': 'human lipoprotein',
            'mul': 'multiple target',
            'stim': 'colony stimulating factor',
            'tima': 'thymidine kinase',
            'tusu': 'tumor supression'
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        if not any([tword.endswith(z) for z in mids]): return (word, tword, ends[end])
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')

    def vec_sfx(self, word):
        ends = {'vec': 'vector', 'repvec': 'replicating vector'}
        if not any(word.endswith(z) for z in ends): return
        mids = {
            'adeno': 'adenovirus',
            'cana': 'canary virus',
            'foli': 'fowlpox virus',
            'herpa': 'herpes virus',
            'lenti': 'lentivirus',
            'morbilli': 'paramoxyviridae morbillivirus',
            'parvo': 'parvoviridae dependovirus',
            'retro': 'retrovirus',
            'vaci': 'vaccinia virus'
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        if not any([tword.endswith(z) for z in mids]): return (word, tword, ends[end])
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')

    def vir_sfx(self, word):
        ends = {'vir': 'antiviral'}
        if not any(word.endswith(z) for z in ends): return
        mids = {
            'ami': 'neuraminidase inhibitor',
            'as': 'hepatitis inhibitor',
            'bu': 'polymerase inhibitor',
            'ca': 'carbocyclic nucleoside',
            'ciclo': '_bi cycle',
            'fo': 'phosphonic',
            'gosi': 'glucoside inhibitor',
            'na': 'HIV protease inhibitor',
            'pre': 'hepatitis protease inhibitor',
            'tegra': 'HIV integrase inhibitor'
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        if not any([tword.endswith(z) for z in mids]): return (word, tword, ends[end])
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')

    def ase_sfx(self, word):
        ends = {'ase': 'enzyme'}
        if not any(word.endswith(z) for z in ends): return
        mids = {
            'dipl': '_di plasminogen activator',
            'dism': 'superoxide dismutase',
            'lip': 'lipid',
            'tepl': 'tissue plasminogen activator',
            'upl': 'urokinase plasminogen activator',
            'dorn': 'deoxyribonucleus',
            'glucer': 'glucosylceramide',
            'glucosid': 'glucoside',
            'ic': 'urate oxide',
            'li': 'lyze',
            'sulf': 'sulfate'
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        if not any([tword.endswith(z) for z in mids]): return (word, tword, ends[end])
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')

    def ast_sfx(self, word):
        ends = {'ast': 'antiallergic'}
        if not any(word.endswith(z) for z in ends): return
        mids = {
            'luk': 'leukotriene antagonist',
            'mil': 'phosphodiesterase inhibitor',
            'tegr': 'integrin antagonist',
            'trod': 'thromboxane antagonist',
            'zol': 'leukotriene biosynthesis inhibitor'
        }
        end = sorted([x for x in ends if word.endswith(x)], key=lambda x: 1/len(x))[0]
        tword = word[:-len(end)]
        if not any([tword.endswith(z) for z in mids]): return (word, tword, ends[end])
        mid = sorted([x for x in mids if tword.endswith(x)], key=lambda x: 1/len(x))[0]
        return (word, tword[:-len(mid)], f'{mids[mid]} {ends[end]}')


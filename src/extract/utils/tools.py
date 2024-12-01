
import numpy as np
from pickle import load, dump
from collections import Counter
from typing import Iterable, Container, Protocol
from tqdm import tqdm
from btk import cprint, fzip, rdx_sort, hprint, lrsort, rrsort, permutes

__all__ = [
    'cprint', 'fzip', 'rdx_sort', 'hprint', 'lrsort', 'rrsort', 'edge_scan',
    'np', 'load', 'dump', 'Counter', 'tqdm', 'permutes',
    'Iterable', 'Container', 'Protocol'
]

def edge_scan(words, side, depth=0, thresholds=None, merge=True):
    if side not in ('r', 'l'): raise ValueError('Invalid Side')
    if not isinstance(depth, int) or depth < 2: raise ValueError("Invalid Depth")
    if thresholds:
        if not isinstance(thresholds, Container) or any(not isinstance(y, int) for y in thresholds): raise ValueError("Invalid Thresholds")
    if not depth:
        depth = int(np.average([len(x) for x in words]))
        depth += (2 if depth > 4 else 1)
    else: depth += 1

    ecnt = Counter()
    for w in words[1:]:
        for i in range(2, depth):
            tgt = (w[-i:] if side == 'r' else w[:i])
            if ' ' in tgt: break
            else: ecnt[tgt] += 1
    for x in [x[0] for x in ecnt.most_common() if x[1] < (3 if thresholds else 2)]: ecnt.pop(x)

    if merge:
        fltr, vmerge = [], []
        for x in [x for x in ecnt.most_common() if (x[0][0] if side == 'r' else x[0][-1]) not in ('a', 'e', 'i', 'o', 'u')]:
            matches = [y for y in ecnt if x[0] in y and len(y) == len(x[0])+1 and (y[0] if side == 'r' else y[-1]) in ('a', 'e', 'i', 'o', 'u')]
            if len(matches) > 1:
                if sum([ecnt[y] for y in matches]) > (x[1]*0.85 if x[1] >= 50 else (x[1]-5 if x[1] > 12 else x[1]-3)):
                    fltr.extend(matches)
                    vmerge.append((x[0], x[1]))
        for x in fltr: ecnt.pop(x)
        
        fltr = []
        for x in ecnt.most_common():
            matches = [y for y in ecnt if x[0] != y and x[0] in y and y not in fltr]
            for y in matches:
                if x[1] == ecnt[y]:
                    fltr.append(y)
                    break
                if ecnt[y] > (x[1]*0.9 if x[1] >= 50 else (x[1]-3 if x[1] > 12 else x[1]-2)):
                    fltr.append(y)

    if thresholds and isinstance(thresholds, Container):
        thresholds = {i+2: x for i, x in enumerate(thresholds)}
        if len(thresholds) < depth:
            for i in range(len(thresholds)+2, depth):
                thresholds[i] = 3
    elif thresholds == True:
        cs = (min(max(2**(16/len(words)), 0), 2) - 1)**0.333
        if cs < 0.10: thresholds = [int(max(min(950/(y**2.22), 768), 3)) for y in range(1, depth+1)]
        elif cs > 0.60: thresholds = [int(max(min((x/4)/(y**2.22), 16), 3)) for y in range(1, depth+1)]
        else:  thresholds = [int(max(min((0.75 if y == 1 else 1) * x*cs / (y**2.22), 768), 3)) for y in range(1, depth+1)]
    else: thresholds = {i: 2 for i in range(2, depth+1)}

    if merge:
        for x in vmerge:
            if x[0] in ecnt:
                ecnt.pop(x[0])
                ecnt[(f'_{x[0]}' if side == 'r' else f'{x[0]}_')] = x[1]
        return [x for x in ecnt.most_common() if x[0] not in fltr and x[1] >= thresholds[len(x[0])]]
    else: return [x for x in ecnt.most_common() if x[1] >= thresholds[len(x[0])]]

def compare(lst1=None, lst2=None, words=None):
    base = r'C:\Users\BBA\Coding\NLP\Embeddings\data\\'
    names = {
        'roots': r'v0\roots\_base',
        'master': r'v1\master_list',
        'b_add': r'v0\roots\_base_add',
        'b_remove': r'v0\roots\_base_remove',
        'adjectives': r'v1\adjectives',
        'nouns': r'v1\nouns',
        'adverbs': r'v1\adverbs',
        'verbs': r'v1\verbs',
        'man_1': r'v0\roots\_pre_manual',
        'syn_1': r'v0\roots\_pre_syn',
        'man_2': r'v0\roots\_manual',
        'syn_2': r'v0\roots\_syn',
        'food': r'v1\food',
        'ingredients': r'v1\ingredients',
        'biota': r'v1\biota',
        'simplex': r'v0\roots\_simplex',
        'ign_a': r'v0\misc\_adj_ignore',
        'ign_pl': r'v0\misc\_plural_ignore',
        'ign_m': r'v0\misc\_member_ignore',
        'ign_pa': r'v0\misc\_past_ignore',
        'ign_pf': r'v0\affixes\_prefix_ignore',
        'ign_pr': r'v0\misc\_preprog_ignore',
        'rep_a': r'v0\misc\_adj_reps',
        'rep_pl': r'v0\misc\_plural_reps',
        'rep_m': r'v0\misc\_member_reps',
        'rep_pa': r'v0\misc\_past_reps',
        'rep_pr': r'v0\misc\_preprog_reps',
        'rep_pf': r'v0\affixes\_prefix_reps',
        'latin': r'v1\latin',
        'chem': r'v1\chems0'
    }
    if words and not (lst1 or lst2):
        if isinstance(words, str): words = (words,)
        for x in names.items():
            if x[0] in ('master', 'nouns', 'adverbs', 'adjectives', 'verbs'): continue
            with open(f'{base}{x[1]}', 'rt') as f:
                if any(x[0].startswith(y) for y in ('rep', 'man', 'syn')):
                    tmp = [z.strip().split() for z in f.readlines()]
                    for w in words:
                        if w in [z[0] for z in tmp]: print(f'{w} found in {x[0]} as a sink')
                        if w in [y for z in tmp for y in z[1:]]: print(f'{w} found in {x[0]} as a source')
                else:
                    for w in words:
                        if w in {z.strip() for z in f.readlines()}: print(f'{w} found in {x[0]}')  
    elif words and lst1:
        if lst1 not in names: raise ValueError(f'Invalid search target {lst1}')
        if isinstance(words, str): words = (words,)
        with open(f'{base}{names[lst1]}', 'rt') as f:
            if any(lst1.startswith(y) for y in ('rep', 'man', 'syn')):
                tmp = {x.strip().split() for x in f.readlines()}
                for w in words:
                    if w in [x[0] for x in tmp]: print(f'{w} found in {lst1} as a sink')
                    if w in [y for x in tmp for y in x[1:]]: print(f'{w} found in {lst1} as a source')
            else:
                if w in {x.strip() for x in f.readlines()}: print(f'{w} found in {lst1}')
    elif lst1 and lst2:
        if lst1 not in names: raise ValueError(f'Invalid search target {lst1}')
        if lst2 not in names: raise ValueError(f'Invalid search target {lst2}')
        with open(f'{base}{names[lst1]}', 'rt') as f:
            tmp1 = {x.strip() for x in f.readlines()}
        with open(f'{base}{names[lst2]}', 'rt') as f:
            tmp2 = {x.strip() for x in f.readlines()}
        if any(lst1.startswith(y) for y in ('rep', 'man', 'syn')) and any(lst2.startswith(y) for y in ('rep', 'man', 'syn')):
            tmp1, tmp2 = [x.split() for x in tmp1], [x.split() for x in tmp2]
            t1snk, t2snk = [x[0] for x in tmp1], [x[0] for x in tmp2]
            if lst1.startswith('syn'): t1src = [y for x in tmp1 for y in x[1:] if all(z not in y for z in ('_', '~', '-', 'I'))]
            else: t1src = [x[1] for x in tmp1]
            if lst2.startswith('syn'): t2src = [y for x in tmp2 for y in x[1:] if all(z not in y for z in ('_', '~', '-', 'I'))]
            else: t2src = [x[1] for x in tmp2]
            for i, z in enumerate((t1snk, t1src)):
                iter_lst = 'sinks' if i == 0 else 'sources'
                for x in z:
                    if x in t2snk: print(f'{x} from {lst1} {iter_lst} found in {lst2} as a sink')
                    if x in t2src: print(f'{x} from {lst1} {iter_lst} found in {lst2} as a source')
        else:
            if any(lst1.startswith(y) for y in ('rep', 'man', 'syn')): multi_lst, foil, orig, dest = tmp1, tmp2, lst1, lst2
            elif any(lst2.startswith(y) for y in ('rep', 'man', 'syn')): multi_lst, foil, orig, dest = tmp2, tmp1, lst2, lst1
            else: multi_lst = False
            if multi_lst:
                multi_lst = [x.split() for x in multi_lst]
                snk = [x[0] for x in multi_lst]
                if orig.startswith('syn'): src = [y for x in multi_lst for y in x[1:] if all(z not in y for z in ('_', '~', '-', 'I'))]
                else: src = [x[1] for x in multi_lst]
                for x in snk:
                    if x in foil: print(f'{x} from {orig} sinks found in {dest}')
                for x in src:
                    if x in foil: print(f'{x} from {orig} sources found in {dest}')
            else:
                for x in tmp1:
                    if x in tmp2: print(f'{x} from {lst1} found in {lst2}')
    else:
        with open(f'{base}{names["syn_2"]}', 'rt') as f:
            check_list = {y.strip('+') for x in f.readlines() for y in x.strip().split()}
        temp = []
        with open(f'{base}{names["man_2"]}', 'rt') as f:
            for x in f.readlines():
                x = x.strip().split()
                if x[1] in check_list: print(f'Source in man_2; {x[1]} reduced prematurely')
                temp.append(x[0])
                temp.append(x[1])
        for x in temp: check_list.add(x)
        temp = []
        with open(f'{base}{names["syn_1"]}', 'rt') as f:
            for x in f.readlines():
                x = x.strip().split()
                for y in x[1]:
                    y = y.strip('+')
                    if y in check_list: print(f'Sourcec in syn_1; {y} reduced prematurely')
                    temp.append(y)
                temp.append(x[0])
        for x in temp: check_list.add(x)
        with open(f'{base}{names["man_1"]}', 'rt') as f:
            for x in f.readlines():
                x = x.strip().split()
                if x[-1] == '|': continue
                if x[1] in check_list: print(f'Source in man_1; {x[1]} reduced prematurely')

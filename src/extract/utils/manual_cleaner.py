
from .tools import Counter, load, dump, cprint
from ..classes.AffixAnalyzer import AffixAnalyzer
from ..classes.MClean import MCleaner


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
        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\manu{int(ld)}', 'rb') as f:
            wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk = load(f)
        print(f'\tLoaded {ld}:\n\t\tProgress: {len(wrd_q)} / {len(a.wlst)}\n\t\t{round(100*len(wrd_q)/len(a.wlst), 4)}%')
    else:
        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\manu_auto', 'rb') as f:
            wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk = load(f)
        print(f'Loaded AutoSave:\n\tProgress: {len(wrd_q)} / {len(a.wlst)}\n\t\t{round(100*len(wrd_q)/len(a.wlst), 4)}%')

    oln = len(a.full_scores)
    wrd_q = sorted(wrd_q, key=lambda x: len(x))
    a.verif = sorted(a.verif, key=lambda x: len(x), reverse=True)

    b = MCleaner(a.verif, a.ldct, a.wlst, a.roots, a.wparts, a.afxscore, a.failed_brk)
    del a
    a = b

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
                        idx = a.merger(res)
                        print(idx)
                        a.pretty_printer(idx)
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
                                    s_pack = a.packer(res, idx[1], inp)
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
                                    with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\manu_auto', 'wb') as f:
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
                                    with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\manu{select}', 'wb') as f:
                                        dump((wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk), f)
                                case _: wrd_q.append(word)

                        if not inp:
                            s_pack = a.packer(res, idx[1])
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
                                with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\manu_auto', 'wb') as f:
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
                                        idx = a.merger(tmp)
                                        found = False
                                        for q in idx:
                                            for qq in q:
                                                if isinstance(qq, tuple):
                                                    for qqq in qq:
                                                        if action[1] in qqq:
                                                            a.pretty_printer(idx)
                                                            print('\n')
                                                            found = True
                                                else:
                                                    if action[1] in qq:
                                                        a.pretty_printer(idx)
                                                        print('\n')
                                                        found = True
                                                if found: break
                                            if found: break
                            case 'try':
                                idx = a.merger(a.find_sub_chain(action[1], tmp_affixes))
                                if idx: a.pretty_printer(idx)
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
                                with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\manu{action[1]}', 'wb') as f:
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
                                        s_pack = a.packer(chain, a.merger(chain)[1])
                                        for i, x in enumerate(s_pack.items()):
                                            if x[0] in wrd_q:
                                                if i == 0: print(f'accepting {x[1][0]} {[y for y in s_pack]}')
                                                a.score_eval(x[0], x[1])
                                                wrd_q.remove(x[0])
                                with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\manu_auto', 'wb') as f:
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
                    with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\ppd2', 'rb') as f:
                        tid, tdct = load(f)
                    with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\fsort', 'rt') as f:
                        corpus = [x.strip() for x in f.readlines()]
                    code = 'pafxrts'
                    if len(sel) > 1 and sel[1] == 'l':
                        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\safxrts', 'rb') as f:
                            afx_hold, root_hold, index = load(f)
                        root_hold.extend(afx_hold)
                        ofound = root_hold.copy()
                        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\{code}', 'rb') as f:
                            afx_hold, root_hold, index = load(f)
                    else: afx_hold, root_hold, index = [], [], 0
                elif sel[0] == 'suf':
                    pre = False
                    with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\spd2', 'rb') as f:
                        tid, tdct = load(f)
                    with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\bsort', 'rt') as f:
                        corpus = [x.strip() for x in f.readlines()]
                    code = 'safxrts'
                    if len(sel) > 1 and sel[1] == 'l':
                        with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\pafxrts', 'rb') as f:
                            afx_hold, root_hold, index = load(f)
                        root_hold.extend(afx_hold)
                        ofound = [x for x in root_hold]
                        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\{code}', 'rb') as f:
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
                                with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\{code}', 'wb') as f:
                                    dump((afx_hold, root_hold, index), f)
 
                    else:
                        index += (8 if index+8 < len(tid)-1 else (len(tid)-2)-index)
                        agprinter(index)

                    if acnt >= 11:
                        with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\{code}-auto', 'wb') as f:
                            dump((afx_hold, root_hold, index), f)
                        acnt = 0

                with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\{code}', 'wb') as f:
                    dump((afx_hold, root_hold, index), f)

            case '7':
                with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\old\\manu_auto', 'wb') as f:
                    dump((wrd_q, a.wlst, a.roots, a.afxscore, a.wparts, a.failed_brk), f)

if __name__ == "__main__":
    main()
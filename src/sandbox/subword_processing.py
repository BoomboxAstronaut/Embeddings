import pickle

from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool


def sub_swap(entry, indices: list[int], rep: str):
    outp = entry.split()
    for i, idx in enumerate(indices):
        outp.insert(idx + i, rep)
    for i, idx in enumerate(indices):
        outp.pop(idx + 1 - i)
        outp.pop(idx + 1 - i)
    return (entry, ' '.join(outp))

def word_check(cpack):
    csub, cword, rep = cpack[0], cpack[1], cpack[2]
    if csub not in cword:
        return
    if f'{csub} ' != cword[:len(csub) + 1] and f' {csub}' != cword[len(cword) - len(csub) - 1:] and f' {csub} ' not in cword:
        return
    cword = cword.split()
    csub = csub.split()
    is_match = False
    idx = []
    for i, y in enumerate(cword[:-1]):
        if is_match:
            is_match = False
            continue
        if csub[0] == y and csub[1] == cword[i+1]:
            idx.append(i)
            is_match = True
    return sub_swap(' '.join(cword), idx, rep)


if __name__ == '__main__':

    with open(r'D:\dstore\nlp\w2v\wcounts16', 'rt', encoding='utf8') as f:
        wlst = [x.strip('\n').split() for x in f.readlines()]

    wlst = Counter({x[1]: int(x[0]) for x in wlst})

    #Get all individual letters
    lcount = Counter()
    for x in wlst:
        for y in x:
            lcount[y] += 1

    lcount = {x[0]: x[1] for x in lcount.items() if x[1] > 1023}
    subw = [*lcount, '_', "'", '[UNK]']


    #Modify word list to include border and spacing for tokenization

    wlst = Counter({' '.join(list(f'{x[0]}_')): x[1] for x in wlst.items()})

    #Replace low frequency with a common identifier [UNK]

    temp = []
    for x in wlst:
        xcp = x
        is_modded = False
        for y in x.split():
            if y not in subw:
                xcp = xcp.replace(y, '[UNK]')
                is_modded = True
        if is_modded:
            temp.append((x, xcp))

    for x in temp:
        wlst[x[1]] += wlst[x[0]]
        wlst.pop(x[0])

    del lcount
    del temp

    candidate_frags = Counter()
    for word in wlst:
        word = word.split()
        for i, _ in enumerate(word[:-1]):
            if word[i] != '[UNK]' and word[i+1] != '[UNK]':
                candidate_frags[f'{word[i]} {word[i+1]}'] += wlst[' '.join(word)]

    candidate_frags = Counter({x[0]: x[1] for x in candidate_frags.items() if x[1] > 255})

    for sub in tqdm(candidate_frags):
        rep = sub.replace(' ', '')
        wpack = [(sub, word, rep) for word in wlst]
        with Pool(processes=4) as pool:
            matches = list(pool.imap_unordered(word_check, wpack, chunksize=100000))
        matches = [x for x in matches if x is not None]
        #matches = [check for word in wlst if (check := word_check(sub, word, rep))]
        for m in matches:
            wlst[m[1]] += wlst[m[0]]
            wlst.pop(m[0])

    for x in candidate_frags:
        subw.append(x)

    with open(r'D:\dstore\nlp\w2v\wlst-1', 'wb') as f:
        pickle.dump(wlst, f)

    with open(r'D:\dstore\nlp\w2v\subw-1', 'wb') as f:
        pickle.dump(subw, f)


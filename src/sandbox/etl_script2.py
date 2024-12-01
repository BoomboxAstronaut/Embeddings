import pickle
from collections import Counter
from tqdm import tqdm
from multiprocessing import pool

with open(r'D:\dstore\nlp\w2v\fwords', 'rt', encoding='utf8') as f:
    wlst = [x.strip().split() for x in f.readlines()]
wlst = {x[1] for x in wlst if len(x[1]) > 2}
singles = {'s_', 'y_', '_a', '_e'}

def get_nested(subject):
    hln = len(subject) * 3
    hold = {x for x in wlst if x in subject and len(x) < hln}
    hold.remove(subject)
    if hold:
        return (subject, tuple(hold))
    return subject

def extract_afx(subject, group) -> list:
    hold = []
    group = [f'_{subject}_'.split(x) for x in group]
    for x in group:
        for y in x:
            if len(y) > 2 or y in singles:
                hold.append(y)
    return hold

if __name__ == "__main__":
    hold = dict()
    nons = []
    with pool.Pool(processes=6) as p:
        out = list(tqdm(p.imap_unordered(get_nested, wlst), total=len(wlst)))
        for x in out:
            if isinstance(x, str):
                nons.append(x)
            else:
                hold[x[0]] = x[1]
    outp = Counter()
    for subject in hold:
        out = extract_afx(subject, hold[subject])
        if out:
            for x in out:
                outp[x] += 1
    tfrags = Counter({x[0]: x[1] for x in outp.most_common() if x[1] > 2})

    with open(r'D:\dstore\nlp\w2v\tft6', 'wb') as f:
        pickle.dump((tfrags, nons), f)

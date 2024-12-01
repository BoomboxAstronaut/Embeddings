
from src.extract.classes.AffixAnalyzer import AffixAnalyzer
from src.extract.classes.Lexicon import Lexicon
from src.extract.classes.Word import Word


def setup():
    with open(r'D:\dstore\nlp\w2v\fwords', 'rt') as f:
        a = AffixAnalyzer([x.strip().split() for x in f.readlines()], 3)
    for x in 'id ax ox ab op ex by on to in'.split():
        a.wlst[f'_{x}_'] = a.cleared[f'_{x}_'] 

    with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\roots\\_root_ext', 'rt') as f:
        roots = [x.strip() for x in f.readlines()]

    with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\_base_remove', 'rt') as f:
        rmls = [x.strip() for x in f.readlines()]
        for x in rmls:
            if f'_{x}_' in a.wlst: a.wlst.pop(f'_{x}_')

    with open('C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\_base_add', 'rt') as f:
        als = [f'_{x.strip()}_' for x in f.readlines()]
        for x in als:
            if x not in a.wlst: a.wlst[x] = 100
            else: print(x, a.wlst[x])
    return a, roots

a, roots = setup()
a.assign_search_dict(a.bare)
wbs = Lexicon(set([x.strip('_') for x in a.wlst]))


with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\roots\\_root_ext', 'rt') as f:
    for x in [x.strip() for x in f.readlines()]:
        if x in wbs.bases:
            wbs.bases[x].mutable = False
        elif x in wbs.bt:
            wbs.bases[wbs.bt[x][0]].mutable = False
        else:
            wbs.bases[x] = Word(x)
            wbs.bases[x].mutable = False
            wbs.bt[x] = None

with open(f'C:\\Users\\BBA\\Coding\\NLP\\Embeddings\\data\\v0\\roots\\_root_syns', 'rt') as f:
    for x in [x.strip().split() for x in f.readlines()]:
        center = x[0]
        x = [y for y in x if y != center]
        for y in x:
            if y in wbs.bases and not wbs.bases[y].mutable: wbs.bases[y].mutable = True
            wbs.rform(center, y)

for x in wbs.mpx:
    if x not in wbs.bases: continue
    if not len(x) > 2: continue
    if x in wbs.bases:
        wbs.bases[x].mutable = False
    elif x in wbs.bt:
        wbs.bases[wbs.bt[x][0]].mutable = False
    else:
        wbs.bases[x] = Word(x)
        wbs.bases[x].mutable = False
        wbs.bt[x] = None

    


def test_s_sfx():
    assert wbs.rep_afx('fates', 1) == 'fate'

def test_ing_e_sfx():
    assert wbs.rep_afx('rating', 3, t='e') == 'rate'

def test_er_e_sfx():
    assert wbs.rep_afx('lamer', 2, t='e') == 'lame'

def test_ed_y_sfx():
    assert wbs.rep_afx('carried', 2, t='iy') == 'carry'

def test_ing_dbl_sfx():
    assert wbs.rep_afx('slamming', 3, t='dbl') == 'slam'

def test_ces_x_sfx():
    assert wbs.rep_afx('matrices', 3, r='x') == 'matrix'

def test_ves_f_sfx():
    assert wbs.rep_afx('elves', 3, r='f') == 'elf'

def test_ing_ck_sfx():
    assert wbs.rep_afx('panicking', 3, t='ck') == 'panic'

def test_p_ing_dbl_sfx():
    assert wbs.rep_afx('unjamming', 3, t='dbl', pref=True)[0] == 'jam'

def test_p_ed_y_sfx():
    assert wbs.rep_afx('interlevied', 2, t='iy', pref=True)[0] == 'levy'

def test_aggressive():
    assert wbs.rep_afx('aggressive', 3, pref=True)[0] == 'gress'

def test_er():
    assert wbs.rep_afx('central', 2, t='er') == 'center'

def test_fv():
    assert wbs.rep_afx('believes', 1, t='fv') == 'belief'

def test_fv_empty():
    assert wbs.rep_afx('believe', 0, t='fv') == 'belief'

def test_icix():
    assert wbs.rep_afx('matrices', 2, t='icix') == 'matrix'

def test_icex():
    assert wbs.rep_afx('vortices', 2, t='icex') == 'vortex'

def test_ctx():
    assert wbs.rep_afx('climactic', 2, t='ctx') == 'climax'

def test_cte():
    assert wbs.rep_afx('production', 3, t='cte') == 'produce'

def test_merge():
    assert wbs.rep_afx('aerial', 3, flex=True) == 'aero'
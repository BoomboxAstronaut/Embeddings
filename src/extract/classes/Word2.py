

class Word:
    __slots__ = ('w', 'syns', 'acomp', 'bcomp', 'homos', 'pos', 'tid', 'exc')
    """
    acomp: Words that use this word as a component
    bcomp: Word components that compose this word
    syns: Words that carry identical meaning
    homos: Identical words with different meaning
    pos: Part of speech. Noun Verb etc
    exc: Exclude from auto decomposition

    tids:
        m0: default:    Can be independant. Can be decomposed.
        m1: root:       Can be independant. Can not be decomposed.
        m2: affix:      Can not be independant (exceptions). Can not be decomposed.
        m3: synonym:    Can be independant. Can not be decomposed.
        m4: transitive: Can not be independant. Can be decomposed.
    """

    def __init__(self, word):
        self.w = word
        self.syns = []
        self.acomp = []
        self.bcomp = []
        self.homos = []
        self.tid = 0
        self.pos = 'noun'
        self.exc = False

    def __len__(self):
        return len(self.w)

    def __repr__(self):
        return ' '.join(self.bcomp)

    def __str__(self):
        return self.w

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

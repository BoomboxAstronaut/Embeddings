

class Word:
    __slots__ = ('w', 'flex', 'all', 'forms', 'mods', 'alias', 'nmut', 'noun', 'verb', 'adj', 'adv', 'part')
    """
    core: words that use this word as a root
    mods: words that use this word to modify another
    aliases: alternate names that are not extensible. nicknames, acronyms, foreign languages
    forms: conjugations and alternate spellings of the same root. merging of nodes
    interwords: words that can be further broken down but remain in the reference set to help with breaking down other words
    merge: form a word must take when combined with other words
    m0: default
    m1: root: Cannot be modified
    m2: affix: Modifies words, cannot be modified, can only be merged into other affixes or roots
    m3: alias: Can only be merged
    m4: transitive: Can not be used on its own
    """

    def __init__(self, word):
        self.w = word
        self.flex = word
        self.all = []
        self.mods = []
        self.forms = []
        self.alias = []
        self.nmut = 0
        self.noun = False
        self.verb = False
        self.adj = False
        self.adv = False

    def __len__(self):
        return len(self.all)

    def __repr__(self):
        return self.w

    def __str__(self):
        if len(self.all) == 0: return self.w
        else:
            ostr = f'{self.w}'
            for x in self.__slots__[2:6]:
                if getattr(self, x):
                    ostr += f'\n    {x}: {getattr(self, x)}'
            return ostr

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

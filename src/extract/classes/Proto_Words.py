
from typing import Protocol



class Word:
    __slots__ = ('w', 'all', 'core', 'forms', 'mods', 'aliases', 'root', 'flexend', 'plur', 'pres', 'pprg', 'past')
    """
    core: words that use this word as a root
    mods: words that use this word to modify another
    aliases: alternate names that are not extensible. nicknames, acronyms, foreign languages
    forms: conjugations and alternate spellings of the same stem
    """

    def __init__(self, word):
        self.w = word
        self.all = [word]
        self.mods = []
        self.modded = []
        self.aliases = []
        self.forms = []
        self.flexend = False
        self.mutable = False
        self.plur = None
        self.pres = None
        self.pprg = None
        self.past = None

    def __len__(self):
        return len(self.all)

    def __repr__(self):
        if len(self.all) == 1: return self.w
        else: return f'{self.all[::-1]}'

    def __str__(self):
        if len(self.all) == 1: return self.w
        else:
            ostr = f'{self.w}'
            for x in [x for x in self.__slots__[2:] if x not in ('flexend', 'root') and getattr(self, x)]:
                ostr += f'\n    {x}: {getattr(self, x)}'
            return ostr

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def adg(self, k, v):
        if v:
            attr = getattr(self, k)
            if isinstance(v, (str, Word)):
                if isinstance(attr, (str, Word)): setattr(self, k, (v, attr))
                elif not attr or len(attr) == 0: setattr(self, k, (v,))
                else: setattr(self, k, (v, *attr))
            else:
                if isinstance(attr, (str, Word)): setattr(self, k, (*v, attr))
                elif not attr or len(attr) == 0: setattr(self, k, v)
                else: setattr(self, k, (*v, *attr))


class Verb(Protocol):

    present_prog: Word
    plural: Word
    past: Word


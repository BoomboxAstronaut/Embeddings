
class Wtree:

    def __init__(self, trunk):
        self.root = trunk[0]
        self.tree = {'word': self.root, 'source': [None], 'path': [self.root],
            trunk[1]: {'word': trunk[2], 'source': [self.root], 'path': [self.root, trunk[1], trunk[2]]}}
        self.active = {self.root: self.tree, trunk[2]: self.tree[trunk[1]]}

    def attach(self, branch):
        if branch[2] in self.active:
            #Connect Branches
            self.active[branch[0]][branch[1]] = self.active[branch[2]]
            self.active[branch[2]]['source'].append(branch[0])
            opath = [x for x in self.active[branch[0]]['path']]
            if isinstance(opath[0], list):
                opath = [[y for y in x] for x in opath]
                for i in range(len(opath)): opath[i].extend([branch[1], branch[2]])
                if isinstance(self.active[branch[2]]['path'][0], list):
                    for x in self.active[branch[2]]['path']: opath.append(x)
                else: opath.append(self.active[branch[2]]['path'].copy())
                self.active[branch[2]]['path'] = opath
            else:
                opath.extend([branch[1], branch[2]])
                if isinstance(self.active[branch[2]]['path'][0], list): self.active[branch[2]]['path'].append(opath)
                else: self.active[branch[2]]['path'] = [self.active[branch[2]]['path'].copy(), opath]
        else:
            #New Branch
            self.active[branch[0]][branch[1]] = {'word': branch[2], 'source': [branch[0]]}
            self.active[branch[2]] = self.active[branch[0]][branch[1]]
            opath = [x for x in self.active[branch[0]]['path']]
            if isinstance(opath[0], list):
                opath = [[y for y in x] for x in opath]
                for i, _ in enumerate(opath): opath[i].extend([branch[1], branch[2]])
            else: opath.extend([branch[1], branch[2]])
            self.active[branch[2]]['path'] = opath
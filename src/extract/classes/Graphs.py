
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from os import environ

plt.style.use(f"{environ['style']}")


class Entropy:

    def __init__(self, afx, ldct, wlst):
        self.afx = afx
        self.ldct = ldct
        self.wlst = wlst
        self.base_wgt = [[1, 1, 1.25], [1, 1.25, 1.5], [1.25, 1.5, 1.75]]
        self.dbg = False

    def prep_entropy_calc(self, over_length: int=7, pull_cutoff: int=2, wgts: list[list[int]]=None) -> None:
        """
        Goal: 
            To filter out fragments of affixes from whole affixes
        Hypothesis: 
            The distribution of letters adjacent to an affix will help me determine whether or not an affix is whole or not.
            Partial affixes will have much lower entropy in atleast one of the measurements because the letter that completes the affix will dominate the distribution.
            Limiting the sampling window size will give amplify the entropies.
        Example: 
            ng_ is a partial affix of ing_.
            When sampling letters to the left of ng_, the letter 'i's dominate the distribution.
            When the window size of is set to 1, the distribution of letters will be almost entirely 'i' giving a very high relative entropy value.
            In contrast with the distribution of ing_ the distribution will be much closer to the general distribution of the whole data set
        Args:
            over_length (int, optional): Maximum length for standard affixes. Affixes longer than this value will have their character distributions separated. Should be slightly over half the length of the average word. Defaults to 7.
            pull_cutoff (int, optional): Maximum difference in affix length when searching for parent / child affix nodes. Defaults to 2.
            wgts (list[list[int]], optional): 3x3 Weight matrix for scaling direction and window size of relative entropy calculations. Defaults to [[1, 1, 1.25], [1, 1.25, 1.5], [1.25, 1.5, 1.75]].
        Returns:
            re_arr, dsts, rntp, drntp dictionaries added to instance
        """
        wgts = self.base_wgt
        dsts, pd, sd, nd = {}, Counter(), Counter(), Counter()
        for x in self.wlst: # Get letter distributions for: letters in front half of words, letters in back half of words, all letters
            x = x.strip('_')
            for l in x: nd[l] += 1
            i = round((len(x)+0.1) / 2)
            for l in x[:i]: pd[l] += 1
            for l in x[-i:]: sd[l] += 1
        pd = Counter({x[0]: x[1] / pd.total() for x in pd.most_common()})
        sd = Counter({x[0]: x[1] / sd.total() for x in sd.most_common()})
        nd = Counter({x[0]: x[1] / nd.total() for x in nd.most_common()})

        rntp = {} # Calculate the relative entropy of letters adjacent to an affix
        for x in tqdm(self.afx): 
            hold = []
            if '_' not in x:
                fd = nd
                for i in range(1, 4): #Define sampling window size
                    o1, o2 = self.surrounds(x, i)
                    hold.append([self.kld(self.surrounds(x, i, merge=True), fd), self.kld(o1, fd), self.kld(o2, fd)])
            else:
                if x[0] == '_': pre = True
                else: pre = False
                if (pre and len(x) > over_length) or (not pre and len(x) <= over_length): fd = sd
                else: fd = pd
                for i in range(1, 4):
                    o1, o2 = self.surrounds(x, i) # Define direction of window here
                    if pre: side = o2 
                    else: side = o1
                    hold.append([self.kld(self.surrounds(x, i, merge=True), fd), self.kld(side, fd), self.kld(self.surrounds(x, i, exact=True), fd)])
            rntp[x] = np.array(hold[::-1]).T

        re_arr, hold = {}, [] # Get letter distributions for the first and last: 1, 2, 3 letters
        for i in range(1, 4):
            frel, brel = Counter(), Counter()
            for x in self.wlst:
                if len(x) >= 3+i:
                    x = x.strip('_')
                    for l in x[-i:]: brel[l] += 1
                    for l in x[:i]: frel[l] += 1
            hold.append([self.kld(frel, pd), self.kld(brel, sd)])
            dsts[f'pd{i}'] = Counter({x[0]: x[1] / frel.total() for x in frel.most_common()})
            dsts[f'sd{i}'] = Counter({x[0]: x[1] / brel.total() for x in brel.most_common()})
        hold = [[x]*3 for x in np.array(hold[::-1]).T]
        #These will be used for calculating the derivatives of root/leaf affixes
        re_arr['pd3'], re_arr['sd3'] = np.array(hold[0]), np.array(hold[1])
        re_arr['lpd3'], re_arr['lsd3'] = np.array([rntp[x] for x in self.afx if len(x) > over_length and x[0] == '_']).mean(axis=0), np.array([rntp[x] for x in self.afx if len(x) > over_length and x[-1] == '_']).mean(axis=0)

        drntp = {}
        for x in tqdm(self.afx): # Derivative of relative entropy values along a sequential chain of affixes
            if '_' in x: #Chains can only beformed with positional affixes
                above, below = self.pulld(x, pull_cutoff), self.pullu(x)
                if above: above = np.array([rntp[y] for y in above]).mean(axis=0)
                else: #If an affix has no affixes above it, use the averaged relative entropies for affixes longer than 6 letters to calculate
                    above = self.pulld(x)
                    if above: above = np.array([rntp[y] for y in above]).mean(axis=0)
                    elif x[0] == '_': above = re_arr['lpd3']
                    elif x[-1] == '_': above = re_arr['lsd3']
                if below: below = rntp[below]
                elif x[0] == '_': below = re_arr['pd3']
                else: below = re_arr['sd3']
                middle = rntp[x]
                drntp[x] = (above-middle) - (middle-below)

        #Find the average derivatives for edge cases
        wgts = np.array(wgts)
        re_arr['lpd3x'], re_arr['lsd3x'] = re_arr['lpd3']*wgts, re_arr['lsd3']*wgts
        re_arr['pd3x'], re_arr['sd3x'] = re_arr['pd3']*wgts, re_arr['sd3']*wgts
        re_arr['dlpd3'], re_arr['dlsd3'] = np.mean([(rntp[x] * wgts) - re_arr['lpd3x'] for x in self.afx if x[0] == '_' and len(x) > over_length], axis=0), np.mean([(rntp[x] * wgts) - re_arr['lsd3x'] for x in self.afx if x[-1] == '_' and len(x) > over_length], axis=0)
        re_arr['dpd3'], re_arr['dsd3'] = np.mean([(rntp[x] * wgts) - re_arr['pd3x'] for x in self.afx if x[0] == '_' and len(x) <= over_length], axis=0), np.mean([(rntp[x] * wgts) - re_arr['sd3x'] for x in self.afx if x[-1] == '_' and len(x) <= over_length], axis=0)
        dsts['pd'], dsts['sd'], dsts['nd'] = pd, sd, nd
        re_arr['wgts'], re_arr['null'] = wgts, [np.array([[0]*3]*3), np.array([[0]*3]*3)]
        self.re_arr, self.dsts, self.rntp, self.drntp = re_arr, dsts, rntp, drntp



    def remean(self, rearr: np.ndarray) -> np.ndarray:
        #Returns the mean array of the input array for each column and row
        return np.array([*[np.mean(y) for y in rearr], *[np.mean(y) for y in rearr.T]])

    def kld(self, P: Counter, Q: Counter=None, pfloor: int=0) -> float:
        """
        Kullback Leibler Divergence Calculation

        Args:
            P (Counter): Counts of letters
            Q (Counter, optional): Letter counts or distribution to compare against P. Defaults to letter distribution of entire the word list.
            base_value (int, optional): Base count of all letters. Higher values reduces effect of 0s. Defaults to 3.

        Returns:
            float: Relative Entropy of the two counts / distributions.
        """
        if not pfloor:
            if P.total() < 156: pfloor = 1
            elif P.total() < 312: pfloor = 2
            else: pfloor = 3
        if not Q: Q = self.dsts['nd']

        pcnt = Counter({x: pfloor for x in self.ldct['alpha']})
        for x in P:
            for y in x: pcnt[y] += P[x]  
        psum = sum(x for x in pcnt.values())
        if Q.total() > 1.5:
            qcnt = Counter({x: pfloor for x in self.ldct['alpha']})
            for x in Q:
                for y in x: qcnt[y] += Q[x]
            for x in self.ldct['alpha']:
                if pcnt[x] == pfloor and qcnt[x] == pfloor:
                    pcnt.pop(x)
                    qcnt.pop(x)
            qsum = sum(x for x in qcnt.values())
            return sum([(pcnt[x] / psum) * np.log2((pcnt[x] / psum) / (qcnt[x] / qsum)) for x in pcnt])
        else: return sum([(pcnt[x] / psum) * np.log2((pcnt[x] / psum) / Q[x]) for x in pcnt])

    def relent_peaks(self, afx: str, bridge_coeff: float=1.0, over_length: int=7) -> list[str, float]:
        """
        Find affixes with significant variations from their parent/child nodes relative entropy.
        Affix chain will be the longest chain that contains the input affix.
        Specific chains can be targetted by inputting the longest affix of a chain.
        Affixes returned indicate a target of interest for removal.

        Args:
            afx (str): Target affix
            bridge_coeff (int, optional): Change in relative entropy to be considered significant. Defaults to 1.0.
            over_length (int, optional): Maximum length for standard affixes. Affixes longer than this value will have their character distributions separated. Should be slightly over half the length of the average word. Defaults to 7.

        Returns:
            list[str, float]: List of affixes with significant relative entropy spikes
        """
        if afx[0] == '_':
            if len(afx) > over_length: scores = [self.re_arr['lpd3'].copy(), self.re_arr['lpd3'].copy()]
            else: scores = [self.re_arr['pd3'].copy(), self.re_arr['pd3'].copy()]
        elif afx[-1] == '_':
            if len(afx) > over_length: scores = [self.re_arr['lsd3'].copy(), self.re_arr['lsd3'].copy()]
            else: scores = [self.re_arr['sd3'].copy(), self.re_arr['sd3'].copy()]

        words = self.chain(afx)
        for x in words: scores.append(self.rntp[x] * self.re_arr['wgts'])
        if afx[0] == '_': scores.append(self.re_arr['pd3'])
        else: scores.append(self.re_arr['sd3'])
        hold = [*self.re_arr['null'].copy()]
        for i in range(1, len(scores)-1): hold.append((scores[i+1]-scores[i])+(scores[i-1]-scores[i]))

        if afx[0] == '_':
            if len(afx) > over_length: hold.extend([self.re_arr['dlpd3'], self.re_arr['dlpd3']])
            else: hold.extend([self.re_arr['dpd3'], self.re_arr['dpd3']])
        elif afx[-1] == '_':
            if len(afx) > over_length: hold.extend([self.re_arr['dsd3'], self.re_arr['dsd3']])
            else: hold.extend([self.re_arr['dsd3'], self.re_arr['dsd3']])

        out = []
        for i in range(2, len(hold)-3):
            u1, d1, md = hold[i-1].copy(), hold[i+1].copy(), hold[i]
            if self.dbg: print(words[i-2], (md-u1).mean(), (md-d1).mean())
            if (md-u1).mean() > bridge_coeff or (md-d1).mean() > bridge_coeff:
                u2, d2 = hold[i-2].copy(), hold[i+2].copy()
                u2[u2 > u1] *= 0
                u1[u1 > hold[i-2]] *= 0
                d2[d2 > d1] *= 0
                d1[d1 > hold[i+2]] *= 0
                u1 = u1 + u2
                d1 = d1 + d2
            out.append((md-u1)+(md-d1))

        if self.dbg:
            for i, x in enumerate(out): print(words[i], '\n', self.remean(x).mean(), '\n', x)
        return [(words[i], self.remean(x).mean()) for i, x in enumerate(out)]

    def pulld_relent(self, afx: str, depth: int=1) -> np.ndarray:
        #Returns the mean relative entropy of all child nodes of the input affix
        hold = [afx]
        while depth > 0:
            grp = []
            while hold:
                tmp = self.pulld(hold.pop(), 0)
                if tmp:
                    for y in tmp:
                        grp.append(y)
            hold.extend(grp)
            depth -= 1
        if hold:
            hold = np.mean([self.rntp[x] for x in hold], axis=0)
            return np.array([*[np.mean(x) for x in hold], *[np.mean(x) for x in hold.T]])
        else: return np.array([0]*6)

    def graph_relent(self, afx: str):
        words = self.chain(afx)
        _, ax = plt.subplots(figsize=(16, 10))
        yvars = [[*[np.mean(x) for x in self.rntp[w]], *[np.mean(x) for x in self.rntp[w].T], *[np.mean(x) for x in self.drntp[w]], *[np.mean(x) for x in self.drntp[w].T]] for w in words]

        plt.xticks(range(len(yvars)), words)
        ax.plot(range(len(yvars)), yvars)
        ax.legend(['arnd', 'dir', 'exct', 'wnd3', 'wnd2', 'wnd1', 'darnd', 'ddir', 'dexct', 'dwnd3', 'dwnd2', 'dwnd1'])
        plt.show()
import numpy as np
import pickle

from multiprocessing import Pool
from tqdm import tqdm


rng = np.random.default_rng()

def sen_to_data(sentence: list[int]) -> tuple[int]:
    
    senlen = len(sentence)
    hold = []
    for i, x in enumerate(sentence):
        #6 skip distance
        temp = []
        ioptions = [*range(max(i - 6, 0), i), *range(i+1, min(i + 6, senlen))]
        for _ in range(4):
            wchoice = rng.integers(0, 151156)
            while wchoice in sentence:
                wchoice = rng.integers(0, 151156)
            temp.append(sentence[rng.choice(ioptions)])
            temp.append(wchoice)
        hold.append((x, *temp))
    return hold


if __name__ == "__main__":
    
    with open('D:\\dstore\\nlp\\w2v\\word_list_i', 'rb') as f:
        all_words = pickle.load(f)

    for z in range(1, 5):
        with open(f'D:\\dstore\\nlp\\w2v\\vtrain{z}', 'rb') as f:
            warray = pickle.load(f)
        warray = [[all_words[y] for y in x] for x in warray]
        halfpoint = len(warray) / 2
        split_point = int(len(warray) / 64)
        tdata = []

        for i in range(1, 65):
            group = []
            tracker = 0
            if i == 65:
                split_point = len(warray)
            while tracker < split_point:
                group.append(warray.pop())
                tracker += 1

            with Pool(processes=6) as pool:
                gdata = list(tqdm(pool.imap_unordered(sen_to_data, group, chunksize=1024), total=len(group)))

            for g in gdata:
                for f in g:
                    tdata.append(f)
            del group, gdata

            if i == 32:
                with open(f'D:\\dstore\\nlp\\w2v\\train{z}-{1}', 'wb') as f:
                    np.save(f, tdata)
                tdata = []

        with open(f'D:\\dstore\\nlp\\w2v\\train{z}-{2}', 'wb') as f:
            np.save(f, tdata)

        del warray, tdata

Problem:
    Algorithm used for extracting subwords did not feel satisfactory.

Goal:
    Reduce the amount of words in the dictionary by decomposing english into the components (affixes) that give meaning to a word
    Target affixes that are most frequently reused and carry meaning consistently
    Final product is a dictionary that will be able to break down any word into its components while maintaining information integrity

Terms
Affix: 
    A word or fragment of a word that carries meaning.
        - Words can be composed entirely of affixes, or out of affixes and a root, or exist only as a root
        - An affix must carry consistent meaning
        - Denoted as a prefix by a underscore on the left or as a suffix by a underscore on the right '_pre' | 'ing_'
Parsing: 
        finding all cases of an affix, counting them and extracting from their root word appropriately, then returning the word to the word list
    OR: finding a subset of words and identifying and counting all affixes before removing the word from the word list

Subgoals:
    Create a list of affixes that compose the english language from a list of english words
    Retain as much information as possible while transitioning the word list to the affix list
    Nested affixes in the affix list should be seperated
    The final affix list should be able to compose most words in english
    Words should not be broken down into affixes if it destroys the meaning of the word
    Words should remain as whole words regardless of length if it cannot be broken down
    The algorithm should accomplish its goal with very little manual intervention

Procedures:
    1) Isolate commonly used words that are affixes to manually parse. Words for orientation are very common (s_ ing_ ed_ est_ er_ | _up _down _over _near _side _under)
    2) Create rules for affix extraction so that the remaining word will be in the most commonly found state (extract ing_ from _writing_ should yield _write_ not _writ_)
    3) Manually parse all contractions (words with ' (I'll / it's)) and remove all remaining words with apostrophes
    4) Create fragments of words by sliding windows of size 2..9 over a word (observing a word fragment) and adding the word count to the tally for that fragment
    5) Parse occuring more than 3000k times are moved to end dictionary
    6) Parse occuring more than 100k times with 4 or less letters are moved to end dictionary
    7) Parse with 3 or less letters are moved to end dictionary
    8) Create a subset of single character affixes were manually identified but keep them in the affix list
    9) Remove affixes that have no vowels (other than the single letter affixes) as the do not carry meaning
    10) Filter out non-affix fragments (no _ indicator) that are length 2 or less
    11) FIlter out non-affix fragments that do not have an affixed version (mip never occurs at the beginning or end of a word in the vocab and is not likely to be an affix)
    12) Filter out non-affix fragments when the sum of their affixed versions occur far more often frequently
    13) Apply affix counting algorithm on the affix list to amplify signal of true affixes
    14) Iterate through fragments, filtering for fragments that only appear in 2 or less words, and remove from the affix list
    15) SVD:
            Create a matrix where each column represents an affix and each row represents a word.
            Each element will be binary. 1 indicating that a affix is present in a word, 0 indicating not present

Problems:
    Identifying the appropriate cutoff point for an affix series 's_', 'es_', 'ities_' all carry meaning and occur uniquely but 'ties_' does not
        Cannot cut off all branches after finding a fragment that carries no information
    Counts are not reliable in all cases. Affixes that carry meaning could occur infrequently but consistently.
    False extractions occur 't_' is an affix for _burnt_ but not for _beat_. How to discriminate?
    False extractions can stop the proper extraction from occuring. extracting s_ _from viruses_ leaves _viruse_ which means es_ cannot be extracted
    Nested ruled affixes. 'ities_' is two affixes. 'ies_' and 'ity_'
    Uncertainty. is '_a' or '_ab' the affix for '_abbreviation_'
    Count uncertainty. 'ng_' occurs more than 'ing_' but 'ing_' extracts leaving a coherent root word

Tasks
Eliminate nested affixes
Ruled replacement
Affix Tree
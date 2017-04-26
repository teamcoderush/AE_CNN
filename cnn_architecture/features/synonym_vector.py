from nltk.corpus import wordnet as wn
from PyDictionary import PyDictionary

# Created by Dulanjaya Tennekoon
# Generates a synonym matrix from a sentence

EMPTY = '<EMPTY/>'
PAD_WORD = '<PAD/>'


# Generate top 3 synonyms of a given word using wordnet
def _generate_synonyms_wn(word):
    synonyms = set()

    for syn in wn.synsets(word):
        for l in syn.lemmas():
            if l.name() != word:
                synonyms.add(l.name())

    return [] if not synonyms else list(synonyms[:3])


# Generates the top 3 synonyms of a given word using pydictionary
def _generate_synonyms_pydic(word):
    dictionary = PyDictionary()
    lst = dictionary.synonym(word)
    return lst[:3] if isinstance(lst, list) else []


# Generates a matrix of synonyms for each word in each matrix
def generate_sentence_matrix(sentence):
    matrix = []

    for w in sentence:
        if w == PAD_WORD :
            lst= [PAD_WORD] * 4
        else:
            lst = _generate_synonyms_pydic(w)

            if len(lst) == 0:
                lst.append(EMPTY)
                lst.append(EMPTY)
                lst.append(EMPTY)
            elif len(lst) == 1:
                lst.append(EMPTY)
                lst.append(EMPTY)
            elif len(lst) == 2:
                lst.append(EMPTY)

            lst.insert(0,w)

        matrix.append(lst)
    print(matrix)
    return matrix

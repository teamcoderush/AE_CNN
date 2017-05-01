from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Created by Dulanjaya Tennekoon
# Returns the bi-gram of cleared set of sentences.


# Returns the matrix of bigrams.
# Input the words and the
def _find_bigram(words, score_fn=BigramAssocMeasures.chi_sq, n = 200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bigrams


# Returns the matrix of bigrams
# Example:  generate_bigrams('This is a test run')
# or:       generate_bigrams(['this','is','a', 'test', 'run']
#
# Returns:  [('This', 'is'), ('a', 'test'), ('is', 'a'), ('test', 'run')]
def generate_bigrams(sentence):
    if isinstance(sentence, list):
        return _find_bigram(sentence)
    elif isinstance(sentence, str):
        return _find_bigram(sentence.split())
    else: return []
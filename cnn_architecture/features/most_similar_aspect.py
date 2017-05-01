from nltk.corpus import wordnet as wn


# Created by Dulanjaya Tennekoon
# Finds the most similar aspect of each word using similarity value

class MostSimilarAspectGenerator:
    _path = ''
    _syn_dic = []
    _h_value = 0.0
    _h_aspect = None

    def __init__(self):
        self._load_aspect_dictionary()

    # loads the aspect dictionary
    def _load_aspect_dictionary(self, _path=_path):
        _dictionary_words = ['Food', 'Color', 'Odour', 'Service']
        for aspect in _dictionary_words:
            self._syn_dic.append(self._get_synset(aspect))
        print('Aspect Dictionary is Created: ')
        print(self._syn_dic)

    # generates the synset of a word
    # returns the top synset of the word
    def _get_synset(self, word):
        synsets = wn.synsets(word)
        return synsets[0] if len(synsets) > 0 else None

    # Gives the most similar aspect of a word
    def _get_most_similar_aspect_per_word(self, syn_word):
        h_value = 0.0;
        h_aspect = '';

        for aspect in self._syn_dic:
            cur = syn_word.path_similarity(aspect)
            if cur is None:
                continue
            if h_value < cur:
                h_value = cur
                h_aspect = aspect

        return h_aspect

    # Gives the most similar aspect of a sentence
    def _get_most_similar_aspect_per_word(self, syn_word):

        for aspect in self._syn_dic:
            cur = syn_word.path_similarity(aspect)
            if cur is None:
                continue
            if self._h_value < cur:
                self._h_value = cur
                self._h_aspect = aspect

        return

    # Gives the most similar aspect of the entire sentence
    def generate_most_similar_aspect(self, sentence):
        self._h_value = 0.0
        self._h_aspect = None

        for word in sentence.split():
            syn_word = self._get_synset(word)
            if syn_word is not None:
                self._get_most_similar_aspect_per_word(syn_word)

        return self._h_aspect

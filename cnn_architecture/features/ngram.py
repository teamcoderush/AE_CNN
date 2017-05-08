

# Created by Dulanjaya Tennekoon


# returns the bigrams array of embeddings
def get_bigrams(embeds, s_sentence = 100):
    bigrams_vector = []
    bigrams_w = []
    for sentence in embeds:
        for a, b in zip(sentence,sentence[1:s_sentence]+sentence[0:1]):
            # print(a,b)
            bigrams_w.append(a+b)
        bigrams_vector.append(bigrams_w)
        bigrams_w = []
    return bigrams_vector

def get_trigrams(embeds, s_sentence = 100):
    trigrams_vector = []
    trigrams_w = []
    for sentence in embeds:
        for a, b, c in zip(sentence, sentence[1:s_sentence] + sentence[0:1], sentence[2:s_sentence] + sentence[0:2]):
            # print(a,b)
            trigrams_w.append((a + b + c))
        trigrams_vector.append(trigrams_w)
        trigrams_w = []
    return trigrams_vector



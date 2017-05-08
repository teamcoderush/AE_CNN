from nltk import pos_tag as pt


# Created by Dulanjaya Tennekoon

class PosTag:
    embeddings = None

    def __init__(self):
        self._create_embeddings()

    def _create_embeddings(self):
        words = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

        self.embeddings = {}

        i = 0

        for w in words:
            # e =  '0' * 35
            # e = e[:i] + '1' + e[i:]
            e = [0]*35
            e.insert(i,1)
            self.embeddings[w] = e
            i += 1

        return

    # returns the pos tag
    # example: generate_pos_tag(['This', 'is', 'a', 'test', 'run'])
    # returns: [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN'), ('run', 'NN')]
    def generate_pos_tag(self, sentence):
        # print(sentence)
        # print(sentence.index('<PAD/>'))
        tags = pt(sentence)
        w = []
        for p in tags:
            # print(self.embeddings[p[1]])
            # print(p, p[0],p[1])
            if p[1] not in self.embeddings:
                w.append([0]*36)
                continue
            w.append(self.embeddings[p[1]])

        return w
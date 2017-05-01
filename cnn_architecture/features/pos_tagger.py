from nltk import pos_tag as pt

# Created by Dulanjaya Tennekoon
# Gives the POS tags of a sentence


# returns the pos tag
# example: generate_pos_tag(['This', 'is', 'a', 'test', 'run'])
# returns: [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN'), ('run', 'NN')]
def generate_pos_tag(sentence):
    return pt(sentence)
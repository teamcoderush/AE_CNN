import numpy as np
import re
import itertools
import pandas as pd
import  nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer


"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/pos.txt").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/neg.txt").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>", max_length = None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if max_length == None:
        max_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = max_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def pad_sentence(sentence, padding_word="<PAD/>", length=45):
    """
    Pads all sentences to the same length. The length is given by the user(default=45).
    Returns padded sentences.
    """
    padded_sentences = []
    num_padding = length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)
    return padded_sentences


def pad_sentences_repeated(sentences, max_length = None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if max_length == None:
        max_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        padding = (max_length//len(sentence)) + 1
        new_sentence = (sentence * padding)[:max_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def build_input_data_for_sentences(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x


def load_data(train_data_path, test_data_path=None):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    if test_data_path==None:
        # Load and preprocess data
        sentences, labels = load_data_multilabel(train_data_path)
        # sentences, labels = load_data_and_labels()
        sentences_padded = pad_sentences(sentences) #required to emmit 0 as the pad key word
        vocabulary, vocabulary_inv = build_vocab(sentences_padded)
        x, y = build_input_data(sentences, labels, vocabulary)
        return [x, y, vocabulary, vocabulary_inv]
    else:
        # Load and preprocess data
        train_x, train_y = load_data_multilabel(train_data_path)
        test_x, test_y = load_data_multilabel(test_data_path)
        train_x_padded = pad_sentences(train_x) #required to emmit 0 as the pad key word
        test_x_padded = pad_sentences(test_x) #required to emmit 0 as the pad key word

        vocabulary, vocabulary_inv = build_vocab(train_x_padded+test_x_padded)
        train_x, train_y = build_input_data(train_x, train_y, vocabulary)
        test_x, test_y = build_input_data(test_x, test_y, vocabulary)
        return [train_x, train_y, test_x, test_y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def my_get_input_sentence(raw, length):
    # raw = input("input a news headline: ")
    raw_comment_cut = raw.split()
    sentence_padded = pad_sentence(raw_comment_cut, length=length)
    vocabulary, vocabulary_inv = build_vocab(sentence_padded)
    x = build_input_data_for_sentences(sentence_padded, vocabulary)
    return x


# newly added
# -----------------------------------------
def load_data_multilabel(data_path):
    """"Reads from csv and preprocess the dataset"""
    df = pd.read_csv(data_path, encoding='latin1')
    df.loc[(df['category'].isnull()), 'category'] = 'NO#ASPECT'
    df = df.drop_duplicates(['text', 'category'], keep='first')

    text = df.text.unique()
    text = sorted(text)

    labels = df.groupby('text')['category'].apply(list)


    index = labels.index.tolist()

    # check for mismatch in review order
    for i in range(len(text)):
        if text[i] != index[i]:
            print(text[i], index[i], i)
            raise ValueError('The order of reviews and labels is not correct')

    cleaned_text = [clean_reviews(sent,remove_stop_words=False) for sent in text]
    labels = labels.tolist()

    empty_reviews = [i for i, x in enumerate(cleaned_text) if x == ""]

    for i in empty_reviews:
        # if labels[i] != "NO#ASPECT":
        cleaned_text[i] = clean_reviews(text[i], remove_stop_words=False)
        # else:
        #     cleaned_text[i] = 'empty'

    vec_dic = {
        'RESTAURANT#GENERAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'RESTAURANT#PRICES': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'RESTAURANT#MISCELLANEOUS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'DRINKS#STYLE_OPTIONS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'DRINKS#PRICES': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'DRINKS#QUALITY': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'FOOD#PRICES': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'FOOD#STYLE_OPTIONS': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'FOOD#QUALITY': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'SERVICE#GENERAL': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'LOCATION#GENERAL': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'AMBIENCE#GENERAL': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'NO#ASPECT': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    vec_labels = []
    for label in labels:
        vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for category in label:
            vec = [x + y for x, y in zip(vec_dic[category], vec)]
        vec_labels.append(vec)
    vec_labels = np.array(vec_labels)

    # check for duplcate rows
    invalid_one_hot = [i for i, x in enumerate(vec_labels) for t in x if t > 1]
    if len(invalid_one_hot) > 0:
        print(text[i] for i in invalid_one_hot)
        raise ValueError('Invalid one hot value. Should be 0 or 1')

    cleaned_text = [s.split(" ") for s in cleaned_text]
    return [cleaned_text, vec_labels]

def clean_reviews(text, remove_stop_words=True):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z0-9\$]", " ", text)
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 3. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 4. Remove stop words
    if remove_stop_words:
        words = [w for w in words if not w in stops]


    # 5. Stemming the words
    # stemmer = LancasterStemmer()
    # words = [stemmer.stem(w) for w in words]

    # 6. Lemmatize the words
    words = nltk.pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w[0], get_wordnet_pos(w[1])) for w in words]

    # 7. Join the words back into one string separated by space,
    # and return the result.
    return (" ".join(words)).strip()


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


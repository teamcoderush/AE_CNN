import numpy as np
import re
import pandas as pd
import re
from nltk.corpus import stopwords

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


def load_data_and_labels(train_data_file, label_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train = list(open(train_data_file, "r").readlines())
    train = [s.strip() for s in train]

    labels = list(open(label_data_file, "r").readlines())
    labels = [s.strip() for s in labels]

    # Split by words
    train = [clean_str(sent) for sent in train]

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

    for l in labels:
        vec_labels.append(vec_dic.get(l))

    vec_labels = np.array(vec_labels)
    return [train, vec_labels]

def load_data_multilabel(data_path):
    """"Reads from csv and preprocess the dataset"""

    df = pd.read_csv(data_path, encoding='latin1')
    df.loc[(df['category'].isnull()), 'category'] = 'NO#ASPECT'
    df = df.drop_duplicates(['text','category'],keep = 'first')

    text = df.text.unique()
    text = sorted(text)

    labels = df.groupby('text')['category'].apply(list)
    index = labels.index.tolist()

    # check for mismatch in review order
    for i in range(len(text)):
        if text[i] != index[i]:
            print (text[i], index[i], i)
            raise ValueError('The order of reviews and labels is not correct')

    cleaned_text = [clean_str(sent) for sent in text]
    labels = labels.tolist()

    empty_reviews = [i for i, x in enumerate(cleaned_text) if x == ""]

    for i in empty_reviews:
        if labels[i] != "NO#ASPECT":
            cleaned_text[i] = clean_reviews(text[i],remove_stop_words=False)
        else:
            cleaned_text[i] = 'empty'

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

    #check for duplcate rows
    invalid_one_hot = [i for i, x in enumerate(vec_labels) for t in x if t > 1]
    if len(invalid_one_hot) > 0:
        print(text[i] for i in invalid_one_hot)
        raise ValueError('Invalid one hot value. Should be 0 or 1')

    return [cleaned_text, vec_labels]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def clean_reviews(text, remove_stop_words = True):
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
        meaningful_words = [w for w in words if not w in stops]
        return (" ".join(meaningful_words)).strip()
    # 5. Join the words back into one string separated by space,
    # and return the result.
    return (" ".join(words)).strip()

def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


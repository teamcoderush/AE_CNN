import numpy as np
import re
import itertools
from collections import Counter


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

    vec_labels =  np.array(vec_labels)
    return [train, vec_labels]


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

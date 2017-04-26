import numpy as np
import pandas as pd
import data_helpers as dh
import os
from cnn_architecture.features import synonym_vector as sv
from os.path import join, exists, split
from gensim.models import word2vec


def load_data_for_w2v(data_path):
    df = pd.read_csv(data_path, encoding='latin1')
    text = df.review_text.unique()
    cleaned_text = [dh.clean_reviews(sent, remove_stop_words=False) for sent in text]

    empty_reviews = [i for i, x in enumerate(cleaned_text) if x == ""]

    for i in empty_reviews:
        cleaned_text[i] = dh.clean_reviews(text[i], remove_stop_words=False)

    return cleaned_text


def get_w2v_model(data_path="../data/w2v/train1.csv", size=50, window=10, min_count=1, workers=2):
    model_dir = 'models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(size, min_count, window)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])

    else:
        text = load_data_for_w2v(data_path)

        lst = []
        for t in text:
            lst += [t.split()]

        model = word2vec.Word2Vec(lst, size=size, window=window, min_count=min_count, workers=workers)

        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        model.save(model_name)

    return model


def load_data_for_cnn(train_data_path, test_data_path, max_sequence_length=100, w2v_train_data=None, size=50, window=10,
                      min_count=1, ):
    size = size//4
    train_x, train_y = dh.load_data_multilabel(train_data_path)
    test_x, test_y = dh.load_data_multilabel(test_data_path)

    train_x_padded = dh.pad_sentences(train_x,
                                      max_length=max_sequence_length)  # required to emmit 0 as the pad key word
    test_x_padded = dh.pad_sentences(test_x, max_length=max_sequence_length)  # required to emmit 0 as the pad key word

    train_syn_arry = [sv.generate_sentence_matrix(w) for w in train_x_padded]
    test_syn_arry = [sv.generate_sentence_matrix(w) for w in test_x_padded]

    train_voc = [[text for texts in sentences for text in texts] for sentences in train_syn_arry]
    test_voc = [[text for texts in sentences for text in texts ] for sentences in test_syn_arry]

    vocabulary, vocabulary_inv = dh.build_vocab(train_voc + test_voc)

    model = get_w2v_model(size=size, window=window, min_count=min_count)

    embeddings = {w: model.wv[w] if w in model else np.random.uniform(-0.25, 0.25, model.vector_size)
                  for w in vocabulary_inv}

    train_wv = [[np.concatenate([embeddings.get(text) for text in texts]).tolist() for texts in sentence] for sentence in train_syn_arry]
    test_wv = [[np.concatenate([embeddings.get(text) for text in texts]).tolist() for texts in sentence] for sentence in test_syn_arry]

    train_wv = np.array(train_wv)
    test_wv = np.array(test_wv)
    return train_wv, train_y, test_wv, test_y


def load_data_for_eval(test_data_path, max_sequence_length=100, w2v_train_data=None, size=50, window=10, min_count=1, ):
    size = size // 4
    test_x, test_y = dh.load_data_multilabel(test_data_path)

    test_x_padded = dh.pad_sentences(test_x, max_length=max_sequence_length)  # required to emmit 0 as the pad key word
    test_syn_arry = [sv.generate_sentence_matrix(w) for w in test_x_padded]

    test_voc = [[text for texts in sentences for text in texts ] for sentences in test_syn_arry]

    vocabulary, vocabulary_inv = dh.build_vocab(test_voc)

    model = get_w2v_model(size=size, window=window, min_count=min_count)

    embeddings = {w: model.wv[w] if w in model else np.random.uniform(-0.25, 0.25, model.vector_size)
                  for w in vocabulary_inv}

    test_wv = [[np.concatenate([embeddings.get(text) for text in texts]).tolist() for texts in sentence] for sentence in test_syn_arry]

    test_wv = np.array(test_wv)
    return test_wv, test_y

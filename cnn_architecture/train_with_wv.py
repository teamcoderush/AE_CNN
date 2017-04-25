import logging
import os
import time

import numpy as np
from keras.layers import Dense, Flatten, Dropout, Input, Convolution1D, MaxPooling1D
from keras.layers.merge import Concatenate, Maximum, Add, Average
from keras.models import Model
from keras import optimizers

from cnn_architecture import eval, gensim_w2v

np.random.seed(2)


def initiate_training(train_data_path, test_data_path,
                      sequence_length=200,
                      embedding_dim=50,
                      filter_sizes=(3, 5),
                      num_filters=300,
                      dropout_prob=(0.5, 0.7),
                      hidden_dims=100,
                      batch_size=50,
                      num_epochs=15,
                      min_word_count=1,
                      context=10):

    # Data Preparation
    # ==================================================
    # Load data
    logging.debug("Loading data...")

    x_train, y_train, x_test, y_test = gensim_w2v.load_data_for_cnn(train_data_path, test_data_path,
                                                                    max_sequence_length=sequence_length,
                                                                    size=embedding_dim,
                                                                    window=context,
                                                                    min_count=min_word_count)
    logging.debug("x_train shape:", x_train.shape)

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_shuffled = x_train[shuffle_indices]
    y_shuffled = y_train[shuffle_indices]

    # Building model
    # ==================================================
    #
    # graph subnet with one input and one output,
    # convolutional layers concateneted in parallel
    input_shape = (sequence_length, embedding_dim,)
    model_input = Input(shape=input_shape, name="model-input")

    z = Dropout(dropout_prob[0], name="dropout-1")(model_input)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="same",
                             activation="tanh",
                             strides=1,
                             use_bias=True,
                             name="conv" + str(sz))(z)
        conv = MaxPooling1D(pool_size=sequence_length, padding="same", name="pool" + str(sz))(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)

    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    # z = LSTM(100, dropout=0.1, recurrent_dropout=0.2, go_backwards=False)(z)

    z = Dropout(dropout_prob[1], name="dropout-3")(z)
    z = Dense(hidden_dims, activation="relu", name="relu")(z)
    model_output = Dense(13, activation="softmax", name="softmax")(z)

    model = Model(model_input, model_output)

    # model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-2, momentum=.8, decay=0.0001), metrics=["categorical_accuracy"])
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-03, decay=0.001), metrics=["categorical_accuracy"])

    # Training model
    # ==================================================
    model.fit(x_shuffled, y_shuffled, batch_size=batch_size,
              epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logging.debug("Writing to {}\n".format(out_dir))
    model_loc = out_dir + "\\model.h5"
    model.save(model_loc)

    logging.info(
        "Train data path : {0}\nTest data path : {1}\n"
        "Sequence Length {2}\nEmbedding Dimensions : {3}\nFilter Sizes : {4}\nNo of Filters : {5}\nDropout Porbabilities : {6}\nHidden Dimensions :{7}\n"
        "Batch size : {8}\nNumber of epochs : {9}\nMin word count : {10}\ncontext :{11}\n"
            .format(train_data_path, test_data_path, sequence_length, embedding_dim, filter_sizes, num_filters,
                    dropout_prob, hidden_dims, batch_size, num_epochs, min_word_count, context))

    eval.eval_cnn(model_loc, test_data_path, sequence_length, embedding_dim, context, min_word_count,[timestamp, sequence_length, embedding_dim, filter_sizes, num_filters,
                    dropout_prob, hidden_dims, batch_size, num_epochs, min_word_count, context])

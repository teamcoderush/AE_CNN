from __future__ import print_function

import numpy as np
import os
import time
from keras.layers import Dense, Input
from keras.layers.merge import add
from keras.layers import LSTM
# from keras.layers import Dense, Dropout, Flatten,Convolution1D, MaxPooling1D
# from keras.layers.merge import Concatenate
from keras.models import Model

from w2v_features_rnn import gensim_w2v, eval

embedding_dim = 50
sequence_length = 80
batch_size = 64

context = 10
min_word_count = 1

train_data_path = "../data/train_data.csv"
test_data_path = "../data/test_data.csv"


print("Loading data...")

x_train, y_train, x_test, y_test = gensim_w2v.load_data_for_cnn(train_data_path, test_data_path,
                                                                    max_sequence_length=sequence_length,
                                                                    size=embedding_dim,
                                                                    window=context,
                                                                    min_count=min_word_count)
print("x_train shape:", x_train.shape)

# Shuffle data
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
x_shuffled = x_train[shuffle_indices]
y_shuffled = y_train[shuffle_indices]

# Building model
input_shape = (sequence_length, embedding_dim,)
model_input = Input(shape=input_shape, name="model-input")


fz = LSTM(100, dropout=0.5, recurrent_dropout=0.5)(model_input)

bz = LSTM(100, dropout=0.5, recurrent_dropout=0.5, go_backwards=True)(model_input)

merged = add([fz,bz])

model_output = Dense(13, activation="softmax", name="softmax")(merged)

model = Model(model_input, model_output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])

print('Train...')

model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test))

# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
print("Writing to {}\n".format(out_dir))
model_loc = out_dir + "\\model.h5"
model.save(model_loc)

eval.eval_rnn(model_loc, test_data_path, sequence_length, embedding_dim, context, min_word_count)

import numpy as np
import data_helpers
import os
import time
import gensim_w2v

from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence

np.random.seed(2)

# ---------------------- Parameters section -------------------
model_type = "CNN-rand"  # CNN-rand|CNN-non-static|CNN-static

# Model Hyperparameters
embedding_dim = 50
filter_sizes = [2,3, 5]
num_filters = 300
dropout_prob = (0.5, 0.6)
hidden_dims = 100

# Training parameters
batch_size = 64
num_epochs = 15
# val_split = 0.2
reg_val = 0.01

# Prepossessing parameters
sequence_length = 50 #200
# max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

# Data path
train_data_path = "data/train_data.csv"
test_data_path = "data/test_data.csv"

# Data Preparation
# ==================================================
# Load data
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
                         activation="relu",
                         strides=1,
                         use_bias=True,
                         name="conv" + str(sz))(z)
    conv = MaxPooling1D(pool_size=sequence_length, padding="same", name="pool" + str(sz))(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1], name="dropout-3")(z)
z = Dense(hidden_dims, activation="relu", name="relu")(z)
model_output = Dense(13, activation="softmax", name="softmax")(z)

model = Model(model_input, model_output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])

# Training model
# ==================================================
model.fit(x_shuffled, y_shuffled, batch_size=batch_size,
          epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
print("Writing to {}\n".format(out_dir))
model.save(out_dir + "\\model.h5")

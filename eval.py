import gensim_w2v
import keras

test_data_path = "data/test_data.csv"
model_path = "./models/1492680454/model.h5"

embedding_dim = 50
min_word_count = 1
sequence_length = 50
context = 10

test_x, test_y = gensim_w2v.load_data_for_eval(test_data_path,
                                               max_sequence_length=sequence_length,
                                               size=embedding_dim,
                                               window=context,
                                               min_count=min_word_count)

model = keras.models.load_model(model_path)
pred_y = model.predict(test_x)

t = [[i for i, j in zip(pred, y) if (j == 1 and i > 0.2)] for pred, y in zip(pred_y, test_y)]
tp = len([j for pred, y in zip(pred_y, test_y) for i, j in zip(pred, y) if (j == 1 and i > 0.15)])
fp = len([j for pred, y in zip(pred_y, test_y) for i, j in zip(pred, y) if (j == 0 and i > 0.15)])

precision = tp/(tp+fp)

print("true positives : {:d}, false positives : {:d}, precision :{:f}".format(tp, fp, precision)
)

print(test_x)
print(pred_y)

print( model.predict_proba(test_x,64))
# print(round(y), float(result))

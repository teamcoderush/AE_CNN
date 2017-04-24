import logging

import keras
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from rnn_architecture import gensim_w2v


def eval_rnn(model_path, test_data_path, sequence_length, embedding_dim, context, min_word_count):
    test_x, test_y = gensim_w2v.load_data_for_eval(test_data_path,
                                                   max_sequence_length=sequence_length,
                                                   size=embedding_dim,
                                                   window=context,
                                                   min_count=min_word_count)
    print("Start Evaluating the model : " + model_path)
    model = keras.models.load_model(model_path)
    pred_y = model.predict(test_x)

    threshold = 0.20

    tp_old = len([j for pred, y in zip(pred_y, test_y) for i, j in zip(pred, y) if (j == 1 and i > threshold)])
    tn_old = len([j for pred, y in zip(pred_y, test_y) for i, j in zip(pred, y) if (j == 0 and i <= threshold)])
    fp_old = len([j for pred, y in zip(pred_y, test_y) for i, j in zip(pred, y) if (j == 0 and i > threshold)])
    fn_old = len([j for pred, y in zip(pred_y, test_y) for i, j in zip(pred, y) if (j == 1 and i <= threshold)])

    precision = tp_old / (tp_old + fp_old)
    recall = tp_old / (tp_old + fn_old)
    f1 = (2 * precision * recall) / (precision + recall)

    total_cases = tn_old + tp_old + fn_old + fp_old

    accuracy = (tn_old + tp_old) / total_cases

    print("Total Cases : " + str(total_cases))
    print("true positives : {:d}, true negatives : {:d}, false positives : {:d}, false negatives : {:d}"
          .format(tp_old, tn_old, fp_old, fn_old))
    print("Accuracy : {:f}, Precision : {:f}, Recall : {:f}, F1-Score : {:f}"
          .format(accuracy, precision, recall, f1))

    print("\n\n-----------------\nWith max filter\n-----------------")

    pred_classes = [[i for i, p in enumerate(pred) if (p > threshold or max(pred) == p)] for pred in pred_y]

    tp = len([j for i, y in enumerate(test_y) for j, cls in enumerate(y) if (cls == 1 and j in pred_classes[i])])
    tn = len([j for i, y in enumerate(test_y) for j, cls in enumerate(y) if (cls == 0 and j not in pred_classes[i])])
    fp = len([j for i, y in enumerate(test_y) for j, cls in enumerate(y) if (cls == 0 and j in pred_classes[i])])
    fn = len([j for i, y in enumerate(test_y) for j, cls in enumerate(y) if (cls == 1 and j not in pred_classes[i])])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    total_cases = tn + tp + fn + fp

    accuracy = (tn + tp) / total_cases

    print("Total Cases : " + str(total_cases))
    print("true positives : {:d}, true negatives : {:d}, false positives : {:d}, false negatives : {:d}"
          .format(tp, tn, fp, fn))
    print("Accuracy : {:f}, Precision : {:f}, Recall : {:f}, F1-Score : {:f}"
          .format(accuracy, precision, recall, f1))

    print("\n\n-----------------\nWith SKLEARN\n----------------")

    pred_y = np.array([1 if (p > threshold or max(pred) == p) else 0 for pred in pred_y for p in pred])
    test_y = np.array([cls for y in test_y for cls in y])

    f1 = f1_score(test_y,pred_y)
    accuracy = accuracy_score(test_y,pred_y)
    precision = precision_score(test_y,pred_y)
    recall = recall_score(test_y,pred_y)

    print("Total Cases : " + str(len(test_y)))

    print("Accuracy : {:f}, Precision : {:f}, Recall : {:f}, F1-Score : {:f}"
          .format(accuracy, precision, recall, f1))

# test_data_path = "data/test_data.csv"
# model_path = "./models/1492679067/model.h5" #f1 59-60%
#
# embedding_dim = 50
# min_word_count = 1
# sequence_length = 50
# context = 10
#
# eval_cnn(model_path, test_data_path, sequence_length, embedding_dim, context, min_word_count)
#
#



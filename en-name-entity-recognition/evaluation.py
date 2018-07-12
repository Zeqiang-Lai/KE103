import numpy as np

from keras.models import load_model
from seqeval.metrics import f1_score
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.preprocessing import MultiLabelBinarizer

import utils

DATA_FLD = 'data'
MODEL_PATH = 'model/NER_Dropout.h5'

# Any Overlap OK
def F1_score_v1(y_test, y_pred, label2idx, idx2label):
    O = label2idx['O']
    TP, FP, FN = 0, 0, 0
    for a, b in zip(y_test, y_pred):
        for true, pred in zip(a, b):
            if true == pred and true != O:
                TP += 1
            elif pred == O and true != O:
                FN += 1
            elif true != pred:
                FP += 1

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# exact match
def F1_score_v2(y_test, y_pred, label2idx, idx2label):
    flatten_y_true = [tag for sent in y_test for tag in sent]
    true = [idx2label[id] for id in flatten_y_true]

    flatten_y_pred = [tag for sent in y_pred for tag in sent]
    pred = [idx2label[id] for id in flatten_y_pred]

    return f1_score(true, pred)


if __name__ == '__main__':
    # Load data
    train, valid, test, dic = utils.load_data(DATA_FLD)
    x_train, y_train = train
    x_valid, y_valid = valid
    x_test, y_test = test
    word2idx, idx2word, label2idx, idx2label = dic

    MAX_LEN = max([len(x) for x in x_train])
    NUM_LABEL = len(label2idx)

    # Load model
    model = load_model(MODEL_PATH)

    # Score
    X_valid = utils.process_data_for_keras(NUM_LABEL, x_valid)
    length = np.array([len(sent) for sent in x_valid], dtype='int32')
    y_pred = model.predict(X_valid)
    y = np.argmax(y_pred, -1)
    y_pred = [iy[:l] for iy, l in zip(y, length)]

    true = MultiLabelBinarizer().fit_transform(y_valid)
    pred = MultiLabelBinarizer().fit_transform(y_pred)
    score = sk_f1_score(true, pred, average='micro')
    print('F1(sk-learn): {0}'.format(score))

    a = np.array(y_valid).flatten()
    b = np.array(y_pred).flatten()
    f1_v1 = F1_score_v1(a, b, label2idx, idx2label)
    print('F1(Any Overlap OK): {0}'.format(f1_v1))

    f1_v2 = F1_score_v2(y_valid, y_pred, label2idx, idx2label)
    print('F1(exact match): {0}'.format(f1_v2))

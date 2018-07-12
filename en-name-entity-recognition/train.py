from keras import Input, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, np
from sklearn.preprocessing import MultiLabelBinarizer
from seqeval.metrics import f1_score
from sklearn.metrics import f1_score as sk_f1_score

import utils
from evaluation import F1_score_v1, F1_score_v2
from prep_emb import load_my_emb

DATA_FLD = 'aedata'
GLOVE_PATH = 'embedding/glove.6B.100d.txt'
MODEL_PATH = 'model/weight.h5'

# Hyperparameters
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 100
DROPOUT = 0.5
WORD_LSTM_SIZE = 100
FC_DIM = 100
OPTIMIZER = 'adam'
SHUFFLE = True
BATCH_SIZE = 64
EPOCHS = 10
VERBOSE = True

# Load data
train, valid, test, dic = utils.load_data(DATA_FLD)
x_train, y_train = train
x_valid, y_valid = valid
x_test, y_test = test
word2idx, idx2word, label2idx, idx2label = dic

MAX_LEN = max([len(x) for x in x_train])
NUM_LABEL = len(label2idx)

# Load embedding
embeddings = utils.load_embedding_matrix(GLOVE_PATH, word2idx, EMBEDDING_DIM, MAX_NUM_WORDS)
# embeddings = load_my_emb(word2idx, EMBEDDING_DIM, MAX_NUM_WORDS)
num_words = min(MAX_NUM_WORDS, len(word2idx) + 1)

# Construct network
word_ids = Input(batch_shape=(None, None), dtype='int32')
lengths = Input(batch_shape=(None, None), dtype='int32')
inputs = [word_ids, lengths]
# inputs = [word_ids]

embedding_layer = Embedding(input_dim=embeddings.shape[0],
                            output_dim=embeddings.shape[1],
                            mask_zero=True,
                            weights=[embeddings])(word_ids)
# embedding_layer = Dropout(DROPOUT)(embedding_layer)
z = Bidirectional(LSTM(units=WORD_LSTM_SIZE, return_sequences=True))(embedding_layer)
z = Dropout(DROPOUT)(z)
z = Dense(FC_DIM, activation='tanh')(z)
z = Dense(FC_DIM, activation='tanh')(z)
pred = Dense(NUM_LABEL, activation='softmax')(z)
model = Model(inputs=inputs, outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER)

# Train
train_steps, train_generator = utils.batch_iter(x_train, y_train,
                                                NUM_LABEL,
                                                BATCH_SIZE,
                                                shuffle=SHUFFLE)
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_steps,
                    epochs=EPOCHS,
                    verbose=VERBOSE)
# model.save(MODEL_PATH)
# model.save_weights(MODEL_PATH)

print('Validation set')
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

print('Test set')
# Score
X_test = utils.process_data_for_keras(NUM_LABEL, x_test)
length = np.array([len(sent) for sent in x_test], dtype='int32')
y_pred = model.predict(X_test)
y = np.argmax(y_pred, -1)
y_pred = [iy[:l] for iy, l in zip(y, length)]

true = MultiLabelBinarizer().fit_transform(y_test)
pred = MultiLabelBinarizer().fit_transform(y_pred)
score = sk_f1_score(true, pred, average='micro')
print('F1(sk-learn): {0}'.format(score))

a = np.array(y_test).flatten()
b = np.array(y_pred).flatten()
f1_v1 = F1_score_v1(a, b, label2idx, idx2label)
print('F1(Any Overlap OK): {0}'.format(f1_v1))

f1_v2 = F1_score_v2(y_test, y_pred, label2idx, idx2label)
print('F1(exact match): {0}'.format(f1_v2))
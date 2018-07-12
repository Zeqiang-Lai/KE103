import json

from keras import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, Dropout, Embedding, Concatenate, Bidirectional, LSTM
from keras_contrib.layers import CRF
from seqeval.metrics import f1_score


class BiLSTM_CNN(object):
    def __init__(self,
                 num_labels,
                 gen_vocab_size,
                 gen_embeddings=None,
                 domain_embeddings=None,
                 domain_vocab_size=None,
                 gen_embedding_dim=300,
                 domain_embedding_dim=100,
                 word_lstm_size=100,
                 domain_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 use_domain=True,
                 use_crf=False,
                 use_decnn=False,
                 use_bilstm=True):
        self.model = None
        self._gen_embedding_dim = gen_embedding_dim
        self._domain_embedding_dim = domain_embedding_dim
        self._num_labels = num_labels
        self._gen_vocab_size = gen_vocab_size
        self._gen_embeddings = gen_embeddings
        self._domain_embeddings = domain_embeddings
        self._domain_vocab_size = domain_vocab_size
        self._word_lstm_size = word_lstm_size
        self._domain_lstm_size = domain_lstm_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._use_domain = use_domain
        self._use_crf = use_crf
        self._use_decnn = use_decnn
        self._use_bilstm = use_bilstm
        self._loss = None

    def build(self):
        word_ids = Input(batch_shape=(None, None), dtype='int32')
        inputs = [word_ids]
        if self._gen_embeddings is None:
            word_embeddings = Embedding(input_dim=self._gen_vocab_size,
                                        output_dim=self._gen_embedding_dim,
                                        # mask_zero=True,
                                        trainable=False)(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._gen_embeddings.shape[0],
                                        output_dim=self._gen_embeddings.shape[1],
                                        # mask_zero=True,
                                        weights=[self._gen_embeddings],
                                        trainable=False)(word_ids)

        if self._use_domain:
            if self._domain_embeddings is None:
                domain_embeddings = Embedding(input_dim=self._domain_vocab_size,
                                              output_dim=self._domain_embedding_dim,
                                              # mask_zero=True,
                                              trainable=False)(word_ids)
            else:
                domain_embeddings = Embedding(input_dim=self._domain_embeddings.shape[0],
                                              output_dim=self._domain_embeddings.shape[1],
                                              # mask_zero=True,
                                              weights=[self._domain_embeddings],
                                              trainable=False)(word_ids)
            word_embeddings = Concatenate(axis=-1)([word_embeddings, domain_embeddings])

        word_embeddings = Dropout(self._dropout)(word_embeddings)

        if self._use_bilstm:
            z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
            z = Dropout(self._dropout)(z)

        if self._use_decnn:
            z1 = Conv1D(128, 5, padding='same', activation='relu')(word_embeddings)
            z2 = Conv1D(128, 3, padding='same', activation='relu')(word_embeddings)
            z = Concatenate(axis=-1)([z1, z2])
            z = Dropout(self._dropout)(z)
            z = Conv1D(256, 3, padding='same', activation='relu')(z)
            z = Dropout(self._dropout)(z)
            z = Conv1D(256, 3, padding='same', activation='relu')(z)
            z = Dropout(self._dropout)(z)
            z = Conv1D(256, 3, padding='same', activation='relu')(z)

        z = Dense(self._fc_dim, activation='tanh')(z)
        z = Dense(self._fc_dim, activation='tanh')(z)

        if self._use_crf:
            crf = CRF(self._num_labels, sparse_target=False)
            self._loss = crf.loss_function
            pred = crf(z)
        else:
            self._loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)

        self.model = Model(inputs=inputs, outputs=pred)

    def get_loss(self):
        return self._loss

    def __getattr__(self, name):
        return getattr(self.model, name)

    def save(self, weights_file, params_file):
        self.save_weights(weights_file)
        self.save_params(params_file)

    def save_weights(self, file_path):
        self.model.save_weights(file_path)

    def save_params(self, file_path):
        with open(file_path, 'w') as f:
            params = {name.lstrip('_'): val for name, val in vars(self).items()
                      if name not in {'_loss', 'model', '_domain_embeddings', '_gen_embeddings'}}
            json.dump(params, f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, weights_file, params_file):
        params = cls.load_params(params_file)
        self = cls(**params)
        self.build()
        self.load_weights(weights_file)

        return self

    @classmethod
    def load_params(cls, file_path):
        with open(file_path) as f:
            params = json.load(f)

        return params


def f1(y_test, y_pred, idx2label):
    flatten_y_true = [tag for sent in y_test for tag in sent]
    true = [idx2label[id] for id in flatten_y_true]

    flatten_y_pred = [tag for sent in y_pred for tag in sent]
    pred = [idx2label[id] for id in flatten_y_pred]

    return f1_score(true, pred)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'f1': f1,
}

import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

import utils

# Model hyperparameters
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 32  # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
SKIP_STEP = 5000

SUB_SAMPLE = False
LOSSTYPE = 'softmax'

VISUAL_FLD = 'visualization/softmax/with_subsample'
NUM_VISUALIZE = 3000  # number of tokens to visualize


class SkipGramModel:
    """ Build the graph for word2vec model """

    def __init__(self, dataset, vocab_size, embed_size, batch_size, num_sampled, learning_rate, sub_sample,
                 type_of_Loss):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.skip_step = SKIP_STEP
        self.dataset = dataset
        self.typeOfLoss = type_of_Loss
        self.log_path = type_of_Loss + '/with_subsample' if sub_sample else type_of_Loss + '/without_subsample'

    def _import_data(self):
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.center_words, self.target_words = self.iterator.get_next()

    def _create_embedding(self):
        with tf.name_scope('embed'):
            self.embed_matrix = tf.get_variable('embed_matrix',
                                                shape=[self.vocab_size, self.embed_size],
                                                initializer=tf.random_uniform_initializer())
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embedding')

    def _create_neg_loss(self):
        with tf.name_scope('loss'):
            neg_weight = tf.get_variable('neg_weight',
                                         shape=[self.vocab_size, self.embed_size],
                                         initializer=tf.truncated_normal_initializer(
                                             stddev=1.0 / (self.embed_size ** 0.5)))
            neg_bias = tf.get_variable('neg_bias', initializer=tf.zeros([VOCAB_SIZE]))

            sample_value = tf.nn.uniform_candidate_sampler(true_classes=tf.cast(self.target_words, tf.int64),
                                                           num_true=1,
                                                           num_sampled=self.num_sampled,
                                                           unique=True,
                                                           range_max=self.vocab_size,
                                                           name='uniform_sampler')

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=neg_weight,
                                                      biases=neg_bias,
                                                      labels=self.target_words,
                                                      inputs=self.embed,
                                                      sampled_values=sample_value,
                                                      num_sampled=self.num_sampled,
                                                      num_classes=self.vocab_size), name='loss')

    def _create_nce_loss(self):
        with tf.name_scope('loss'):
            nce_weight = tf.get_variable('nce_weight',
                                         shape=[self.vocab_size, self.embed_size],
                                         initializer=tf.truncated_normal_initializer(
                                             stddev=1.0 / (self.embed_size ** 0.5)))
            nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                      biases=nce_bias,
                                                      labels=self.target_words,
                                                      inputs=self.embed,
                                                      num_sampled=self.num_sampled,
                                                      num_classes=self.vocab_size), name='loss')

    def _create_sample_softmax_loss(self):
        with tf.name_scope('loss'):
            softmax_weight = tf.get_variable('softmax_weight',
                                             shape=[self.vocab_size, self.embed_size],
                                             initializer=tf.truncated_normal_initializer(
                                                 stddev=1.0 / (self.embed_size ** 0.5)))
            softmax_bias = tf.get_variable('softmax_bias', initializer=tf.zeros([VOCAB_SIZE]))

            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weight,
                                                                  biases=softmax_bias,
                                                                  labels=self.target_words,
                                                                  inputs=self.embed,
                                                                  num_sampled=self.num_sampled,
                                                                  num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                  global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._import_data()
        self._create_embedding()

        if self.typeOfLoss == 'nce':
            self._create_nce_loss()
        elif self.typeOfLoss == 'neg':
            self._create_neg_loss()
        else:
            self._create_sample_softmax_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, num_train_steps):
        saver = tf.train.Saver()  # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias

        utils.safe_mkdir('checkpoints/' + self.log_path)
        # utils.safe_mkdir('checkpoints/'+self.log_path)

        with tf.Session() as sess:
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/' + self.log_path + '/checkpoint'))
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/neg/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
            writer = tf.summary.FileWriter('graphs/' + self.log_path + '/lr' + str(self.lr), sess.graph)
            # writer = tf.summary.FileWriter('graphs/neg/lr' + str(self.lr), sess.graph)
            initial_step = self.global_step.eval()

            for index in range(initial_step, initial_step + num_train_steps):
                try:
                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op])
                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch
                    if (index + 1) % self.skip_step == 0:
                        print('Average loss at step {}: {:5.1f}'.format(index, total_loss / self.skip_step))
                        total_loss = 0.0
                        saver.save(sess, 'checkpoints/' + self.log_path + '/skip-gram', index)
                        # saver.save(sess, 'checkpoints/neg/skip-gram', index)
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)
            writer.close()

    def visualize(self, visual_fld, num_visualize):
        """ run "'tensorboard --logdir='visual_fld'" to see the embeddings """

        # create the list of num_variable most common words to visualize
        utils.most_common_words(visual_fld, num_visualize)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/' + self.log_path + '/checkpoint'))
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/neg/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            final_embed_matrix = sess.run(self.embed_matrix)

            # you have to store embeddings in a new variable
            embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embedding')
            sess.run(embedding_var.initializer)

            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter(visual_fld)

            # add embedding to the config file
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # link this tensor to its metadata file, in this case the first NUM_VISUALIZE words of vocab
            embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'

            # saves a configuration file that TensorBoard will read during startup.
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, os.path.join(visual_fld, 'model.ckpt'), 1)

    def save_embedding(self, emb_path):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/' + self.log_path + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            final_embed_matrix = sess.run(self.embed_matrix)
            print(type(final_embed_matrix))
            print(final_embed_matrix.shape)
            print(final_embed_matrix[1])
            if not os.path.exists(emb_path):
                os.makedirs(emb_path)

            emb_path = os.path.join(emb_path, 'embedding.npy')
            np.save(emb_path, final_embed_matrix)


def gen():
    yield from utils.batch_gen(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW, SUB_SAMPLE, VISUAL_FLD)


def save_dict(dict_path):
    local_dest = 'data/text8'
    words = utils.read_data(local_dest)
    dictionary, _ = utils.build_vocab(words, VOCAB_SIZE, SUB_SAMPLE, VISUAL_FLD)
    dict = json.dumps(dictionary)
    with open(dict_path,'w') as f:
        f.write(dict)

def main():
    dataset = tf.data.Dataset.from_generator(gen,
                                             (tf.int32, tf.int32),
                                             (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    model = SkipGramModel(dataset, VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE, SUB_SAMPLE, LOSSTYPE)
    model.build_graph()
    # model.train(NUM_TRAIN_STEPS)
    # model.visualize(VISUAL_FLD, NUM_VISUALIZE)

    # model.save_embedding('embedding')
    save_dict('embedding/dict.txt')

if __name__ == '__main__':
    main()

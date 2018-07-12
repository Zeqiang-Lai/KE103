"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from seqeval.metrics import f1_score


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params, emb_gen, emb_domain=None):
        """
        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net, self).__init__()

        # General and domain embedding
        self.embedding_gen = nn.Embedding(params.vocab_size, params.gen_embedding_dim)
        self.embedding_gen.weight = nn.Parameter(torch.from_numpy(emb_gen).float(), requires_grad=False)
        if emb_domain is not None:
            self.embedding_domain = nn.Embedding(params.vocab_size, params.domain_embedding_dim)
            self.embedding_domain.weight = nn.Parameter(torch.from_numpy(emb_domain).float(), requires_grad=False)
        else:
            self.embedding_domain = None
        # Convolution
        self.conv1 = torch.nn.Conv1d(params.gen_embedding_dim + params.domain_embedding_dim, 128, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(params.gen_embedding_dim + params.domain_embedding_dim, 128, 3, padding=1)
        self.conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)

        self.lstm = nn.LSTM(params.gen_embedding_dim + params.domain_embedding_dim, params.lstm_hidden_dim,
                            bidirectional=True, batch_first=True)

        # Dropout
        self.dropout = torch.nn.Dropout(params.dropout)

        # Liner
        # self.dense = torch.nn.Linear(200, 100)
        self.linear_ae = torch.nn.Linear(256, params.number_of_tags)

    def forward(self, x):
        """
        Args:
            x: (Variable) contains a batch of sentences, of dimension [batch_size, seq_len], where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADing tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.
            y:

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """

        # Concat two embeddings
        # general embedding: [batch_size, seq_len, gen_emb_dim]
        # domain embedding: [batch_size, seq_len, domain_emb_dim]
        # out:  [batch_size, seq_len, gen_emb_dim + domain_emb_dim]
        if self.embedding_domain is not None:
            x_emb = torch.cat((self.embedding_gen(x), self.embedding_domain(x)), dim=2)
        else:
            x_emb = self.embedding_gen(x)

        # s, _ = self.lstm(x_emb)
        # s = self.dropout(s)

        # # Transpose x_emb for convolution
        # # out:  [batch_size, gen_emb_dim + domain_emb_dim, seq_len]
        x_emb = self.dropout(x_emb).transpose(1, 2)
        #
        # # Convolution Period1
        # # out:  [batch_size, 256, seq_len]
        x_conv = torch.nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1))
        x_conv = self.dropout(x_conv)
        #
        # # Convolution Period2
        # # out:  [batch_size, 256, seq_len]
        x_conv = torch.nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv5(x_conv))

        # Transpose x_emb back
        # out:  [batch_size, seq_len, 256]
        s = x_conv.transpose(1, 2)

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        # reshape the Variable so that each row contains one token
        # out:  [batch_size * seq_len, 256]
        s = s.view(-1, s.shape[2])

        # Produce score for each label
        x_logit = self.linear_ae(s)

        return F.log_softmax(x_logit, dim=1)


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0).float()

    # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
    # number. This does not affect training, since we ignore the PADded tokens with the mask.
    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask).data[0])

    # # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
    return -torch.sum(outputs[range(outputs.shape[0]), labels] * mask) / num_tokens

    # labels = labels[mask].flatten()
    # outputs = outputs[mask].flatten()

    # return F.nll_loss(outputs, labels)


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels) / float(np.sum(mask))


def f1(outputs, labels):
    idx2tag = {}
    with open('data/semeval/restaurants/tags.txt') as f:
    # with open('data/ner/tags.txt') as f:
        for i, tag in enumerate(f.readlines()):
            tag = tag.strip().split()[0]
            idx2tag[i] = tag

    labels = labels.ravel()
    outputs = np.argmax(outputs, axis=1)
    mask = np.argwhere(labels >= 0)
    labels = labels[mask].flatten()
    outputs = outputs[mask].flatten()

    outputs = [idx2tag[tag] for tag in outputs]
    labels = [idx2tag[tag] for tag in labels]

    return f1_score(labels, outputs)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'f1': f1,
    # could add more metrics such as accuracy for each token type
}

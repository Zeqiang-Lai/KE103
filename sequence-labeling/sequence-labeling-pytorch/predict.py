"""Predict the tag"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.net as net
from model.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/semeval/laptop', help="Directory containing the dataset")
parser.add_argument('--emb_dir', default='embedding/semeval/laptop', help="Directory containing the embedding")
parser.add_argument('--model_dir', default='experiments/base_model_laptop', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def predict(model, loss_fn, data_iterator, metrics, params, num_steps):

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    prediction = []

    # compute metrics over the dataset
    for i in range(num_steps):
        # fetch the next evaluation batch
        data_batch, labels_batch = next(data_iterator)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        data = data_batch.data.cpu().numpy()

        idx2tag = {}
        with open('data/semeval/laptop/tags.txt') as f:
            # with open('data/ner/tags.txt') as f:
            for i, tag in enumerate(f.readlines()):
                tag = tag.strip().split()[0]
                idx2tag[i] = tag

        idx2word = {}
        with open('data/semeval/laptop/words.txt') as f:
            # with open('data/ner/tags.txt') as f:
            for i, word in enumerate(f.readlines()):
                word = word.strip().split()[0]
                idx2word[i] = word

        labels = labels_batch.ravel()
        data = data.ravel()
        outputs = np.argmax(output_batch, axis=1)
        mask = np.argwhere(labels >= 0)
        labels = labels[mask].flatten()
        outputs = outputs[mask].flatten()
        data = data[mask].flatten()

        labels = [idx2tag[tag] for tag in labels]
        outputs = [idx2tag[tag] for tag in outputs]
        data = [idx2word[word] for word in data]

        for j in range(len(data)):
            prediction.append(str(data[j]) + ' ' + labels[j] + ' ' + outputs[j] + '\n')

    return prediction


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.update('embedding/semeval/embedding_params.json')
    params.update(os.path.join(args.data_dir, 'dataset_params.json'))

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['test'], args.data_dir)
    test_data = data['test']

    # specify the test set size
    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.info("- done.")

    # Load embeddings
    gen_emb = np.load(os.path.join(args.emb_dir, 'gen.npy'))
    domain_emb = np.load(os.path.join(args.emb_dir, 'domain.npy'))

    # Define the model
    model = net.Net(params, gen_emb, domain_emb).cuda() if params.cuda else net.Net(params, gen_emb, domain_emb)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting Prediction")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    prediction = predict(model, loss_fn, test_data_iterator, metrics, params, num_steps)

    save_path = os.path.join(args.model_dir, "prediction_{}.text".format(args.restore_file))

    with open(save_path,'w') as f:
        f.writelines(prediction)

"""Train the model"""

import argparse
import logging
import os

import numpy as np

import utils
import model.net as net
import evaluate
from model.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/semeval/laptop', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model_laptop', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train_and_evaluate(model, train_data, val_data, test_data, metrics, params, model_dir, tag_map_path, restore_file=None):
    """Train the model and evaluate.

    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validaion data with keys 'data' and 'labels'
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        tag_map_path: (string) txt file containing the list of tags
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # TODO: reload weights from restore_file if specified

    train_steps, train_generator = data_loader.data_iterator(train_data, params, shuffle=True)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps,
                        epochs=params.num_epochs,
                        verbose=True)
    val_metrics = evaluate.evaluate_by_metrics(model, params, val_data, metrics, tag_map_path)
    train_metrics = evaluate.evaluate_by_metrics(model, params, train_data, metrics, tag_map_path)
    test_metrics = evaluate.evaluate_by_metrics(model, params, test_data, metrics, tag_map_path)

    all_metrics = {}
    for key, val in val_metrics.items():
        all_metrics['val-' + key] = val
    for key, val in train_metrics.items():
        all_metrics['train-' + key] = val
    for key, val in test_metrics.items():
        all_metrics['test-' + key] = val

    print(all_metrics)

    metrics_json_path = os.path.join(model_dir, "metrics.json")
    utils.save_dict_to_json(all_metrics, metrics_json_path)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # load parameters of embeddings
    params.update('embedding/semeval/embedding_params.json')
    params.update(os.path.join(args.data_dir, 'dataset_params.json'))

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['train', 'val', 'test'], args.data_dir)
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    # specify the train and val dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    # Load embeddings
    gen_emb = np.load('embedding/semeval/restaurants/gen.npy')
    domain_emb = np.load('embedding/semeval/restaurants/domain.npy')

    # Define the model
    model = net.BiLSTM_CNN(params.number_of_tags, params.vocab_size,
                           domain_vocab_size=params.vocab_size,
                           gen_embeddings=gen_emb,
                           domain_embeddings=domain_emb,
                           use_domain=True)
    model.build()
    model.compile(loss=model.get_loss(), optimizer='adam')

    # fetch metrics
    metrics = net.metrics

    tag_map_path = os.path.join(args.data_dir, 'tags.txt')
    # Train the model
    train_and_evaluate(model, train_data, val_data, test_data, metrics, params, args.model_dir, tag_map_path,
                       args.restore_file)

import numpy as np

import utils


def evaluate_by_metrics(model, params, data, metrics, tag_map_path):
    X_valid = utils.process_data_for_keras(params.number_of_tags, data['data'])
    length = np.array([len(sent) for sent in data['data']], dtype='int32')
    y_pred = model.predict(X_valid)
    y = np.argmax(y_pred, -1)
    y_pred = [iy[:l] for iy, l in zip(y, length)]

    idx2label = {}
    with open(tag_map_path) as f:
        for i, tag in enumerate(f.readlines()):
            tag = tag.strip().split()[0]
            idx2label[i] = tag

    val_metrics = {metric: metrics[metric](data['labels'], y_pred, idx2label)
                   for metric in metrics}
    return val_metrics

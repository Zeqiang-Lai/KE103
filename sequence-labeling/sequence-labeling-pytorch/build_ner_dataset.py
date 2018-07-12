import os

def load_dataset(path):
    with open(path) as f:
        lines = f.readlines()[2:]
        dataset = []
        words, tags = [], []
        for line in lines:
            w = line.split()
            if len(w) > 0:
                words.append(w[0])
                tags.append(w[-1])
            else:
                dataset.append((words, tags))
                words = []
                tags = []

        return dataset


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
            for words, tags in dataset:
                file_sentences.write("{}\n".format(" ".join(words)))
                file_labels.write("{}\n".format(" ".join(tags)))
    print("- done.")

def build_one_dataset(path_dataset, name_fld):
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("Loading ner dataset into memory...")
    dataset = load_dataset(path_dataset)
    print("- done.")

    # Save the datasets to files
    save_dataset(dataset, os.path.join('data/ner', name_fld))


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the
    # `train.text` and `valid.txt` and `test.text`)
    path_train_dataset = 'data/ner/train.txt'
    path_valid_dataset = 'data/ner/valid.txt'
    path_test_dataset = 'data/ner/test.txt'

    build_one_dataset(path_train_dataset, 'train')
    build_one_dataset(path_valid_dataset, 'val')
    build_one_dataset(path_test_dataset, 'test')

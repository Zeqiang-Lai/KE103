"""Read, split and save the semeval dataset for our model"""

import os
import xml
from string import punctuation
from xml.dom.minidom import parse
import xml.dom.minidom

def load_dataset(path_xml):
    """Loads dataset into memory from xml file"""

    def get_label(c, length):
        s = ""
        for _ in range(length):
            s += c
        return s

    DOMTree = xml.dom.minidom.parse(path_xml)
    collection = DOMTree.documentElement
    sents = collection.getElementsByTagName("sentence")
    LABEL = ['B', 'I']

    dataset = []
    for sent in sents:
        text = sent.getElementsByTagName('text')[0].childNodes[0].data  # type: str
        text = text.lower()
        tags_text = text

        aspects = sent.getElementsByTagName('aspectTerm')

        for aspect in aspects:
            term = aspect.getAttribute('term')  # type: str
            start = int(aspect.getAttribute('from'))
            end = int(aspect.getAttribute('to'))

            tokens = term.split()
            ttags = [get_label('B', len(tokens[i])) if i == 0 else get_label('I', len(tokens[i])) for i in
                     range(len(tokens))]
            tags_str = " ".join(ttags)

            tags_text = tags_text[:start] + tags_str + tags_text[end:]

        tags_text = "".join(c for c in tags_text if c not in punctuation.replace('\'', ''))
        tags = tags_text.split()
        tags = [tag[0] if tag[0] in LABEL else 'O' for tag in tags]
        text = "".join(c for c in text if c not in punctuation.replace('\'', ''))
        words = text.split()

        assert len(tags) == len(words)

        dataset.append((words, tags))

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
    print("Loading semeval dataset into memory...")
    dataset = load_dataset(path_dataset)
    print("- done.")

    # Split the dataset into train, val and split (dummy split with no shuffle)
    train_dataset = dataset[:int(0.7 * len(dataset))]
    val_dataset = dataset[int(0.7 * len(dataset)): int(0.85 * len(dataset))]
    test_dataset = dataset[int(0.85 * len(dataset)):]

    # Save the datasets to files
    save_dataset(train_dataset, os.path.join('data/semeval', name_fld, 'train'))
    save_dataset(val_dataset, os.path.join('data/semeval', name_fld, 'val'))
    save_dataset(test_dataset, os.path.join('data/semeval', name_fld, 'test'))


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the
    # `Laptop_Train_v2.xml` and `Restaurants_Train_v2.xml`)
    path_laptop_dataset = 'data/semeval/Laptop_Train_v2.xml'
    path_restaurants_dataset = 'data/semeval/Restaurants_Train_v2.xml'

    build_one_dataset(path_restaurants_dataset, 'restaurants')
    build_one_dataset(path_laptop_dataset, 'laptop')
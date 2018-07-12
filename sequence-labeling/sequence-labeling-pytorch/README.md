# Sequence Labeling with PyTorch

*Authors: Zeqiang Lai*

Note : all scripts must be run in `sequence-labeling-pytorch`.

## Requirements

We recommend using python3 and a virtual env. See instructions [here](https://cs230-stanford.github.io/project-starter-code.html).

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

deactivate the virtual environment with `deactivate`.

## Quickstart

### Download dataset

We use semeval 2014 Aspect Based Sentiment Analysis [(ABSA)](http://alt.qcri.org/semeval2014/task4/) dataset.

1. __Download the dataset__ `Laptop_Train_v2.xml` and `Restaurants_Train_v2.xml` on [Semeval](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) and save it under the `nlp/data/kaggle` directory. Make sure you download the simple version `ner_dataset.csv` and NOT the full version `ner.csv`.

2. __Build the dataset__ Run the following script
```
python build_semeval_dataset.py
python build_vocab.py
python build_vocab.py --data_dir data/semeval/laptop
```
It will extract the sentences and labels from the dataset, split it into train/val/test and save it in a convenient format for our model.

*Debug* If you get some errors, check that you downloaded the right file and saved it in the right directory.


### Download embedding

1. __Download the general embedding__

2. __Download the domain embedding__

3. __Process the embeddings__ Run the following script
```
python build_embedding.py
```

### Training

Train on restaurants dataset
```
python train.py --data_dir data/semeval/restaurants --model_dir experiments/base_model_res --emb_dir embedding/semeval/restaurants
```

Train on laptop dataset
```
python train.py --data_dir data/semeval/laptop --model_dir experiments/base_model_laptop --emb_dir embedding/semeval/laptop
```

### Evaluation
Evaluation on the test set
```
python evaluate.py --data_dir data/semeval/restaurants --model_dir experiments/base_model_res --emb_dir embedding/semeval/restaurants
```

## Prediction
```
python predict.py --data_dir data/semeval/restaurants --model_dir experiments/base_model_res --emb_dir embedding/semeval/restaurants
```

The prediction is store in a txt file `experiments/base_model_res/prediction_best.txt`

## TODO
- [x] prediction
- [ ] train on ner by command

## Reference

This project shares some starter code in [CS230](https://cs230-stanford.github.io/project-code-examples.html) and code snippets in [De-CNN](https://github.com/howardhsu/Double-Embeddings-and-CNN-based-Sequence-Labeling-for-Aspect-Extraction)

- [PyTorch documentation](http://pytorch.org/docs/0.3.0/)
- [Tutorials](http://pytorch.org/tutorials/)
- [PyTorch warm-up](https://github.com/jcjohnson/pytorch-examples)

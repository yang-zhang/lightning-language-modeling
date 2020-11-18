# lightning-language-modeling
Language Modeling Example with [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [🤗 Huggingface Transformers](https://huggingface.co/transformers/).

Language modeling fine-tuning adapts a pre-trained language model to a new domain and benefits downstream tasks such as classification. 
The script here applies to fine-tuning masked language modeling (MLM) models include ALBERT, BERT, DistilBERT and RoBERTa, on a text dataset. 
Details about the models can be found in Transformers [model summary](https://huggingface.co/transformers/model_summary.html).

The Transformers part of the code is adapted from [examples/language-modeling/run_mlm.py](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py). Finetuning causal language modeling (CLM) models can be done in a similar way, following [run_clm.py](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py). 

PyTorch Lightning is "The lightweight PyTorch wrapper for high-performance AI research.
Scale your models, not the boilerplate." [Quote from its doc](https://pytorch-lightning.readthedocs.io/en/latest/new-project.html):
> Organizing your code with PyTorch Lightning makes your code:
> - Keep all the flexibility (this is all pure PyTorch), but removes a ton of boilerplate
> - More readable by decoupling the research code from the engineering
> - Easier to reproduce
> - Less error prone by automating most of the training loop and tricky engineering
> - Scalable to any hardware without changing your model
 


## Setup environment
```bash
pip install -r requirements.txt
```

## Usage of this repo
To fine-tune a language model, run:
```bash
python language_model.py \ 
--model_name_or_path="The model checkpoint for weights initialization" \
--train_file="The input training data file (a text file)." \
--validation_file="The input validation data file (a text file)."
```

For example:
```bash
python language_model.py \
--model_name_or_path="distilbert-base-cased" \
--train_file="data/wikitext-2/wiki.train.small.raw" \
--validation_file="data/wikitext-2/wiki.valid.small.raw"
```

To run a “unit test” by running 1 training batch and 1 validation batch:
```bash
python language_model.py --fast_dev_run
```

See `language_model.py` and Transformers [scrip](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py) for more options.

To run with GPU:
```bash
python language_model.py --gpus=1
```


### Tensorboard:
To launch tensorboard:
```bash
tensorboard --logdir lightning_logs/
```

# lightning-language-modeling
Language Modeling Example with [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and 🤗 Huggingface [Transformers](https://huggingface.co/transformers/).

Language modeling fine-tuning adapts a pre-trained language model to the a new domain.
The script here applies to fine-tuning masked language modeling (MLM) models include ALBERT, BERT, DistilBERT and RoBERTa, on a text dataset. Details about the models can be found in Transformers [model summary](https://huggingface.co/transformers/model_summary.html).

The Transformers part of the code is adapted from [examples/language-modeling/run_mlm.py](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py). Finetuning causal language modeling (CLM) models can be done in a similar way, following [run_clm.py](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py). 
 


## Setup
```bash
make create_environment
conda activate pllm
make requirements
```

## Usage

To fine-tune a lanuage model, run:
```bash
python language_model.py 
```

To run a “unit test” by running 1 training batch and 1 validation batch:
```bash
python language_model.py --fast_dev_run
```

To run with GPU:
```bash
python language_model.py --gpus=1
```


### Tensorboard:
To launch tensorboard:
```bash
tensorboard --logdir lightning_logs/
```

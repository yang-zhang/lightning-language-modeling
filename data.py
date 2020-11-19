# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
import warnings
from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

class LMDataModule(pl.LightningDataModule):
    def __init__(self, model_name_or_path, train_file, validation_file, line_by_line, pad_to_max_length,
                 preprocessing_num_workers, overwrite_cache, max_seq_length, mlm_probability,
                 train_batch_size, val_batch_size, dataloader_num_workers):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.model_name_or_path = model_name_or_path
        self.line_by_line = line_by_line
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers

    def setup(self, stage):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        extension = self.train_file.split(".")[-1]
        if extension in ("txt", "raw"):
            extension = "text"

        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        datasets = load_dataset(extension, data_files=data_files)

        column_names = datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if self.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples["text"] = [line for line in examples["text"]
                                    if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples["text"],
                    padding=padding,
                    truncation=True,
                    max_length=self.max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not self.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.overwrite_cache,
            )

            if self.max_seq_length is None:
                self.max_seq_length = tokenizer.model_max_length
            else:
                if self.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.max_seq_length = min(self.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.max_seq_length) * self.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.max_seq_length]
                        for i in range(0, total_length, self.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=not self.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=self.mlm_probability)

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

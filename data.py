# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
import warnings
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def get_loaders(train_file,
                validation_file,
                model_name_or_path="distilbert-base-cased",
                line_by_line=False,
                pad_to_max_length=True,
                preprocessing_num_workers=1,
                overwrite_cache=True,
                max_seq_length=32,
                mlm_probability=0.15):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    extension = train_file.split(".")[-1]
    if extension in ("txt", "raw"):
        extension = "text"

    data_files = {}
    data_files["train"] = train_file
    data_files["validation"] = validation_file
    datasets = load_dataset(extension, data_files=data_files)

    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"]
                                if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not overwrite_cache,
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
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
        )

        if max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
        else:
            if max_seq_length > tokenizer.model_max_length:
                warnings.warn(
                    f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(max_seq_length, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)]
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
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=mlm_probability)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    return train_dataset, eval_dataset, data_collator

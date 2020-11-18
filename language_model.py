from data import get_loaders
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers.optimization import AdamW


class LMDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_file,
                 validation_file,
                 model_name_or_path="distilbert-base-cased",
                 line_by_line=False,
                 pad_to_max_length=True,
                 preprocessing_num_workers=1,
                 overwrite_cache=True,
                 max_seq_length=32,
                 mlm_probability=0.15,
                 train_batch_size=4,
                 dataloader_num_workers=4,
                 val_batch_size=8):
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
        self.dataloader_num_workers = dataloader_num_workers
        self.val_batch_size = val_batch_size

    def setup(self, stage):
        self.train_dataset, self.eval_dataset, self.data_collator = get_loaders(
            self.train_file,
            self.validation_file,
            self.model_name_or_path,
            self.line_by_line,
            self.pad_to_max_length,
            self.preprocessing_num_workers,
            self.overwrite_cache,
            self.max_seq_length,
            self.mlm_probability)

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


class LMModel(pl.LightningModule):
    def __init__(self,
                 model_name_or_path="distilbert-base-cased",
                 learning_rate=5e-5,
                 adam_beta1=0.9,
                 adam_beta2=0.999,
                 adam_epsilon=1e-8,
                 ):
        super().__init__()

        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(
            model_name_or_path, return_dict=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            config=config)

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        return {'loss': loss, 'log': {'val_loss': loss}}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.hparams.learning_rate,
                          betas=(self.hparams.adam_beta1,
                                 self.hparams.adam_beta2),
                          eps=self.hparams.adam_epsilon,)
        return optimizer


data_module = LMDataModule(train_file='data/wikitext-2/wiki.valid.raw',
                           validation_file='data/wikitext-2/wiki.test.raw')
lmmodel = LMModel()

trainer = pl.Trainer(fast_dev_run=True, gpus=1)
trainer.fit(lmmodel, data_module)

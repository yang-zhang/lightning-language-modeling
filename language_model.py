from argparse import ArgumentParser
import pytorch_lightning as pl
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers.optimization import AdamW
from data import LMDataModule


class LMModel(pl.LightningModule):
    def __init__(self, model_name_or_path, learning_rate, adam_beta1, adam_beta2, adam_epsilon):
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
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('valid_loss', loss, on_step=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.hparams.learning_rate,
                          betas=(self.hparams.adam_beta1,
                                 self.hparams.adam_beta2),
                          eps=self.hparams.adam_epsilon,)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=5e-5)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default="distilbert-base-cased")
    parser.add_argument('--train_file', type=str,
                        default="data/wikitext-2/wiki.train.small.raw")
    parser.add_argument('--validation_file', type=str,
                        default="data/wikitext-2/wiki.valid.small.raw")
    parser.add_argument('--line_by_line', action='store_true', default=False)
    parser.add_argument('--pad_to_max_length', action='store_true', default=False)
    parser.add_argument('--preprocessing_num_workers', type=int, default=4)
    parser.add_argument('--overwrite_cache', action='store_true', default=False)
    parser.add_argument('--max_seq_length', type=int, default=32)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    data_module = LMDataModule(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        validation_file=args.validation_file,
        line_by_line=args.line_by_line,
        pad_to_max_length=args.pad_to_max_length,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        max_seq_length=args.max_seq_length,
        mlm_probability=args.mlm_probability,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    # ------------
    # model
    # ------------
    lmmodel = LMModel(
        model_name_or_path=args.model_name_or_path,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(lmmodel, data_module)


if __name__ == '__main__':
    cli_main()

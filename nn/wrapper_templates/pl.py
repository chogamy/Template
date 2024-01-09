import os
import yaml

import torch
import lightning as L
from transformers import get_scheduler

# from pytorch_lightning import LightningModule


class Wrapper(L.LightningModule):
    def __init__(self, args, nn):
        super().__init__()
        self.nn = nn

        self.args = args

        self.save_hyperparameters()

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        path = os.path.join("args", "trainer", self.args.trainer)

        with open(f"./{path}.yaml") as f:
            trainer_args = yaml.load(f, Loader=yaml.FullLoader)

        if self.args.optimizers == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=float(trainer_args["lr"])
            )
        else:
            raise ValueError

        path = os.path.join("args", "trainer", "lrschedulers", self.args.lrscheduler)

        with open(f"./{path}.yaml") as f:
            scheduler_args = yaml.load(f, Loader=yaml.FullLoader)

        scheduler = get_scheduler(
            self.args.lrscheduler,
            optimizer,
            num_warmup_steps=scheduler_args["warmup"]
            * self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

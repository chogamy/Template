from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import get_scheduler

from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor
from src.openlib.models import BASE
from src.openlib.utils.metric import Metric


# 여기를 그냥...뭐....plm을 불러오면 안되나?
class FeatureExtractor(BASE):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.3,
        num_classes: int = None,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
        freeze=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.unseen_label_id = num_classes

        self.model = TransformerFeatureExtractor(
            model_name_or_path, dropout_prob=dropout_prob
        )
        self.classifier = nn.Linear(self.model.dense.out_features, num_classes)
        self.metric = Metric(num_classes, pre_train=True)

        # if freeze:
        #     self.freeze_model(self.model)

    def forward(self, batch):
        outputs = self.model(batch)
        logits = self.classifier(outputs)
        return logits

    def training_step(self, batch, batch_idx):
        self.train()
        model_input, labels, _ = self.step(batch, batch_idx)
        logits = self(model_input)

        loss = F.cross_entropy(logits, labels.long().squeeze(-1))
        self.log_dict({"loss": loss}, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        model_input, labels, _ = self.step(batch, batch_idx)
        logits = self(model_input)

        preds = logits.argmax(dim=-1)

        if labels[labels != -1].numel() > 0:
            # val_acc = self.metric.all_compute(preds[labels != -1], labels[labels != -1], pre_train=True, single=True)
            val_acc = self.metric.all_compute(
                preds[labels != -1], labels[labels != -1], pre_train=True
            )

            self.log_dict({"val_acc": val_acc["ACC"]})

    @torch.no_grad()
    def on_validation_epoch_end(self):
        self.eval()
        self.log_dict(
            self.metric.all_compute_end(pre_train=True, test=False), prog_bar=True
        )
        self.metric.all_reset()

        # self.log("val_acc", self.metric.compute(), prog_bar=True)
        # self.metric.reset()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()

        model_input, labels, _ = self.step(batch, batch_idx)

        logits = self(model_input)
        preds = logits.argmax(dim=-1)

        preds[preds == -1] = self.unseen_label_id
        labels[labels == -1] = self.unseen_label_id

        test_acc = self.metric.all_compute(
            preds, labels, pre_train=True, single=True, test=True
        )
        self.log("test_acc", test_acc["ACC"])

    @torch.no_grad()
    def on_test_epoch_end(self):
        self.eval()

        result = self.metric.all_compute_end(single=True, pre_train=True, test=True)
        self.log_dict(result, prog_bar=True)
        self.metric.all_reset(single=True)

        self.metric.end(self.trainer.checkpoint_callback.filename)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
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
        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.hparams.lr)

        warm_up_steps = int(self.trainer.estimated_stepping_batches * 0.1)
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            # num_warmup_steps=self.hparams.warmup_steps,
            num_warmup_steps=warm_up_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def freeze_model(self, model):
        for name, param in model.model.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

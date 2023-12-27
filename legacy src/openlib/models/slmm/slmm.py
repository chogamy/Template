from collections import defaultdict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from transformers import get_scheduler
import torchmetrics
from tqdm import tqdm

import numpy as np

from src.openlib.models import BASE
from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor, MixText


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


# class SLMM(pl.LightningModule):
class SLMM(BASE):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.3,
        num_classes: int = None,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
        freeze=True,
        n=1,
        l=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MixText(model_name_or_path, n=n, l=l)
        self.dense = nn.Linear(
            self.model.model.config.hidden_size, self.model.model.config.hidden_size
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.model.model.config.hidden_size, num_classes)
        self.classifier_2 = nn.Linear(self.model.model.config.hidden_size, 3)

        self.metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes + 1  # +1: unknown
        )
        # self.criterion = BoundaryLoss()

    def forward(self, batch0, batch1=None):
        all_hidden, pooler = self.model(batch0, batch1)

        outputs = torch.mean(all_hidden, 1)

        pooled_output = self.dense(outputs)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        logits_none = self.classifier_2(pooled_output).max(1).values.unsqueeze(1)
        prediction = torch.cat((logits, logits_none), 1)
        return logits, prediction

    def training_step(self, batch, batch_idx):
        print("training step")
        #
        l = np.random.beta(2, 2)
        #
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]
        #
        idx = torch.randperm(model_input["input_ids"].shape[0])  # 결국 batch size (코드 변경 가능)

        ###### generate pseudo data via manifold mixup
        batch1 = {k: v[idx] for k, v in batch.items() for v_i in v}
        model_input1 = {k: v for k, v in batch1.items() if k != "labels"}

        label_ids_1 = batch1["labels"]
        mask = label_ids_1 != labels

        label_ids_1[:] = self.hparams.num_classes

        ###### original data
        _, prediction = self.forward(batch0=model_input)

        labels_ = torch.zeros(prediction.shape).cuda()
        for i, j in enumerate(labels):
            labels_[i, j] = 1 - 0.2  # args.beta 바꿔야함
        labels_[:, -1] = 0.2  # args.beta
        loss = nn.KLDivLoss(reduction="batchmean")(prediction.softmax(dim=-1).log(), labels_)

        # #
        _, prediction1 = self.forward(batch0=model_input, batch1=model_input1)
        loss_2 = F.cross_entropy(prediction1[mask], label_ids_1[mask])

        # ##### calculate total loss
        total_loss = 0.1 * loss + (1 - 0.1) * loss_2  # 0.1 은 args.gamma 바꿔야함
        print("total_loss", total_loss)

        self.log_dict({"loss": total_loss}, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        print("validation_step")
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        _, preds = self.forward(model_input)

        labels = batch["labels"]
        preds = preds.argmax(dim=-1)
        # print("pred", preds)
        # print("labels", labels)

        val_acc = self.metric(preds, labels)
        self.log("val_acc", val_acc)
        return val_acc

    def on_validation_epoch_end(self):
        self.log("val_acc", self.metric.compute(), prog_bar=True)
        self.metric.reset()

    def test_step(self, batch, batch_nb):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        _, preds = self.forward(model_input)
        labels = batch["labels"]
        preds = preds.argmax(dim=-1)

        test_acc = self.metric(preds, labels)
        self.log("test_acc", test_acc)

    def on_test_epoch_end(self):
        self.log("test_acc", self.metric.compute(), prog_bar=True)
        self.metric.reset()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

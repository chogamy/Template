from collections import defaultdict
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from transformers import get_scheduler
import torchmetrics

from src.openlib.models import BASE
from src.openlib.models.default.FeatureExtractor import FeatureExtractor


# class ANS(pl.LightningModule):
class ANS(BASE):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
        freeze=True,
        sampler: str = "Negative",
        sampler_config: dict = None,
    ):
        sampler_config["num_classes"] = num_classes
        super().__init__(sampler, sampler_config)
        self.save_hyperparameters()

        self.num_classes = num_classes

        self.model = FeatureExtractor.load_from_checkpoint(model_name_or_path)

        self.model, self.classifier = self.model.model, self.model.classifier

        self.metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes + 1  # +1: unknown
        )

        hidden = self.model.dense.out_features
        self.classifiers = nn.ModuleList([nn.Linear(hidden, 1) for i in range(num_classes)])

    def forward(self, batch):
        pooled_output = self.model(batch)

        outputs = [classifier(pooled_output) for classifier in self.classifiers]

        return pooled_output, outputs

    def training_step(self, batch, batch_idx):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        pooled_output, outputs = self(model_input)
        device = pooled_output.device
        loss0 = torch.tensor(0.0, device=device)
        for i, output in enumerate(outputs):
            labels = torch.where(batch["labels"] == i, 1.0, 0.0)
            loss0 += F.binary_cross_entropy_with_logits(output.squeeze(1), labels)

        return loss0

        if self.sampler:
            pass

        zs = {i: pooled_output[batch["labels"] == i] for i in range(self.num_classes)}
        zs = self.negative_sampler(zs, self.classifiers)

        loss1 = torch.tensor(0.0, device=device)
        for i, classifier in enumerate(self.classifiers):
            logits = classifier(zs[i])
            logits = logits.squeeze(1)
            labels = torch.zeros(logits.size()).to(device)
            loss1 += F.binary_cross_entropy_with_logits(logits, labels)

        return loss0 + loss1

    def on_validation_start(self):
        super().on_validation_start()

    def validation_step(self, batch, batch_idx):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        pooled_output, outputs = self(model_input)

        device = pooled_output.device

        b, h = pooled_output.size()
        answers = torch.full((b,), -1).to(device)

        for output in outputs:
            ids = torch.where(output.squeeze(1) > 0)
            answers[ids] = 0

        known_ids = torch.where(answers == 0)
        logits = self.classifier(pooled_output[known_ids])
        act = nn.Softmax(dim=1)
        probs = act(logits)
        ans_ids = torch.argmax(probs, dim=1)

        answers[known_ids] = ans_ids

        val_acc = self.metric(answers, batch["labels"])
        self.log("val_acc", val_acc)

        return {"val_acc", val_acc}

    def on_validation_epoch_end(self) -> None:
        self.log("val_acc", self.metric.compute(), prog_bar=True)
        # self.metric.all_reset(single=True)

    def test_step(self, batch, batch_idx):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        pooled_output, outputs = self(model_input)

        device = pooled_output.device

        b, h = pooled_output.size()
        answers = torch.full((b,), -1).to(device)

        for output in outputs:
            ids = torch.where(output.squeeze(1) > 0)
            answers[ids] = 0

        known_ids = torch.where(answers == 0)[0]

        logits = self.classifier(pooled_output[known_ids])
        act = nn.Softmax(dim=1)
        probs = act(logits)
        ans_ids = torch.argmax(probs, dim=1)

        answers[known_ids] = ans_ids
        # from time import sleep

        # # print(answers)
        # # print(batch["labels"])
        # print(answers.shape, batch["labels"].shape)

        # sleep(2)

        test_acc = self.metric(answers, batch["labels"])
        self.log("test_acc", test_acc)

        return {"test_acc": test_acc}

    def on_test_epoch_end(self) -> None:
        self.log("test_acc", self.metric.compute(), prog_bar=True)
        # self.metric.all_reset(single=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        self.model.requires_grad_(False)
        self.classifier.requires_grad_(False)
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


# class NegativeSampler(nn.Module):
#     def __init__(self, num_classes, hidden, r, radius):
#         super().__init__()

#         self.cov = nn.Parameter(torch.zeros(num_classes, hidden, hidden), requires_grad=False)
#         self.r = r
#         self.radius = radius
#         self.num_classes = num_classes
#         self.k = 3
#         self.steps = 10
#         self.lr = 0.1  # 하이퍼 파라미터??????

#     def forward(self, zs, classifiers):
#         device = zs[0].device

#         # first, sample e
#         es = {i: None for i in range(self.num_classes)}
#         for i in range(self.num_classes):
#             num_samples, hidden_size = zs[i].size()
#             diag = (self.radius / 2) * torch.diag(self.cov[i])
#             e = torch.randn(num_samples, hidden_size).to(device) * torch.sqrt(diag)
#             es[i] = e.requires_grad_(True)

#         # update e by gradient ascent
#         for _ in range(self.steps):
#             """
#             validation에서는 gradient를 계산 못하는 듯 하다.
#             """
#             # classifiers.train()
#             for i, classifier in enumerate(classifiers):
#                 # classifier.train()

#                 logits = classifier(es[i]).squeeze(1)
#                 # print(es[i].requires_grad)
#                 # print(logits.requires_grad)
#                 # assert 1 == 0

#                 labels = torch.zeros(logits.size()).to(device)

#                 # for n, p in classifier.named_parameters():
#                 #     print(f"{n} {p.requires_grad}")

#                 # print(logits)
#                 # print(logits.shape)
#                 # print(logits.requires_grad)
#                 # print(labels)
#                 # print(labels.shape)
#                 # print(labels.requires_grad)
#                 # assert 1 == 0

#                 loss = F.binary_cross_entropy_with_logits(logits, labels)

#                 loss.backward()

#                 with torch.no_grad():
#                     es[i] += self.lr * es[i].grad
#                 es[i].grad.zero_()

#                 # classifier.eval()

#         # calculate alpha, then update e
#         for i in range(self.num_classes):
#             norm = torch.norm(es[i] - zs[i])
#             if self.radius <= norm <= self.r * self.radius:
#                 alpha = 1
#             elif self.r * self.radius <= norm:
#                 alpha = self.r * self.radius / norm
#             elif norm <= self.radius:
#                 alpha = self.radius / norm

#             norm_e = torch.norm(es[i])
#             alpha = torch.clamp(self.radius / norm_e, max=1)

#             es[i] = (alpha / norm_e) * es[i]

#         for i in range(self.num_classes):
#             zs[i] = zs[i] + es[i]

#         return zs

#     def update_cov(self, train_dataloader):
#         zs = {i: [] for i in range(self.num_classes)}
#         device = self.device
#         for batch in train_dataloader:
#             model_input = {k: v.to(device) for k, v in batch.items() if k != "labels"}

#             pooled_output, outputs = self(model_input)

#             for i in range(self.num_classes):
#                 zs[i].append(pooled_output[batch["labels"] == i])

#         for i in range(self.num_classes):
#             zs[i] = torch.cat(zs[i])
#             zs[i] = zs[i] - zs[i].mean(dim=0)
#             self.cov[i] = torch.mm(zs[i].t(), zs[i]) / zs[i].size(0)

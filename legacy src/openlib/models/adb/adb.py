from collections import defaultdict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from transformers import get_scheduler
from tqdm import tqdm

from src.openlib.models import BASE
from src.openlib.models.default.FeatureExtractor import FeatureExtractor
from src.openlib.utils.metric import Metric
from .boundaryloss import BoundaryLoss


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


class ADB(BASE):
    # class ADB(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
        freeze=True,
        tsne_path: str = None,
        tsne_config: dict = None,
        sampler: str = None,
        sampler_config: dict = None,
    ):
        if sampler:
            sampler_config["unseen_label_id"] = num_classes  # 이런게 부모에 있어도 될거같은데
        super().__init__(tsne_path, tsne_config, sampler, sampler_config)
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.unseen_label_id = num_classes

        # self.model = TransformerFeatureExtractor(model_name_or_path)
        # print("model_name_or_path: ", model_name_or_path)

        self.model = FeatureExtractor.load_from_checkpoint(model_name_or_path)

        self.model = self.model.model
        self.model.requires_grad_(False)

        feature_output_dim = self.model.dense.out_features
        self.centroids = nn.Parameter(torch.zeros(self.num_classes, feature_output_dim))
        self.delta = nn.Parameter(
            torch.abs(torch.randn(num_classes))
        )  # 강제로 delta 양수로 initialization

        self.metric = Metric(num_classes + 1)

        self.criterion = BoundaryLoss()

    def forward(self, batch):
        with torch.no_grad():
            pooled_output = self.model(batch)  # call forward in feature_extractor
        return pooled_output

    def on_fit_start(self) -> None:
        """Initialize the centroid."""

        device = self.device

        self.model.eval()
        with torch.no_grad():
            outputs = []
            for batch in tqdm(self.trainer.train_dataloader):
                model_input = {
                    k: v.to(device) for k, v in batch.items() if k != "labels"
                }
                pooled_output = self.model(model_input)

                labels = batch["labels"]

                for l, output in zip(labels, pooled_output):
                    # print("self.centroids[l]", self.centroids[l].device)
                    # print("output", output.device)
                    # output = output.to("cpu")
                    # print("output", output.device)
                    # assert 1==0

                    # self.centroids[l] += output.to("cpu")
                    self.centroids[l] += output

                outputs.append({"label": labels})

            # labels = torch.cat([x["label"] for x in outputs]).detach().to(device)
            labels = torch.cat([x["label"] for x in outputs])
            # print("labels", labels.device)
            # assert 1==0
            self.centroids /= torch.bincount(labels).float().unsqueeze(1).to(device)
        self.model.train()

    def training_step(self, batch, batch_idx):
        self.train()
        pooled_output, labels, _ = super().training_step(batch, batch_idx)

        loss = self.criterion(
            pooled_output[labels != self.unseen_label_id],
            self.centroids,
            self.delta,
            labels[labels != self.unseen_label_id],
        )

        self.log_dict({"loss": loss}, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        model_input, labels, _ = self.step(batch, batch_idx)

        pooled_output = self(model_input)
        preds = self.open_classify(pooled_output)

        preds[preds == -1] = self.unseen_label_id
        labels[labels == -1] = self.unseen_label_id

        val_acc = self.metric.all_compute(preds, labels, pre_train=False)

        self.log_dict({"val_acc": val_acc["ACC"]})

    @torch.no_grad()
    def on_validation_epoch_end(self):
        self.eval()
        self.log_dict(
            self.metric.all_compute_end(pre_train=False, test=False), prog_bar=True
        )
        self.metric.all_reset()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
        model_input, labels, _ = self.step(batch, batch_idx)
        pooled_output = self(model_input)
        preds = self.open_classify(pooled_output)

        preds[preds == -1] = self.unseen_label_id
        labels[labels == -1] = self.unseen_label_id

        test_acc = self.metric.all_compute(preds, labels, pre_train=False, test=True)
        self.log("test_acc", test_acc["ACC"])

        if self.tsne_path:
            # for visualization
            self.pooled_output = torch.cat(
                (self.pooled_output.to(self.device), pooled_output)
            )
            self.total_labels = torch.cat((self.total_labels.to(self.device), labels))
            self.total_y_pred = torch.cat((self.total_y_pred.to(self.device), preds))

    @torch.no_grad()
    def on_test_epoch_end(self):
        self.eval()
        result = self.metric.all_compute_end(pre_train=False, test=True)
        self.log_dict(result, prog_bar=True)
        self.metric.all_reset()

        # 결과 저장
        self.metric.end(self.trainer.checkpoint_callback.filename)

        # to parent!!!!!!!!!!!
        super().on_test_epoch_end()

    ####### delta만 optimize & warmup proportion
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([self.delta], lr=self.hparams.lr)

        warm_up_steps = int(self.trainer.estimated_stepping_batches * 0.1)
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def open_classify(self, features):
        device = self.device
        self.centroids = self.centroids.to(device)

        logits = euclidean_metric(features, self.centroids)
        _, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        boundary = F.softplus(self.delta)[preds]
        preds[euc_dis >= boundary] = self.unseen_label_id
        return preds

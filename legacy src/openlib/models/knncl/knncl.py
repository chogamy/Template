import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from transformers import get_scheduler
import torchmetrics
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

from src.openlib.models.knncl.base_knncl import Base_KNNCL
from src.openlib.models.knncl.knncl_util import generate_positive_sample, _prepare_inputs
from src.openlib.utils.metric import Metric

import os


class KNNCL(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.3,
        num_classes: int = None,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.unseen_label_id = num_classes  # should be fix

        self.model = Base_KNNCL(model_name_or_path=model_name_or_path, num_classes=num_classes)
        # self.metric = torchmetrics.Accuracy(
        #     task="multiclass", num_classes=num_classes + 1  # +1: unknown
        # )

        self.metric = Metric(num_classes + 1, single=True, pre_train=False)
        # self.train_total_features = torch.empty((0, self.model.encoder_q.config.hidden_size))
        # self.test_total_features = torch.empty((0, self.model.encoder_q.config.hidden_size))
        self.train_total_features = torch.empty((0, self.model.encoder_k.dense.out_features))
        self.test_total_features = torch.empty((0, self.model.encoder_k.dense.out_features))
        self.total_labels = torch.empty(0, dtype=torch.long)
        self.total_probs = torch.empty((0, num_classes))
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)

    def forward(self, batch, positive_sample=None, negative_sample=None):
        if positive_sample != None:
            output = self.model(batch, positive_sample, negative_sample)
            return output
        else:
            probs, seq_embed = self.model(batch)
            return probs, seq_embed

    def on_fit_start(self) -> None:
        """Initialize the centroid."""
        self.create_negative_dataset(
            data_loader=self.trainer.datamodule.train_dataloader()
        )  # type: ignore

    def training_step(self, batch, batch_idx):
        # print("training_step")
        # model_input = {k: v for k, v in batch.items() if k != "labels"}
        device = next(self.model.parameters()).device
        labels = batch["labels"]
        # print("negative", self.negative_dataset.keys())
        positive_sample = None
        positive_sample = generate_positive_sample(self.negative_dataset, labels)
        positive_sample = _prepare_inputs(device, positive_sample)
        # print("forward")

        loss = self.forward(batch, positive_sample=positive_sample)
        # print("outputs", loss)

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.log("loss", loss[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # print("\nvalidation\n")
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]

        probs, seq_embed = self.forward(model_input)
        preds = torch.argmax(probs, dim=1)

        preds[preds == -1] = self.unseen_label_id
        labels[labels == -1] = self.unseen_label_id

        val_acc = self.metric.all_compute(preds, labels, pre_train=False, single=True)

        self.log_dict({"val_acc": val_acc["ACC"]})

    def on_validation_epoch_end(self):
        self.log_dict(
            self.metric.all_compute_end(single=True, pre_train=False, test=True), prog_bar=True
        )
        self.metric.all_reset(single=True)

    # 첫번째 방법
    def on_test_start(self) -> None:    
        device = next(self.model.parameters()).device
        for batch in tqdm(self.trainer.datamodule.train_dataloader()):
        
            labels = batch["labels"]
            model_input = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.set_grad_enabled(False):
                output = self.model(model_input)
                # print("\nstart output", self.train_total_features.device)  # cpu
                self.train_total_features = torch.cat(
                    (self.train_total_features.to(device), output[1])
                )
                # print("\ntotal", self.train_total_features.device)  # cuda
        self.train_total_features = self.train_total_features.detach().cpu().numpy()
        
        
        
        self.lof.fit(self.train_total_features)
        
        return super().on_test_start()

    # 수정하려는 두번째 방법
    # def on_train_end(self):
    #     print("\none_test_start")  # 이부분이 학습 다 맞칠 때 진행해서 lof 저장해 두기
    #     device = next(self.model.parameters()).device
    #     self.lof = LocalOutlierFactor(n_neighbors=20, contamination = 0.05, novelty=True, n_jobs=-1)
    #     for batch in tqdm(self.trainer.datamodule.train_dataloader()):
    #         labels = batch['labels']
    #         model_input = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    #         with torch.set_grad_enabled(False):
    #             output = self.model(model_input)
    #             print("\nstart output", self.train_total_features.device)  # cpu
    #             self.train_total_features = torch.cat((self.train_total_features.to(device), output[1]))
    #             print("\ntotal", self.train_total_features.device)  # cuda
    #     self.train_total_features = self.train_total_features.detach().cpu().numpy()
    #     self.lof.fit(self.train_total_features)
    #     with open('lof_model.pkl', 'wb') as file:
    #         pickle.dump(self.lof, file)

    # def on_test_start(self) -> None:
    #     with open('lof_model.pkl', 'rb') as file:
    #         self.lof = pickle.load(file)
    #     return super().on_test_start()

    def test_step(self, batch, batch_nb):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]
        
        device = labels.device
        outputs = self.forward(model_input)

        total_prob, y_pred = outputs[0].max(dim=1)
        y_pred = y_pred.cpu().numpy()

        feats = outputs[1].cpu().numpy()

        # lof 로 예측하기
        y_pred_lof = pd.Series(self.lof.predict(feats))  # -1로 예측한 이상치 데이터의 인덱스 추출

        y_pred[y_pred_lof[y_pred_lof == -1].index] = self.unseen_label_id  # 특정 레이블로 할당함
        preds = torch.tensor(y_pred)

        preds[preds == -1] = self.unseen_label_id
        labels[labels == -1] = self.unseen_label_id
        
        preds = preds.to(device)
        
        test_acc = self.metric.all_compute(preds, labels, pre_train=False, single=True, test=True)
        
        self.log("test_acc", test_acc["ACC"])

    def on_test_epoch_end(self):
        result = self.metric.all_compute_end(single=True, pre_train=False, test=True)
        self.log_dict(result, prog_bar=True)
        self.metric.all_reset(single=True)

        # 결과 저장
        self.metric.end(self.trainer.checkpoint_callback.filename)

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

    def create_negative_dataset(self, data_loader: DataLoader) -> None:
        """create negative dataset."""
        self.negative_dataset = {}
        device = next(self.model.parameters()).device
        for batch in tqdm(data_loader):
            labels = batch["labels"]
            neg_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            for i, label in enumerate(labels):
                label = int(label)
                if label not in self.negative_dataset.keys():
                    self.negative_dataset[label] = [
                        {key: value[i] for key, value in neg_inputs.items()}
                    ]
                else:
                    self.negative_dataset[label].append(
                        {key: value[i] for key, value in neg_inputs.items()}
                    )

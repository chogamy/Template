import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import get_scheduler
import torchmetrics
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from src.openlib.utils.metric import Metric
from src.openlib.models import BASE
from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor


class K_1_way(BASE):
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
        tsne_path=None,
        tsne_config: dict = None,
        sampler: str = None,
        sampler_config: dict = None,
    ):
        if sampler:
            sampler_config["unseen_label_id"] = num_classes - 1
        super().__init__(tsne_path, tsne_config, sampler, sampler_config)
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.unseen_label_id = num_classes - 1

        self.model = TransformerFeatureExtractor(
            model_name_or_path, dropout_prob=dropout_prob
        )

        self.classifier = nn.Linear(self.model.dense.out_features, num_classes)
        self.total_labels = torch.empty(0, dtype=torch.long)
        self.total_preds = torch.empty((0, self.model.dense.out_features))

        self.metric = Metric(num_classes, pre_train=False)
        self.loss_fct = nn.CrossEntropyLoss()
        self.t = 0.1  # hyperpararmeter args.temp
        if tsne_path != None:
            self.tsne_path = tsne_path
        # if freeze:
        #     self.model = self.freeze_bert_parameters(self.model)
        self.total_y_pred = torch.empty(0, dtype=torch.long)

    def forward(self, batch):
        pooled_output = self.model(batch)
        logits = self.classifier(pooled_output)
        return pooled_output, logits

    def training_step(self, batch, batch_idx):
        self.train()
        # if self.sampler:
        #     pooled_output, labels = super().training_step(batch, batch_idx)
        #     logits = self.classifier(pooled_output)

        # else:
        #     model_inputs, labels = self.step(batch, batch_idx)
        #     pooled_output, logits = self(model_inputs)

        pooled_output, labels, _ = super().training_step(batch, batch_idx)
        logits = self.classifier(pooled_output)

        labels[labels == -1] = self.unseen_label_id

        loss = self.loss_fct(torch.div(logits, self.t), labels)

        self.log_dict({"loss": loss}, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        model_input, labels, _ = self.step(batch, batch_idx)
        pooled_output, logits = self(model_input)

        labels[labels == -1] = self.unseen_label_id

        probs = F.softmax(logits, dim=1)
        _, preds = probs.max(dim=1)

        val_acc = self.metric.all_compute(preds, labels, pre_train=False)

        self.log_dict({"val_acc": val_acc["ACC"]})

        return val_acc

    @torch.no_grad()
    def on_validation_epoch_end(self):
        self.log_dict(
            self.metric.all_compute_end(pre_train=False, test=True), prog_bar=True
        )
        self.metric.all_reset()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
        model_input, labels, _ = self.step(batch, batch_idx)

        labels[labels == -1] = self.unseen_label_id
        # device = next(self.model.parameters()).device
        pooled_output, logits = self(model_input)
        probs = F.softmax(logits, dim=1)
        _, preds = probs.max(dim=1)

        # test_acc = self.metric(preds, labels.to(device)).to(device)
        # test_acc = self.metric(preds, labels)
        # self.total_preds = torch.cat((self.total_preds.to(self.device), pooled_output))
        # self.total_labels = torch.cat((self.total_labels.to(self.device), labels))

        # self.total_y_pred = torch.cat((self.total_y_pred.to(self.device), preds))
        # self.log("test_acc", test_acc, prog_bar=True)
        # return {"test_acc": test_acc}

        test_acc = self.metric.all_compute(preds, labels, pre_train=False, test=True)

        self.log("test_acc", test_acc["ACC"])

    @torch.no_grad()
    def on_test_epoch_end(self):
        # # y_pred = self.total_preds.cpu().numpy()
        # # y_true = self.total_labels.cpu().numpy()
        # # self.draw_label(y_pred, y_true)

        # self.log("test_acc", self.metric.compute(), prog_bar=True)
        # self.metric.reset()
        # # print("self.total_y_pred[-1]", self.total_y_pred[-1].detach().cpu().numpy())

        result = self.metric.all_compute_end(pre_train=False, test=True)
        self.log_dict(result, prog_bar=True)
        self.metric.all_reset()

        # 결과 저장
        self.metric.end(self.trainer.checkpoint_callback.filename)

    def predict_step(self, batch, batch_nb):
        self.eval()
        model_input = {k: v.squeeze(0) for k, v in batch.items()}
        with torch.no_grad():
            _, logits = self.forward(model_input, pred=True)
            probs = F.softmax(logits, dim=1)
            _, total_preds = probs.max(dim=1)

        return total_preds

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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def freeze_bert_parameters(self, model):
        print("freeze_bert_para")
        for name, param in model.model.named_parameters():
            param.requires_grad = False
            # print('name', name)
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
        return model

    def draw_label(self, weights, labels):
        print("TSNE: fitting start...")
        tsne = TSNE(n_components=2, metric="cosine", random_state=0, n_jobs=4)
        # n_components: Dimension of the embedded space    # n_job: the number of CPU core to run parallel
        embedding = tsne.fit_transform(weights)

        df = pd.DataFrame(embedding, columns=["x", "y"])  # x, y 값을 DataFrame으로 변환
        df["label"] = labels  # 라벨을 DataFrame에 추가
        df.to_csv("hi.csv", index=False)

        # unique_labels = np.unique(labels)
        # colors = plt.cm.RdPu(np.linspace(0, 1, len(unique_labels)))

        # plt.figure(figsize=(10, 10))
        # for i, label in enumerate(unique_labels):
        #     mask = labels == label
        #     plt.scatter(embedding[mask, 0], embedding[mask, 1], color=colors[i], label=label)

        # plt.legend()
        # plt.show()
        # plt.savefig(self.tsne_path + '.pdf')

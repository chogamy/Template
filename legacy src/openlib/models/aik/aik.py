from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import get_scheduler
import torchmetrics

import numpy as np

from src.openlib.models import BASE
from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor


# class AIK(pl.LightningModule):
class AIK(BASE):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.3,
        num_classes: int = None,
        lr: float = 1e-3,
        plm_lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
        freeze=None,
        method: str = "aik",  # aik, bce
        l1: float = 1.0,
        l2: float = 1.0,
        l3: float = 1.0,
        margin: int = 300,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.method = method
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.margin = margin
        self.num_classes = num_classes

        self.model = TransformerFeatureExtractor(model_name_or_path, dropout_prob=dropout_prob)
        self.softmax = nn.Softmax(dim=1)
        self.n_ints_pred = nn.Linear(self.model.dense.out_features, 1)
        self.query = nn.Linear(self.model.dense.out_features, num_classes, bias=False)
        if self.method == "bce":
            self.mlp_logit = nn.Parameter(
                torch.rand(
                    num_classes, self.model.dense.out_features, requires_grad=True
                ).unsqueeze(0)
            )

        self.mu = nn.Parameter(
            torch.rand(num_classes, self.model.dense.out_features, requires_grad=True)
        )

        # if not self.args.use_mmc:
        #     self.mu = torch.rand(num_classes, self.model.dense.out_features)
        #     self.mu.requires_grad = True
        # else:  # Section 5.5
        #     assert self.args.dataset == "multiwoz23"
        #     self.mu = init_mmc_center(self.args, self.ind_set)

        self.intent_num_criterion = torch.nn.MSELoss()
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)

        self.mu_intent = torch.zeros(num_classes, self.model.dense.out_features)

        self.sigma = torch.zeros(
            num_classes, self.model.dense.out_features, self.model.dense.out_features
        )
        self.sigma_I = torch.zeros(
            num_classes, self.model.dense.out_features, self.model.dense.out_features
        )

        self.step_outputs_y_ood = []
        self.step_outputs_pred = []

        # auroc, fpr95, aupr_out, aupr_in
        self.roc = torchmetrics.ROC(task="binary")
        self.auc = torchmetrics.AUROC(task="binary")
        self.p_r_c = torchmetrics.PrecisionRecallCurve(task="binary")

    def forward(self, batch):
        outputs = self.model(batch, aik=True)

        n_ints = self.n_ints_pred(outputs.pooler_output)

        outputs = outputs.last_hidden_state

        weight = self.query(outputs)  # [batch, seq, intent]
        weight = torch.transpose(weight, 1, 2)  # [batch, intent, seq]
        weight = self.masked_softmax(weight, batch["attention_mask"])  # [batch, intent, seq]
        rep = torch.bmm(weight, outputs)  # [batch, intent, hidden]

        if self.method == "bce":
            logit = torch.sum(self.mlp_logit * rep, dim=2)
            return logit, n_ints
        else:
            return rep, n_ints

    def training_step(self, batch, batch_nb):
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        output, pred_intent_num = self(model_input)

        pos_loss = self.pos_loss(output, batch["labels"])
        neg_loss = self.neg_loss(output, batch["labels"])

        intent_num_loss = self.intent_num_criterion(
            pred_intent_num.squeeze(1), batch["n_ints_label"].float()
        )

        loss = pos_loss * self.l1 + neg_loss * self.l2 + intent_num_loss * self.l3

        self.log_dict({"loss": loss}, prog_bar=True)

        return loss

    # calculate sigma, sigma_I, mu_intent
    def on_train_epoch_end(self):
        device = self.device

        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

        train_rep = {i: [] for i in range(self.num_classes)}

        for batch in self.trainer.train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}

            output, pred_intent_num = self(model_input)

            y_ind, y_ood = batch["labels"].split([self.num_classes, 1], dim=1)

            [
                train_rep[i.item()].append(output[b.item()][i.item()].unsqueeze(0))
                for b, i in zip(*torch.where(y_ind == 1))
            ]

        self.sigma = self.sigma.fill_(0).to(device)
        self.sigma_I = self.sigma_I.fill_(0).to(device)
        self.mu_intent = self.mu_intent.fill_(0).to(device)

        # mu_intent = {}
        train_num = 0
        for i in range(self.num_classes):
            train_rep[i] = torch.cat(train_rep[i], dim=0)
            self.mu_intent[i] = torch.mean(train_rep[i], dim=0, dtype=torch.double)

            diff = train_rep[i] - self.mu_intent[i]

            index = 0  # avoid out of memory
            while index < diff.shape[0]:
                tmp_diff = torch.unsqueeze(diff[index : index + 200], axis=2)
                tmp_diff_T = torch.transpose(tmp_diff, 2, 1)
                self.sigma += torch.sum(tmp_diff * tmp_diff_T, axis=0)
                train_num += tmp_diff.shape[0]
                index += 200

        self.sigma = self.sigma / train_num
        self.sigma_I = torch.inverse(self.sigma)

        self.train()
        for param in self.parameters():
            param.requires_grad_(True)

    def validation_step(self, batch, batch_nb):
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}

        output, n_ints_pred = self(model_input)

        device = output.device

        n_ints_pred = torch.round(n_ints_pred.squeeze(1))

        intent_num_acc = self.metric(n_ints_pred, batch["n_ints_label"])

        diff = output - self.mu_intent[None, :, :].to(device)
        # b i h

        self.sigma_I = self.sigma_I.to(device)

        m = torch.einsum("bih,ihh,bih->bi", diff, self.sigma_I, diff)

        use_intent_num = True
        if use_intent_num:
            row_indices = torch.arange(m.size(0))
            score = m[row_indices, n_ints_pred.long()]    
        else:
            # score = m[:, 0].tolist()
            score = m[:, 0]

        _, y_ood = batch["labels"].split([self.num_classes, 1], dim=1)
        # y_ood = y_ood.squeeze(1).tolist()
        y_ood = y_ood.squeeze(1)

        self.step_outputs_y_ood.append(y_ood)
        self.step_outputs_pred.append(score)

        return intent_num_acc

    def on_validation_epoch_end(self):
        y_ood = torch.concat(self.step_outputs_y_ood, dim=0)
        pred = torch.concat(self.step_outputs_pred, dim=0)

        self.step_outputs_y_ood.clear()
        self.step_outputs_pred.clear()

        # fpr, tpr, _ = metrics.roc_curve(y_ood, score, pos_label=1)
        fpr, tpr, _ = self.roc(pred, y_ood.long())
        # auroc = metrics.auc(fpr, tpr)
        auroc = self.auc(fpr, tpr)

        # precision, recall, _ = metrics.precision_recall_curve(y_ood, score, pos_label=1)
        precision, recall, _ = self.p_r_c(pred, y_ood.long())
        # aupr_out = metrics.auc(recall, precision)
        aupr_out = self.auc(recall, precision)

        pos_pred = [pred[i] for i in range(len(y_ood)) if y_ood[i] == 1]
        pos_pred.sort()
        neg_pred = [pred[i] for i in range(len(y_ood)) if y_ood[i] == 0]
        if pos_pred:
            # threshold = pos_pred[int(len(pos_pred) * (1 - 0.95))]
            # fpr95 = sum(np.array(neg_pred) > threshold) / len(neg_pred)
            threshold = pos_pred[int(len(pos_pred) * (1 - 0.95))]
            len_neg_pred = len(neg_pred)
            neg_pred = [neg for neg in neg_pred if neg > threshold]
            fpr95 = sum(neg_pred).float() / len_neg_pred
        else:
            fpr95 = 0.0

        # pred = [-1 * one for one in pred]
        pred = -1 * pred
        # precision, recall, _ = metrics.precision_recall_curve(y_ood, score, pos_label=0)
        precision, recall, _ = self.p_r_c(pred, y_ood.long())
        # aupr_in = metrics.auc(recall, precision)
        aupr_in = self.auc(recall, precision)

        self.log_dict(
            {
                "val_acc": self.metric.compute(),
                "auroc": auroc,
                "aupr_out": aupr_out,
                "fpr95": fpr95,
                "aupr_in": aupr_in,
            },
            sync_dist=True,
        )

        self.metric.reset()

    def test_step(self, batch, batch_nb):
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}

        output, n_ints_pred = self(model_input)

        device = output.device

        n_ints_pred = torch.round(n_ints_pred.squeeze(1))

        intent_num_acc = self.metric(n_ints_pred, batch["n_ints_label"])

        diff = output - self.mu_intent[None, :, :].to(device)
        # b i h

        self.sigma_I = self.sigma_I.to(device)

        m = torch.einsum("bih,ihh,bih->bi", diff, self.sigma_I, diff)

        use_intent_num = True
        if use_intent_num:
            row_indices = torch.arange(m.size(0))
            score = m[row_indices, n_ints_pred.long()]
        else:
            # score = m[:, 0].tolist()
            score = m[:, 0]

        _, y_ood = batch["labels"].split([self.num_classes, 1], dim=1)
        # y_ood = y_ood.squeeze(1).tolist()
        y_ood = y_ood.squeeze(1)

        self.step_outputs_y_ood.append(y_ood)
        self.step_outputs_pred.append(score)

        return intent_num_acc

    def on_test_epoch_end(self):
        y_ood = torch.concat(self.step_outputs_y_ood, dim=0)
        pred = torch.concat(self.step_outputs_pred, dim=0)

        self.step_outputs_y_ood.clear()
        self.step_outputs_pred.clear()

        # fpr, tpr, _ = metrics.roc_curve(y_ood, score, pos_label=1)
        fpr, tpr, _ = self.roc(pred, y_ood.long())
        # auroc = metrics.auc(fpr, tpr)
        auroc = self.auc(fpr, tpr)

        # precision, recall, _ = metrics.precision_recall_curve(y_ood, score, pos_label=1)
        precision, recall, _ = self.p_r_c(pred, y_ood.long())
        # aupr_out = metrics.auc(recall, precision)
        aupr_out = self.auc(recall, precision)

        pos_pred = [pred[i] for i in range(len(y_ood)) if y_ood[i] == 1]
        pos_pred.sort()
        neg_pred = [pred[i] for i in range(len(y_ood)) if y_ood[i] == 0]
        if pos_pred:
            # threshold = pos_pred[int(len(pos_pred) * (1 - 0.95))]
            # fpr95 = sum(np.array(neg_pred) > threshold) / len(neg_pred)
            threshold = pos_pred[int(len(pos_pred) * (1 - 0.95))]
            len_neg_pred = len(neg_pred)
            neg_pred = [neg for neg in neg_pred if neg > threshold]
            fpr95 = sum(neg_pred).float() / len_neg_pred
        else:
            fpr95 = 0.0

        # pred = [-1 * one for one in pred]
        pred = -1 * pred
        # precision, recall, _ = metrics.precision_recall_curve(y_ood, score, pos_label=0)
        precision, recall, _ = self.p_r_c(pred, y_ood.long())
        # aupr_in = metrics.auc(recall, precision)
        aupr_in = self.auc(recall, precision)

        self.log_dict(
            {
                "acc": self.metric.compute(),
                "auroc": auroc,
                "aupr_out": aupr_out,
                "fpr95": fpr95,
                "aupr_in": aupr_in,
            },
            sync_dist=True,
        )

        self.metric.reset()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        self.sigma.requires_grad_(False)
        self.sigma_I.requires_grad_(False)

        plm_param_names = set(f"model.{n}" for n, p in self.model.named_parameters())

        plm_params = ((n, p) for n, p in self.named_parameters() if n in plm_param_names)
        rest_params = ((n, p) for n, p in self.named_parameters() if n not in plm_param_names)

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in plm_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.plm_lr,
            },
            {
                "params": [p for n, p in plm_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.plm_lr,
            },
            {
                "params": [p for n, p in rest_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.lr,
            },
            {
                "params": [p for n, p in rest_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.lr,
            },
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters)

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

    @staticmethod
    def masked_softmax(x, m=None, axis=-1):
        if len(m.size()) == 2:
            m = m.unsqueeze(1)
        if m is not None:
            m = m.float()
            x = x * m
        e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
        if m is not None:
            e_x = e_x * m
        softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
        return softmax

    def pos_loss(self, output, y):
        y_ind, y_ood = y.split([self.num_classes, 1], dim=1)

        b, h = output[torch.where(y_ind == 1)].shape

        return torch.pow(
            output[torch.where(y_ind == 1)] - self.mu[torch.where(y_ind == 1)[1]], 2
        ).sum() / (2 * b)

    def neg_loss(self, output, y):
        y_ind, y_ood = y.split([self.num_classes, 1], dim=1)

        b, h = output[torch.where(y_ind == 1)].shape

        return F.relu(
            self.margin
            - torch.pow(
                output[torch.where(y_ind == 0)] - self.mu[torch.where(y_ind == 0)[1]], 2
            ).sum(dim=1)
        ).sum() / (2 * b)

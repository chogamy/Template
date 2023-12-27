import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
)
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput


from src.openlib.models import BASE
from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor
from src.openlib.heads.head import Head
from src.openlib.samplers.pos_neg_sampler import PosNegSampler


class Base_KNNCL(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.1,
        num_classes: int = None,  # open 없는 개수
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
    ):
        super().__init__()

        # Use pretrained language model
        self.encoder_q = TransformerFeatureExtractor(model_name_or_path, dropout_prob=dropout_prob)
        self.encoder_k = TransformerFeatureExtractor(model_name_or_path, dropout_prob=dropout_prob)

        hidden_size = self.encoder_q.dense.out_features

        self.classifier_linear = Head(
            hidden_size,
            dropout_prob=dropout_prob,
            num_labels=num_classes,
            head_type="classification",
        )

        self.contrastive_liner_q = Head(
            hidden_size,
            dropout_prob=dropout_prob,
            num_labels=None,
            head_type="contrastive",
        )
        self.contrastive_liner_k = Head(
            hidden_size,
            dropout_prob=dropout_prob,
            num_labels=None,
            head_type="contrastive",
        )

        self.m = 0.999

        self.init_weights()  # Exec
        self.contrastive_rate_in_training = (
            0.1  # arg.contrastive_rate_in_training  -> hyperparameter
        )

        # create the label_queue and feature_queue
        self.num_classes = num_classes

        self.sampler = PosNegSampler(num_classes, hidden_size)

        self.update_num = 3  # args.positive_num -> hyperparamter
        self.device = next(self.encoder_q.parameters()).device

    def init_weights(self):
        for param_q, param_k in zip(
            self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()
        ):
            param_k.data = param_q.data

    def forward(self, batch, positive_sample=None, negative_sample=None):
        if "labels" in batch.keys():  ## train
            # print("label in batch")
            labels = batch["labels"]
            # print("labels", labels.shape)
            labels = labels.view(-1)

            with torch.no_grad():
                self.update_encoder_k()
                update_sample = self.reshape_dict(positive_sample)
                # bert_output_p = self.encoder_k(**update_sample)
                # update_keys = bert_output_p.last_hidden_state.mean(dim=1)
                update_keys = self.encoder_k(update_sample)
                # update_keys = bert_output_p[1]
                update_keys = self.contrastive_liner_k(update_keys)
                update_keys = self.l2norm(update_keys)
                tmp_labels = labels.unsqueeze(-1)
                tmp_labels = tmp_labels.repeat([1, self.update_num])
                tmp_labels = tmp_labels.view(-1)

                self.sampler._dequeue_and_enqueue(update_keys, tmp_labels)

            model_input = {k: v for k, v in batch.items() if k != "labels"}

            # bert_output_q = self.encoder_q(**model_input)
            # q = bert_output_q.last_hidden_state.mean(dim=1)
            q = self.encoder_q(model_input)

            liner_q = self.contrastive_liner_q(q)
            liner_q = self.l2norm(liner_q)
            logits_cls = self.classifier_linear(q)

            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.num_classes), labels)

            # logits_con = self.select_pos_neg_sample(liner_q, labels)
            logits_con = self.sampler(liner_q, labels)

            if logits_con is not None:
                labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
                loss_fct = CrossEntropyLoss()
                loss_con = loss_fct(logits_con, labels_con)

                loss = loss_con * self.contrastive_rate_in_training + loss_cls * (
                    1 - self.contrastive_rate_in_training
                )
            else:
                loss = loss_cls
            loss = SequenceClassifierOutput(loss)
            return loss

        else:  # valid, test # lof fit
            # seq_embed = self.encoder_q(**batch)
            # seq_embed = seq_embed.last_hidden_state.mean(dim=1)
            seq_embed = self.encoder_q(batch)
            # breakpoint()

            logits_cls = self.classifier_linear(seq_embed)
            probs = torch.softmax(logits_cls, dim=1)
            return probs, seq_embed

    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def l2norm(self, x: torch.Tensor):
        norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
        x = torch.div(x, norm)
        return x

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(
            self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

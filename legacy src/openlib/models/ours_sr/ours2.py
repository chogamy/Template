import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import get_scheduler
import torchmetrics

import numpy as np

from src.openlib.models import BASE
from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor
from src.openlib.utils.metric import Metric


class ScopeRecognizer(nn.Module):
    def __init__(self, hidden) -> None:
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        
        self.softmax = nn.Softmax(dim=-1)

        self.n = 5
        
        self.hidden = hidden
        
    def forward(self, q, k):
        '''
        어텐션
        '''
        b, l, e = q.shape
        qs = self.q(q)
        ks = self.k(k)
        # ble

        attn_logits = torch.bmm(qs, ks.transpose(1,2)) #/ np.sqrt(self.hidden)

        attn_mask = torch.full((l, l), float('-inf')).to(q.device)
        # attn_mask = torch.full((7, 7), float('-inf'))

        # for i in range(7):
        for i in range(l):
            attn_mask[i,i] = 0
            for j in range(1, self.n+1):
                end = i + j if i + j < l else l
                attn_mask[i, i : end] = 0
                attn_mask[i : end, i] = 0
        
        attn_mask = attn_mask.unsqueeze(0).expand(b, l, l)

        attn_logits += attn_mask

        attn_probs = self.softmax(attn_logits)

        return attn_probs

    
class Ours2(BASE):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.3,
        num_classes: int = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
        freeze=None,
        l = 0.3
        
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        # self.n_ints = 5 # 0 , 123, u
        self.n_ints = 4 # 0 , 123
        self.l = l
        

        self.model = TransformerFeatureExtractor(model_name_or_path, dropout_prob=dropout_prob)
        
        # self.classifier = nn.Linear(self.model.dense.out_features * 2, num_classes + 1)
        self.classifier = nn.Linear(self.model.dense.out_features, num_classes + 1)
        
        # self.emb_for_pre = nn.Linear(num_classes + 1, self.model.dense.out_features)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.sr = ScopeRecognizer(self.model.dense.out_features)    
        
        # self.layer_norm = nn.LayerNorm(self.model.dense.out_features)
    
        # self.n_ints_classifier = nn.Linear(self.model.dense.out_features * 2, self.n_ints)
        self.n_ints_classifier = nn.Linear(self.model.dense.out_features, self.n_ints)
            
        self.sigmoid = nn.Sigmoid()
    
        self.metric = Metric(self.num_classes + 1, num_labels=self.num_classes + 1)
        
    # new
    def forward(self, batch):
        outputs = self.model(batch, pooling=False)
        
        hs = outputs.last_hidden_state
        # last_hidden_state, pooler_output

        b,l,e = hs.shape

        cls = hs[:,0,:]

        attn_probs = self.sr(hs, hs)
        # b ll

        # b ll, b l e
        hs = torch.bmm(attn_probs, hs)
        # b l e

        # print(hs.shape)

        # print('avg: ',torch.mean(hs))
        # print('min', torch.min(hs))
        # print('max', torch.max(hs))

        i_logits = self.classifier(hs.mean(dim=1))

        # print(i_logits[0])

        
        n_ints_logits = self.n_ints_classifier(cls)

            
        return i_logits, n_ints_logits
    
    def training_step(self, batch, batch_idx):
        self.train()
        model_input, labels, n_labels = self.step(batch, batch_idx)
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        i_logits, n_logits = self(model_input)


        b, c= i_logits.shape
        
        # intent_loss = F.binary_cross_entropy_with_logits(Is_logits, batch['labels'].unsqueeze(1).expand(b,l,c))
        intent_loss = F.binary_cross_entropy_with_logits(i_logits, labels)
        
        n_ints_loss = F.cross_entropy(n_logits, n_labels)

        # loss = intent_loss + self.l * n_ints_loss
        loss = intent_loss + n_ints_loss
        # loss = n_ints_loss
        
        self.log_dict({"loss": loss}, prog_bar=True)
        
        return loss
        
    
    def prediction(self, i_logits, n_logits):
        n_probs = self.softmax(n_logits)
        n_ints = n_probs.argmax(dim=1)
        
        n_ints[n_ints==0] = 1 # 절대 0은 안나옴
        
        i_probs = self.sigmoid(i_logits)
    
        preds = torch.zeros(i_probs.shape).to(self.device)

        preds[i_probs > 0.5] = 1
        preds[:, -1] = 0 # unk는 0으로 둠
        

        # 개수 비교 해서 unk를 체크
        for b in range(preds.shape[0]):
            if sum(preds[b][:-1]) > n_ints[b]:
                preds[b][-1] = 1


        # # preds
        # for i, k in enumerate(n_ints):
        #     v, ids = i_probs[i].topk(k.item())
        #     preds[i][ids] = 1
            

        return preds, n_ints

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        model_input, labels, n_labels = self.step(batch, batch_idx)
        
        i_logits, n_logits = self(model_input)
        i_preds, n_preds = self.prediction(i_logits, n_logits)
        
        metric = self.metric.all_compute(i_preds, labels, n_ints_preds=n_preds, n_ints_labels=n_labels, pre_train=False)
            
        self.log_dict({"val_acc": metric['em']})
        
        
    @torch.no_grad()
    def on_validation_epoch_end(self):
        self.eval()
        self.log_dict(
            self.metric.all_compute_end(pre_train=False, test=True), prog_bar=True
        )
        # self.log_dict(
        #     self.n_ints_metric.all_compute_end(pre_train=False, test=False), prog_bar=True
        # )
        # self.n_ints_metric.all_reset()
        self.metric.all_reset()
    
    @torch.no_grad()    
    def test_step(self, batch, batch_idx):
        self.eval()
        model_input, labels, n_labels = self.step(batch, batch_idx)
        
        i_logits, n_logits = self(model_input)
        i_preds, n_preds = self.prediction(i_logits, n_logits)
        
        metric = self.metric.all_compute(i_preds, labels, n_ints_preds=n_preds, n_ints_labels=n_labels, pre_train=False)
        
        self.log_dict({"test_acc": metric["em"]})
       
    @torch.no_grad()
    def on_test_epoch_end(self):
        self.eval()
        self.log_dict(
            self.metric.all_compute_end(pre_train=False, test=True), prog_bar=True
        )
        # self.log_dict(
        #     self.n_ints_metric.all_compute_end(pre_train=False, test=False), prog_bar=True
        # )
        # self.n_ints_metric.all_reset()
        self.metric.all_reset()
        
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
        
        warm_up_steps = int(self.trainer.estimated_stepping_batches * 0.1)
        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.hparams.lr)
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            # num_warmup_steps=self.hparams.warmup_steps,
            num_warmup_steps=warm_up_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

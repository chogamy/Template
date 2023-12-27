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
        self.W1 = nn.Linear(hidden, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.hidden = hidden
        
    def forward(self, hs, Is):
        '''
        어텐션
        '''
        
        w1 = self.W1(hs)
        w2 = self.W2(hs+Is)
        
        attn = torch.bmm(w1, w2.transpose(1,2)) / np.sqrt(self.hidden)
        # w = self.softmax(attn)
        
        return attn
    
class FeedForward(nn.Module):
    def __init__(self, hidden, dropout_prob=0.3) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        
        self.relu = nn.ReLU()
        
        # self.dropout = nn.Dropout(dropout_prob)
        
        self.layer_norm = nn.LayerNorm(hidden)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        
        x = self.linear2(x)
        
        
        return self.layer_norm(x + x)
    
class Attention(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x,y,z):
        q = self.q(x)
        k = self.k(y)
        v = self.v(z)
        
        attn = torch.bmm(q, k.transpose(1,2)) / (self.hidden_size ** 0.5)
        
        attn_probs = self.softmax(attn)
        
        out = torch.bmm(attn_probs, v)
        
        return out
        
        
class Ours(BASE):
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
        
        self.classifier = nn.Linear(self.model.dense.out_features * 2, num_classes + 1)
        
        self.emb_for_pre = nn.Linear(num_classes + 1, self.model.dense.out_features)
        
        self.emb_for_pre_n = nn.Linear(self.n_ints, self.model.dense.out_features)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.sr = ScopeRecognizer(self.model.dense.out_features)    
        
        self.layer_norm = nn.LayerNorm(self.model.dense.out_features)
        
        self.R_attn = Attention(self.model.dense.out_features)
        self.NRI_attn = Attention(self.model.dense.out_features)
        self.IRN_attn = Attention(self.model.dense.out_features)
        
        # self.ffn = FeedForward(self.model.dense.out_features) # 몇인데?
    
        # self.decoder = bert 4 layers
        
        self.n_ints_classifier = nn.Linear(self.model.dense.out_features * 2, self.n_ints)
        # self.n_ints_classifier = nn.Linear(self.model.dense.out_features, self.n_ints)
        
        self.sigmoid = nn.Sigmoid()
    
        self.metric = Metric(self.num_classes + 1, multi=True)
        

    def forward(self, batch):
        outputs = self.model(batch, pooling=False)
        
        hs = outputs.last_hidden_state
        # last_hidden_state, pooler_output
        
        b,l,e = hs.shape
        
        Is = self.classifier(torch.concat([hs, hs.mean(dim=1).unsqueeze(1).expand(b,l,e)], dim=2))
        Is = self.softmax(Is) # BLC
        Is = self.emb_for_pre(Is) # BLE
        
        Ns = self.n_ints_classifier(torch.concat([hs, hs.mean(dim=1).unsqueeze(1).expand(b,l,e)], dim=2))
        Ns = self.softmax(Ns) # BLC
        Ns = self.emb_for_pre_n(Ns) # BLE
        
        ws = self.sr(Ns, Is)
        
        # 1. scope hidden
        hs_hat = hs + torch.einsum('bll,ble->ble', ws, hs) # BLE
                
        # 2. scope emb
        Is_hat = Is + torch.einsum('bll,ble->ble', ws, Is) # BLE
        
        # 3. n_ints emb
        Ns_hat = Ns + torch.einsum('bll,ble->ble', ws, Ns) # BLE
        
        # h_rs = self.layer_norm(hs_hat + Is_hat + Ns_hat)
        
        r = Is_hat + Ns_hat
        
        r_bar = self.R_attn(r,r,r)
        r_attn = self.layer_norm(r_bar)
        
        N_bar = self.NRI_attn(Ns_hat, r_attn, Is_hat)
        N_hat_prime = self.layer_norm(N_bar + Ns_hat)
        
        I_bar = self.IRN_attn(Is_hat, r_attn, Ns_hat)
        I_hat_prime = self.layer_norm(I_bar + Is_hat)
        
        # r_bar_hat = N_hat_prime + I_hat_prime
        
        h_rs = self.layer_norm(hs_hat + I_hat_prime + N_hat_prime)
        
        Is_logits = self.classifier(torch.concat([h_rs, h_rs.mean(dim=1).unsqueeze(1).expand(b,l,e)], dim=2))
        
        n_ints_logits = self.n_ints_classifier(torch.concat([h_rs, h_rs.mean(dim=1).unsqueeze(1).expand(b,l,e)], dim=2))   # 0.3 없애니까 안되는데
        # n_ints_logits = self.n_ints_classifier(hs.mean(dim=1))   
        # n_ints_logits = self.n_ints_classifier(hs[:,0,:])  # 모니터링을 n_ints_acc으로 할때만 잘 됐음  
                                                            # 0.3 없애니까 안되는데
        return Is_logits, n_ints_logits.mean(dim=1)
    
        
        
    def training_step(self, batch, batch_idx):
        self.train()
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        Is_logits, n_ints_logits = self(model_input)
        
        b,l,c= Is_logits.shape
        
        # intent_loss = F.binary_cross_entropy_with_logits(Is_logits, batch['labels'].unsqueeze(1).expand(b,l,c))
        intent_loss = F.binary_cross_entropy_with_logits(Is_logits.mean(dim=1), batch['labels'])
        
        n_ints_loss = F.cross_entropy(n_ints_logits, batch['n_ints_label'])

        # loss = intent_loss + self.l * n_ints_loss
        loss = intent_loss + n_ints_loss
        # loss = n_ints_loss
        
        self.log_dict({"loss": loss}, prog_bar=True)
        
        return loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        Is_logits, n_ints_logits = self(model_input)
        
        n_ints_probs = self.softmax(n_ints_logits)
        n_ints = n_ints_probs.argmax(dim=1)
        
        n_ints[n_ints==0] = 1 # 절대 0은 안나옴
        
        Is_probs = self.sigmoid(Is_logits)
        Is_probs = self.softmax(Is_probs)
        Is_probs = Is_probs.sum(dim=1) # BLC -> BC
        
        preds = torch.zeros(Is_probs.shape).to(self.device)
        for i, k in enumerate(n_ints):
            v, ids = Is_probs[i].topk(k.item())
            preds[ids] = 1
        
        metric = self.metric.all_compute(preds, batch['labels'], n_ints_preds=n_ints, n_ints_labels=batch['n_ints_label'], pre_train=False)
            
        self.log_dict({"val_acc": metric['ACC']})
        # self.log_dict({"val_acc": metric['n_ints_ACC']})
        # self.log_dict({"val_acc": n_ints_metric['ACC']})
        
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
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        Is_logits, n_ints_logits = self(model_input)
        
        n_ints_probs = self.softmax(n_ints_logits)
        n_ints = n_ints_probs.argmax(dim=1)
        
        n_ints[n_ints==0] = 1 # 절대 0은 안나옴
        
        Is_probs = self.sigmoid(Is_logits)
        Is_probs = self.softmax(Is_probs)
        Is_probs = Is_probs.sum(dim=1) # BLC -> BCZ
        
        preds = torch.zeros(Is_probs.shape).to(self.device)
        for i, k in enumerate(n_ints):
            v, ids = Is_probs[i].topk(k.item())
            preds[ids] = 1
        
        # n_ints_metric = self.n_ints_metric.all_compute(n_ints, batch['n_ints_label'])
        
        # metric = self.metric.all_compute(preds, batch['labels'], pre_train=False)
        metric = self.metric.all_compute(preds, batch['labels'], n_ints_preds=n_ints, n_ints_labels=batch['n_ints_label'], pre_train=False)
            
        # metric['n_ints_ACC'] = n_ints_metric['ACC']
        # metric['n_ints_F1'] = n_ints_metric['all_F1-score']
        
        self.log_dict({"test_acc": metric["ACC"]})
        # self.log_dict({"test_acc": n_ints_metric["ACC"]})
       
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

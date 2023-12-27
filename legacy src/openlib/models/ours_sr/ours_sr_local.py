'''
pooling해서 intent 만 예측
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler

# from local_attention import LocalAttention
from src.openlib.models.ours_sr.local_attention import LocalAttention



from src.openlib.models import BASE
from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor
from src.openlib.utils.metric import Metric

torch.set_float32_matmul_precision('high')

class Attention(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.d_model = d_model
        self.softmax = nn.Softmax(dim=-1)
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)
    
    def forward(self, q, k):

        q_ = self.w1(q) # ble
        k_ = self.w2(q + k) # ble
        
        attn = torch.matmul(q_, k_.transpose(-1, -2))
        
        attn_probs = self.softmax(attn)

        return attn_probs
                
class OurSRLocal(BASE):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.3,
        num_classes: int = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        # warmup_steps: int = 0,
        warmup_rate: float = 0.0,
        freeze=None,
        l = 0.3
        
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # # unk
        self.num_classes = num_classes + 1
        # all known
        # self.num_classes = num_classes
        
        self.n_ints = 4 # #0, 123
        self.l = l
        
        self.model = TransformerFeatureExtractor(model_name_or_path, dropout_prob=dropout_prob)
        
        self.classifier = nn.Linear(self.model.dense.out_features * 2, self.num_classes)
        self.n_classifier = nn.Linear(self.model.dense.out_features, self.n_ints)
        
        # self.attn = Attention(self.model.dense.out_features)

        self.local_attn = LocalAttention(dim=self.model.dense.out_features, window_size=10, causal=False, shared_qk=True)

        self.pre_emb = nn.Linear(self.num_classes, self.model.dense.out_features)

        self.layer_norm = nn.LayerNorm(self.model.dense.out_features)
        
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    
        self.metric = Metric(self.num_classes, num_labels=self.num_classes)
        
    def forward(self, batch, debug=False):        
        outputs = self.model(batch, pooling=False)
        
        h = outputs.last_hidden_state
        
        b, l, e = h.shape
        
        pool = h.mean(dim=1)
        
        i_logits = self.classifier(torch.concat([h, pool.unsqueeze(1).expand(b, l, e)], dim=-1))
        i_logits_probs = self.softmax(i_logits)

        pre_emb = self.pre_emb(i_logits_probs)

        exp_pre_emb = pre_emb.unsqueeze(1).expand(b, 1, l, e)
        exp_h = h.unsqueeze(1).expand(b, 1, l, e)
        
        h, attn_probs = self.local_attn(exp_pre_emb, exp_pre_emb, exp_h)
        h = h.squeeze(1)

        pool__ = h.mean(dim=1)

        i_logits = self.classifier(torch.concat([h, pool__.unsqueeze(1).expand(b, l, e)], dim=-1))
        n_logits = self.n_classifier(pool__)

        if debug == True:
            return i_logits, n_logits, attn_probs

        return i_logits, n_logits

        attn_probs = self.attn(h, pre_emb) # b l l         

        h_ = torch.bmm(h.transpose(1,2), attn_probs).transpose(1,2)

        emb = torch.bmm(pre_emb.transpose(1,2), attn_probs).transpose(1,2)

        # result = self.i_attn(emb, emb, emb)

        h__ = self.layer_norm(h_ + emb)

        pool__ = h__.mean(dim=1)

        i_logits = self.classifier(torch.concat([h__, pool__.unsqueeze(1).expand(b, l, e)], dim=-1))
        n_logits = self.n_classifier(pool__)

        if debug == True:
            return i_logits, n_logits, attn_probs

        return i_logits, n_logits
    
    def training_step(self, batch, batch_idx):
        self.train()
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        i_logits, n_logits = self(model_input)

        b, l, i = i_logits.shape
        
        intent_loss = F.binary_cross_entropy_with_logits(i_logits, batch['labels'].unsqueeze(1).expand(b, l, i))

        n_ints_loss = F.cross_entropy(n_logits, batch['n_ints_label'])
        
        loss = intent_loss + (n_ints_loss * self.l)
                
        self.log_dict({"loss": loss}, prog_bar=True)
        
        return loss

    # # AAAI 논문 버전 prediction        
    # def prediction(self, i_logits, n_logits):
    #     # b, l, i = i_logits.shape
    #     n_probs = self.softmax(n_logits)
    #     n_preds = n_probs.argmax(dim=-1)

    #     n_preds[n_preds==0] = 1

    #     i_probs = self.softmax(i_logits)
    #     i_probs = i_probs.sum(dim=1)

    #     i_preds = torch.zeros(i_probs.shape).to(self.device)

    #     for b in range(i_logits.shape[0]):
    #         i_pred = torch.topk(i_probs[b], k=n_preds[b])[1]
    #         i_preds[b][i_pred] = 1


    #     return i_preds, n_preds


    # # intent 개수 wise 하게!
    def prediction(self, i_logits, n_logits):
        # b, l, i = i_logits.shape
        n_probs = self.softmax(n_logits)
        n_preds = n_probs.argmax(dim=-1)

        n_preds[n_preds==0] = 1

        i_probs = self.sigmoid(i_logits.mean(dim=1))
        
        # known_probs, unknown_probs = torch.split(i_probs, [4, 1], dim=1)

        i_preds = torch.zeros(i_probs.shape).to(self.device)

        for b in range(i_logits.shape[0]):
            k = n_preds[b]
            cur_probs = i_probs[b]
            cur_probs[-1] = 0
            
            i_preds[b][cur_probs > 0.5] = 1

            if sum(i_preds[b]) < k:
                i_preds[b][-1] = 1


        return i_preds, n_preds
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        i_logits, n_logits = self(model_input)
        i_preds, n_preds = self.prediction(i_logits, n_logits)
        
        metric = self.metric.all_compute(i_preds, batch['labels'], n_ints_preds=n_preds, n_ints_labels=batch['n_ints_label'], pre_train=False)
            
        self.log_dict({"val_acc": metric['ACC']})
        
    @torch.no_grad()
    def on_validation_epoch_end(self):
        self.eval()
        self.log_dict(
            self.metric.all_compute_end(pre_train=False, test=True), prog_bar=True
        )
        self.metric.all_reset()
        
        # self.metric.end(self.trainer.checkpoint_callback.filename)
    
    @torch.no_grad()    
    def test_step(self, batch, batch_idx):
        self.eval()
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        i_logits, n_logits = self(model_input)
        i_preds, n_preds = self.prediction(i_logits, n_logits)
        
        metric = self.metric.all_compute(i_preds, batch['labels'], n_ints_preds=n_preds, n_ints_labels=batch['n_ints_label'], pre_train=False)
    
        self.log_dict({"test_acc": metric['ACC']})

       
    @torch.no_grad()
    def on_test_epoch_end(self):
        self.eval()
        self.log_dict(
            self.metric.all_compute_end(pre_train=False, test=True), prog_bar=True
        )
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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
        # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.hparams.lr)
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            # num_warmup_steps=self.hparams.warmup_steps,
            num_warmup_steps=warm_up_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

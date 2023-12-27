from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler

from src.openlib.models import BASE
from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor
from src.openlib.utils.metric import Metric

        
class Ours(BASE):
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
        
        self.num_classes = num_classes
        self.l = l
        
        ### 멀티 unk 세팅
        
        self.n_ints = 4 # 0 , 123

        self.model = TransformerFeatureExtractor(model_name_or_path, dropout_prob=dropout_prob)
        
        self.classifier = nn.Linear(self.model.dense.out_features, num_classes + 1)
        self.n_ints_classifier = nn.Linear(self.model.dense.out_features, self.n_ints)
        
    
        # self.intents = nn.Parameter(torch.randn(self.n_ints+1, self.model.dense.out_features), requires_grad=True)
        self.intents = nn.Parameter(torch.randn(num_classes + 1, self.model.dense.out_features), requires_grad=True)
        
        self.softmax = nn.Softmax(dim=-1)        
        self.sigmoid = nn.Sigmoid()
    
        self.metric = Metric(self.num_classes + 1, num_labels=self.n_ints)
        
    def forward(self, batch):
        outputs = self.model(batch, pooling=False)

        hs = outputs.last_hidden_state # b l e
        
        # only cls
        n_logits = self.n_ints_classifier(hs[:,0,:]) # b n
        
        return hs[:,1:,:], n_logits
    
    def intent_predict(self, hidden, intent_labels=None):
        # in training
        if intent_labels is not None: 
            intent = self.intents.expand(hidden.shape[0], -1, -1)
            # intent = self.intents[intent_labels]
        # in test
        else:
            intent = self.intents.expand(hidden.shape[0], -1, -1)
            
        attn = torch.bmm(hidden, intent.transpose(1,2)) # b l i
        attn = self.softmax(attn.transpose(1,2)) # b i l
        
        i_logits = self.classifier(hidden) # b l i
        
        i_logits = i_logits * attn.transpose(1,2) # b l i
        
        return i_logits, attn.transpose(1,2)
    
    def training_step(self, batch, batch_idx):
        self.train()
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        hidden, n_logits = self(model_input)
        
        n_ints_loss = F.cross_entropy(n_logits, batch['n_ints_label'])
        # ble , bn
        
        #self.intents[batch['labels']] # b i e
        i_logits, _ = self.intent_predict(hidden, batch['labels'])

        intent_loss = F.binary_cross_entropy_with_logits(i_logits.mean(dim=1), batch['labels'])
        
        loss = intent_loss + n_ints_loss
                
        self.log_dict({"loss": loss}, prog_bar=True)
        
        return loss
        
    def select_span(self, attn, n_pred):
        '''
        n_pred - 1 개의 ids를 고르는 문제
        '''
        l, c = attn.size()

        # span candidates
        spans = [comb for comb in combinations(range(1, l-1), n_pred) if sum(comb) == l]
        
        pred_span = l
        max_sum = float('-inf')
        
        for span in spans:
            attn_spans = torch.split(attn, span, dim=0)
            span_sum = 0
            for attn_span in attn_spans:
                span_sum += max(attn_span.sum(dim=0))
            if span_sum > max_sum:
                max_sum = span_sum
                pred_span = span
        
        return pred_span
        
    def prediction(self, hidden, n_logits):
        n_probs = self.softmax(n_logits)
        n_preds = torch.argmax(n_probs, dim=-1)
        n_preds[n_preds==0] = 1
        
        i_logits, attn = self.intent_predict(hidden)
        
        i_preds = torch.zeros(size=(i_logits.size(0), i_logits.size(2)), dtype=torch.long, device=attn.device)

        for b in range(attn.size(0)):

            span = self.select_span(attn[b], n_preds[b])
            
            i_logit_spans = torch.split(i_logits[b], span, dim=0)
            
            for i_logit_span in i_logit_spans:
                i_pred = torch.argmax(i_logit_span.mean(dim=0), dim=-1)
                i_preds[b][i_pred] = 1
            
        return i_preds, n_preds
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        hidden, n_logits = self(model_input)
        
        i_preds, n_preds = self.prediction(hidden, n_logits)
        
        metric = self.metric.all_compute(i_preds, batch['labels'], n_ints_preds=n_preds, n_ints_labels=batch['n_ints_label'], pre_train=False)
            
        self.log_dict({"val_acc": metric['ACC']})
        
    @torch.no_grad()
    def on_validation_epoch_end(self):
        self.eval()
        self.log_dict(
            self.metric.all_compute_end(pre_train=False, test=True), prog_bar=True
        )
        self.metric.all_reset()
        
        self.metric.end(self.trainer.checkpoint_callback.filename)
        
    @torch.no_grad()    
    def test_step(self, batch, batch_idx):
        self.eval()
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        hidden, n_logits = self(model_input)
        i_preds, n_preds = self.prediction(hidden, n_logits)
        
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
        warm_up_steps = int(self.trainer.estimated_stepping_batches * self.hparams.warmup_rate)
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

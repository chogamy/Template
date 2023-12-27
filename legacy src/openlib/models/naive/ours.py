from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler

from src.openlib.models import BASE
from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor
from src.openlib.utils.metric import Metric

        
class Naive(BASE):
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
        
        self.n_ints = 9 # 0 , 123

        self.model = TransformerFeatureExtractor(model_name_or_path, dropout_prob=dropout_prob)
        
        self.classifier = nn.Linear(self.model.dense.out_features, num_classes)
        self.n_ints_classifier = nn.Linear(self.model.dense.out_features, self.n_ints)
        
        
        self.softmax = nn.Softmax(dim=-1)        
        self.sigmoid = nn.Sigmoid()
    
        self.metric = Metric(self.num_classes, num_labels=self.n_ints)
        
    def forward(self, batch):
        outputs = self.model(batch, pooling=False)

        hs = outputs.last_hidden_state # b l e
        
        # only cls
        n_logits = self.n_ints_classifier(hs[:,0,:]) # b n

        i_logits = self.classifier(hs[:,1:,:].mean(dim=1)) # b c
        
        return i_logits, n_logits
    
    
    def training_step(self, batch, batch_idx):
        self.train()
        model_input = {k: v for k, v in batch.items() if k != "labels" and k != "n_ints_label"}
        i_logits, n_logits = self(model_input)
        
        n_ints_loss = F.cross_entropy(n_logits, batch['n_ints_label'])
        # ble , bn
        
        intent_loss = F.binary_cross_entropy_with_logits(i_logits, batch['labels'])
        
        loss = intent_loss + n_ints_loss
                
        self.log_dict({"loss": loss}, prog_bar=True)
        
        return loss
    
    def prediction(self, i_logits, n_logits):

        n_probs = self.softmax(n_logits)

        n_preds = torch.argmax(n_probs, dim=-1)
        n_preds[n_preds==0] = 1

        i_probs = self.sigmoid(i_logits)

        i_preds = torch.zeros(i_probs.shape, device=i_probs.device)

        for i in range(i_probs.shape[0]):
            k = n_preds[i]
            www = torch.topk(i_probs[i], k)
            i_preds[i][www.indices] = 1

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
        
        self.metric.end(self.trainer.checkpoint_callback.filename)
        
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

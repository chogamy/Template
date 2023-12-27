'''
어텐션 맵을 그려보고 싶어서 만듬
    length * intent로 만드는 어텐션 맵
컨벡스 샘ㅍ플러 ㅇㅇㅇ
or
멀티 컨벡스 샘플러
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler

from src.openlib.models import BASE
from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor
from src.openlib.utils.metric import Metric

torch.set_float32_matmul_precision('high')
          
class OurSR3(BASE):
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
        l = 0.3,
        sampler: str = None,
        sampler_config: dict = None,
    ):
        if sampler:
            sampler_config["unseen_label_id"] = num_classes
        super().__init__(sampler=sampler, sampler_config=sampler_config)
        self.save_hyperparameters()
        
        # # unk
        self.num_classes = num_classes + 1
        # all known
        # self.num_classes = num_classes
        
        self.n_ints = 4 # # 0, 123
        self.l = l
        
        self.model = TransformerFeatureExtractor(model_name_or_path, dropout_prob=dropout_prob)
        
        self.classifier = nn.Linear(self.model.dense.out_features, self.num_classes)
        self.n_classifier = nn.Linear(self.model.dense.out_features, self.n_ints)

        self.intent_embs = nn.Parameter(torch.randn(self.num_classes, self.model.dense.out_features), requires_grad=True)

        self.layer_norm = nn.LayerNorm(self.model.dense.out_features)
        
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    
        self.metric = Metric(self.num_classes, num_labels=self.num_classes)
        
    def forward(self, batch=None, debug=False, outputs=None):
        if outputs is not None:
            h = outputs
        else:
            outputs = self.model(batch, pooling=False)
            h = outputs.last_hidden_state
        
        b, l, e = h.shape

        intent_embs = self.intent_embs.unsqueeze(0).expand(b, -1, -1)
        # b, i, e

        attn = torch.bmm(h, intent_embs.transpose(-1, -2))
        # b, l, i

        attn_probs = F.softmax(attn, dim=1)
        # b, l, i

        pool = h.mean(dim=1)

        i_logits = self.classifier(pool)
        # b, i

        pool_attn_probs = attn_probs.mean(dim=1)

        i_logits = i_logits * pool_attn_probs

        n_logits = self.n_classifier(pool)

        if debug == True:
            return i_logits, n_logits, attn_probs

        return i_logits, n_logits

    
    def training_step(self, batch, batch_idx):
        self.train()
        pooled_output, labels, n_ints_labels = super().training_step(batch, batch_idx, pooling=False)

        i_logits, n_logits = self(outputs=pooled_output)

        intent_loss = F.binary_cross_entropy_with_logits(i_logits, labels)

        n_ints_loss = F.cross_entropy(n_logits, n_ints_labels)
        
        loss = intent_loss + (n_ints_loss * self.l)
                
        self.log_dict({"loss": loss}, prog_bar=True)
        
        return loss

    
    # # intent 개수 wise 하게!
    def prediction(self, i_logits, n_logits):
        # b, l, i = i_logits.shape
        n_probs = self.softmax(n_logits)
        n_preds = n_probs.argmax(dim=-1)

        n_preds[n_preds==0] = 1

        i_probs = self.sigmoid(i_logits)
        
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
        model_input, labels, n_ints_labels = self.step(batch, batch_idx)
        i_logits, n_logits = self(model_input)
        i_preds, n_preds = self.prediction(i_logits, n_logits)
        
        metric = self.metric.all_compute(i_preds, labels, n_ints_preds=n_preds, n_ints_labels=n_ints_labels, pre_train=False)

        self.log_dict({"val_acc": metric['em']})
        
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
        model_input, labels, n_ints_labels = self.step(batch, batch_idx)
        i_logits, n_logits = self(model_input)
        i_preds, n_preds = self.prediction(i_logits, n_logits)
        
        metric = self.metric.all_compute(i_preds, labels, n_ints_preds=n_preds, n_ints_labels=n_ints_labels, pre_train=False)
    
        self.log_dict({"test_acc": metric['em']})

       
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

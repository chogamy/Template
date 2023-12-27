import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.1, num_labels=None, head_type="classification"):
        super(Head, self).__init__()
        assert head_type in ["classification", "contrastive"]
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        if head_type == "classification":
            self.out_proj = nn.Linear(hidden_size, num_labels)
        elif head_type == "contrastive":
            self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

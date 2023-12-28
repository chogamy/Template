import torch.nn as nn
from transformers import AutoConfig, AutoModel

from nn.nn_templates.encoder import Encoder


class E1C1(nn.Module):
    def __init__(self, args, data_config) -> None:
        super().__init__()

        if args.enc == "custom":
            self.encoder = Encoder(args)
        else:
            config = AutoConfig.from_pretrained(args.enc)
            self.encoder = AutoModel.from_pretrained(args.enc)

        self.label_to_id = data_config["label_to_id"]
        self.id_to_label = data_config["id_to_label"]

        self.classifier = nn.Linear(config.hidden_size, len(self.label_to_id))

        self.args = args

    def forward(self, batch):
        hidden = self.encoder(batch["input_ids"])
        pass

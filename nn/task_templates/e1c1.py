import torch.nn as nn
from transformers import AutoConfig, AutoModel


class E1C1(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        try:
            self.encoder = AutoModel.from_pretrained(args.enc)
            config = AutoConfig.from_pretrained(args.enc)
        except:
            from nn_templates.encoder import Encoder

            self.encoder = Encoder(args)
            config = 

        print(config)

        assert 1 == 0

        self.classifier = nn.Linear()

        self.args = args

    def forward(self):
        pass

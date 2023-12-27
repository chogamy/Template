import torch.nn as nn
from transformers import AutoConfig, AutoModel


class E1C1(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        try:
            self.encoder = AutoModel.from_pretrained(args.enc)
            config = AutoConfig.from_pretrained(args.enc)

            print(config)
        except:
            from nn.nn_templates.encoder import Encoder

            self.encoder = Encoder(args)

            # config:뭐.... 데이터 클래스 개수 이런거....

        assert 1 == 0

        self.classifier = nn.Linear()

        self.args = args

    def forward(self):
        pass

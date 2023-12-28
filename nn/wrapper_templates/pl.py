import pytorch_lightning as pl


class Wrapper(pl.LightningModule):
    def __init__(self, args, nn):
        super().__init__()
        self.nn = nn

        self.args = args

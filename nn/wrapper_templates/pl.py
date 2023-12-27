import pytorch_lightning as pl


class Wrapper(pl.LightningModule):
    def __init__(self, args, nn):
        super().__init__(args, nn)
        self.nn = nn

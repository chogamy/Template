import lightning as L

# from pytorch_lightning import LightningModule


class Wrapper(L.LightningModule):
    def __init__(self, args, nn):
        super().__init__()
        self.nn = nn

        self.args = args

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        """
        Optimizers and LR Schedulers
        """
        assert 1 == 0
        pass

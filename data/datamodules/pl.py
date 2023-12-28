import lightning as L
from torch.utils.data import DataLoader


class DataModule(L.LightningDataModule):
    def __init__(self, dataset, args) -> None:
        super().__init__()

        self.dataset = dataset
        self.args = args

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        return super().setup(stage)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"])

    def val_dataloader(self):
        return DataLoader(self.dataset["val"])

    def test_dataloader(self):
        return DataLoader(self.dataset["test"])

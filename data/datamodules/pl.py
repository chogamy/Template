import lightning as L
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(L.LightningDataModule):
    def __init__(self, args, dataset, data_config) -> None:
        super().__init__()

        self.dataset = dataset
        self.args = args
        self.data_config = data_config

    def prepare_data(self) -> None:
        assert 1 == 0

    def setup(self, stage: str) -> None:
        self.prepare_data()
        if stage == "fit":
            print("fit")

        elif stage == "test":
            print("test")

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["val"],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

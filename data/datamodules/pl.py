import lightning as L
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transformers import default_data_collator


class DataModule(L.LightningDataModule):
    def __init__(self, args, dataset, data_config) -> None:
        super().__init__()

        self.dataset = dataset
        self.args = args
        self.data_config = data_config

    def prepare_data(self, example) -> None:
        print()
        assert 1 == 0

    def setup(self, stage: str) -> None:
        if stage == "fit":
            print(self.dataset["train"])
            print("fit")

        elif stage == "test":
            print("test")

        assert 1 == 0

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=default_data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["val"],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=default_data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=default_data_collator,
        )

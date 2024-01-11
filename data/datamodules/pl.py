import lightning as L
from torch.utils.data import DataLoader, Dataset

# import pytorch_lightning as pl


from transformers import default_data_collator


class DataModule(L.LightningDataModule):
    def __init__(self, args, dataset: Dataset, data_config: dict) -> None:
        super().__init__()

        self.dataset = dataset
        self.args = args
        self.data_config = data_config

    # def prepare_data(self) -> None:
    #     pass

    def setup(self, stage: str) -> None:
        # 아 맘에 안든다
        # 데이터셋 별로 이렇게 해야하나
        if self.args.data == "jeanlee/kmhas_korean_hate_speech":
            from data.datasets.kmhas.kmhas import preprocess

            remove_columns = []

        if stage == "fit":
            datas = ["train", "val"]

        elif stage == "test":
            datas = ["test"]

        for data in datas:
            self.dataset[data] = self.dataset[data].map(
                preprocess,
                batched=True,
                remove_columns=remove_columns,
                fn_kwargs={
                    "tokenizer": self.data_config["tokenizer"],
                    "args": self.args,
                },
                load_from_cache_file=False,
                desc="Data Pre-processing",
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.args.trainer["batch_size"],
            # num_workers=self.args.num_workers, 이 arg를 어디 추가하는게 맞을까?
            num_workers=8,
            collate_fn=default_data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["val"],
            batch_size=self.args.trainer["batch_size"],
            # num_workers=self.args.num_workers,
            num_workers=8,
            collate_fn=default_data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.args.trainer["batch_size"],
            # num_workers=self.args.num_workers,
            num_workers=8,
            collate_fn=default_data_collator,
        )

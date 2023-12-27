"""
여기 데이터셋별로 datamodule
from ? import ? 
"""

# from pytorch_lightning import LightningDataModule ???? 라이트닝이 데이터모듈도 있나?


"""
    task에 따라서 다른 데이터 모듈
"""


def get_datamodule(args):
    if args.data == "jeanlee/kmhas_korean_hate_speech":
        from .kmhas import _load_dataset

        datasets = _load_dataset()

        print(datasets)

    data_module = None
    assert 1 == 0

    return data_module

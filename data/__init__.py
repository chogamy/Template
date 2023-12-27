"""
여기 데이터셋별로 datamodule
from ? import ? 
"""

# from pytorch_lightning import LightningDataModule ???? 라이트닝이 데이터모듈도 있나?


"""
    task에 따라서 다른 데이터 모듈
"""


def get_datamodule(args):
    if args.dataset == "?":
        # from ? import ? as datamodule
        data_module = ""
        pass

    return data_module

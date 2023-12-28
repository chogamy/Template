"""
task에 따라서 다른 데이터 모듈
"""


def preprocess(dataset, args):
    """ """
    args

    return dataset


def get_datamodule(args):
    if args.data == "jeanlee/kmhas_korean_hate_speech":
        from .datasets.kmhas.kmhas import _load_dataset

    datasets = _load_dataset()
    """
    DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: ????
        })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: ????
        })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: ????
        })
    })
    """

    if args.wrapper == "PL":
        from .datamodules.pl import DataModule
    elif args.wrapper in ["HF", None]:
        from .datamodules.custom import DataModule

    data_module = DataModule(datasets, args)
    assert 1 == 0

    return data_module

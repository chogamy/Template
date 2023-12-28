def preprocess(dataset, args):
    """
    task에 따라서 다른 전처리
    어떻게 해야 할지 나중에 좀 생각하자
    """
    if args.task == "e1c1":
        dataset = dataset.map(
            lambda example: {
                "input": example["text"],
                "label": example["label"],
            },
        )
    else:
        raise NotImplementedError

    return dataset


def get_datamodule(args):
    if args.data == "jeanlee/kmhas_korean_hate_speech":
        from .datasets.kmhas.kmhas import _load_dataset, get_dataconfig

    dataset = _load_dataset()
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

    data_config = get_dataconfig(args)
    dataset = preprocess(dataset, args)

    if args.wrapper == "PL":
        from .datamodules.pl import DataModule
    elif args.wrapper in ["HF", None]:
        from .datamodules.custom import DataModule

    data_module = DataModule(args, dataset, data_config)

    return data_module

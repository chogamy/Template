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
    val: Dataset({
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

    if args.wrapper == "lightning.pytorch":
        from .datamodules.pl import DataModule
    elif args.wrapper in ["transformers", "custom"]:
        from .datamodules.custom import DataModule

    data_module = DataModule(args, dataset, data_config)

    return data_module

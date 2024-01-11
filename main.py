from args import build_args
from nn import get_model
from data import get_datamodule
from trainer import get_trainer


if __name__ == "__main__":
    args = build_args()

    datamodule = get_datamodule(args)
    model = get_model(args, datamodule.data_config)
    trainer = get_trainer(args)

    trainer.run(model, datamodule)

    if args.mode == "analysis":
        print("analysis")
    else:
        raise ValueError("Invalid mode")

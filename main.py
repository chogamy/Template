from args import build_args
from nn import get_model
from data import get_datamodule
from trainer import get_trainer


if __name__ == "__main__":
    args = build_args()

    datamodule = get_datamodule(args)
    model = get_model(args, datamodule.data_config)
    trainer = get_trainer(args)
    
    assert 1==0

    if args.mode == "train":
        trainer.train(model, datamodule)
    elif args.mode == "infer":
        print("infer")
    elif args.mode == "analysis":
        print("analysis")
    else:
        raise ValueError("Invalid mode")


# def get_callbacks(callback_list):
#     callbacks = []
#     for callback in callback_list:
#         callback_args = {
#             callback["class_path"].split(".")[-1]: callback.get("init_args", None)
#         }

#         # TODO: too naive :(
#         if "EarlyStopping" in callback_args:
#             early_stopping = EarlyStopping(**callback_args["EarlyStopping"])
#             callbacks.append(early_stopping)

#         if "ModelCheckpoint" in callback_args:
#             model_checkpoint = ModelCheckpoint(**callback_args["ModelCheckpoint"])
#             callbacks.append(model_checkpoint)

#         if "RichProgressBar" in callback_args:
#             progress_bar = RichProgressBar()
#             callbacks.append(progress_bar)

#     return callbacks

# def train():
#     args = get_args()

#     config = get_configurable_parameters(args=args)

#     seed_everything(config.seed, workers=True)

#     datamodule = get_datamodule(config)
#     datamodule.prepare_data()

#     config.model.init_args.num_classes = datamodule.num_classes
#     model = get_model(config)
#     callbacks = get_callbacks(config.trainer.pop("callbacks", []))

#     trainer = Trainer(**config.trainer, callbacks=callbacks)

#     if config.mode == "train":
#         trainer.fit(model=model, datamodule=datamodule)
#     if config.mode == "test":
#         trainer.test(model=model, datamodule=datamodule)
#     if config.mode == "visual":

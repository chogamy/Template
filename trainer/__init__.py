import os
import yaml
from importlib import import_module


class TrainerWrapper:
    def __init__(self, trainer, args) -> None:
        self.trainer = trainer
        self.args = args

    def run(self, model, datamodule):
        pass

    def train(self, model, datamodule):
        if self.args.wrapper == "PL":
            self.trainer.fit(model, datamodule)
        elif self.args.wrapper == "HF":
            pass
        elif self.args.wrapper == None:
            pass

    def eval(self):
        pass

    def test(self):
        pass


def build_trainerargs(args):
    path = os.path.join("args", "trainer")

    wrapper = os.path.join(path, args.wrapper)
    trainer = os.path.join(path, args.trainer)
    lrscheduler = os.path.join(path, "lrschedulers", args.lrscheduler)
    optimizers = os.path.join(path, "optimizers", args.optimizers)
    callbacks = [
        os.path.join(path, "callbacks", f"{callback}.yaml")
        for callback in args.callbacks.split(",")
    ]

    with open(f"{wrapper}.yaml") as f:
        wrapper = yaml.load(f, Loader=yaml.FullLoader)
        trainer_args = {**wrapper}

    with open(f"{trainer}.yaml") as f:
        trainer = yaml.load(f, Loader=yaml.FullLoader)

    with open(f"{lrscheduler}.yaml") as f:
        lrscheduler = yaml.load(f, Loader=yaml.FullLoader)

    with open(f"{optimizers}.yaml") as f:
        optimizers = yaml.load(f, Loader=yaml.FullLoader)

    if args.wrapper == "lightning.pytorch":
        trainer_args["default_root_dir"] = args.model_path
        trainer_args["max_epochs"] = trainer["epoch"]

        optimizers

        # callbacks
        _callbacks = []
        for callback in callbacks:
            if "earlystopping" in callback:
                from lightning.pytorch.callbacks import EarlyStopping

                with open(callback) as f:
                    _callback = yaml.load(f, Loader=yaml.FullLoader)
                    _callback = EarlyStopping(**_callback)
                    _callbacks.append(_callback)

            if "modelcheckpoint" in callback:
                from lightning.pytorch.callbacks import ModelCheckpoint

                with open(callback) as f:
                    _callback = yaml.load(f, Loader=yaml.FullLoader)
                    _callback = ModelCheckpoint(**_callback)
                    _callbacks.append(_callback)

        callbacks = _callbacks
        trainer_args["callbacks"] = callbacks

    return trainer_args

    return {
        "trainer_args": trainer_args,
        "trainer": trainer,
        "lrscheduler": lrscheduler,
        "optimizers": optimizers,
        "callbacks": callbacks,
    }


def get_trainer(args):
    trainer_module = import_module(f"{args.wrapper}")

    trainer_args = build_trainerargs(args)

    trainer = getattr(trainer_module, "Trainer")(**trainer_args)

    return TrainerWrapper(trainer, args)

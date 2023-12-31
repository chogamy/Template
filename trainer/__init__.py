# 이 패키지 안에 hf.py, pl.py, custom.py가 있어야 하나?

class TrainerWrapper:
    def __init__(self, trainer, args) -> None:
        self.trainer = trainer
        self.args = args

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

def get_trainerargs(args):
    pass



def get_trainer(args):
    if args.wrapper == "PL":
        from lightning.pytorch import Trainer
    elif args.wrapper == "HF":
        from transformers import Trainer
    elif args.wrapper == None:
        from .custom import Trainer

    
    trainer_args = args.trainer

    print(args.trainer)
    print(args.lrscheduler)
    print(args.optimizers)
    print(args.callbacks)

    assert 1==0
        

    trainer = Trainer()
        

    return TrainerWrapper(trainer, args)

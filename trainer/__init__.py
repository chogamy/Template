class TrainerWrapper:
    def __init__(self, trainer, args) -> None:
        self.trainer = trainer
        self.args = args

    def train(self):
        if self.args.wrapper == "pytorch_lightning":
            pass
        elif self.args.wrapper == "pytorch":
            pass
        elif self.args.wrapper == None:
            pass

    def eval(self):
        pass

    def test(self):
        pass


def get_trainer(args):
    """
    wrapper에 따라서 학습..?
    pytorch lighting
    그냥 파이토치
    그냥 쌩으로
    허깅페이스
    이런걸루?
    """
    if args.wrapper == "PL":
        """
        import pytorch_lightning as Trainer
        trainer = Trainer()
        """

        trainer = ""
        pass

    return TrainerWrapper(trainer, args)

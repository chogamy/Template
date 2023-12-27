'''
쌩으로 트레이닝
'''


class Trainer():
    def __init__(self, model, datamodule, args): 
        self.model = model
        self.datamodule = datamodule
        self.args = args
    
    def train(self):
        pass

    def eval(self):
        pass

    def test(self):
        pass
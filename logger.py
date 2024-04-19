from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, exptId):
        self.exptId = exptId
        self.writer = SummaryWriter("runs/" + exptId)

    def log(self, quantity, value, index):
        self.writer.add_scalar(quantity, value, index)

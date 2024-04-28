from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, exptId, path='runs/'):
        self.exptId = exptId
        self.writer = SummaryWriter(path + exptId)

    def log(self, quantity, value, index):
        self.writer.add_scalar(quantity, value, index)

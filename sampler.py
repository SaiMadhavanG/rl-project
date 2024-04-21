import random


class Sampler:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self, buffer):
        assert buffer.size >= self.batch_size
        probs = buffer.getProbabilities()
        choices = random.choices(range(buffer.size), weights=probs, k=self.batch_size)
        batch = [buffer.chunks[i] for i in choices]
        return batch

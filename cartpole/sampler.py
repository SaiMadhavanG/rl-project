import random


class Sampler:
    def __init__(self, batch_size):
        self.batch_size = int(batch_size)

    def sample(self, buffer):
        assert buffer.size >= self.batch_size
        if len(buffer.chunks) < self.batch_size:
            raise Exception("Not enough chunks to sample")
        probs = buffer.getProbabilities()
        choices = random.choices(range(buffer.size), weights=probs, k=self.batch_size)
        batch = [buffer.chunks[i] for i in choices]
        for i in choices:
            buffer.chunks[i].timesSampled += 1
        return batch

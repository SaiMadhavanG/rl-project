import random
import numpy as np

class Sampler:
    def __init__(self, batch_size):
        self.batch_size = int(batch_size)

    def sample(self, buffer, b):
        assert buffer.size >= self.batch_size
        if len(buffer.chunks) < self.batch_size:
            raise Exception("Not enough chunks to sample")
        probs = buffer.getProbabilities()
        choices = random.choices(range(buffer.size), weights=probs, k=self.batch_size)
        batch = [buffer.chunks[i] for i in choices]
        probs_choices = [probs[i] for i in choices]
        chunk_IS_weights = self.calc_IS_weights(b, buffer.size, probs_choices)
        for i in choices:
            buffer.chunks[i].timesSampled += 1
        return batch, chunk_IS_weights

    def calc_IS_weights(self, b, buffer_size, probs):
        probs_np = np.array(probs)
        return list(((1 / buffer_size) * (1 / probs_np)) ** b)
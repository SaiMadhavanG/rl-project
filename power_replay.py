from sampler import Sampler
from transition_chunk import Chunk


class PowerReplay:
    def __init__(
        self, buffer, weight_assigner, weight_factors, batch_size, chunk_size
    ) -> None:
        self.buffer = buffer
        self.weight_assigner = weight_assigner
        self.weight_factors = weight_factors
        self.batch_size = batch_size
        self.sampler = Sampler(batch_size)
        self.chunk_size = chunk_size

    def getBatch(self):
        return self.sampler.sample(self.buffer)

    def addTransitions(self, transitions, episode_id):
        while len(transitions) > self.chunk_size:
            chunk = Chunk(
                self.chunk_size, episode_id, _transitions=transitions[: self.chunk_size]
            )
            self.buffer.addChunk(chunk)
            for i in range(self.chunk_size):
                transitions.pop(0)
        chunk = Chunk(self.chunk_size, episode_id, _transitions=transitions)
        self.buffer.addChunk(chunk)

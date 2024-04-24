from sampler import Sampler
from transition_chunk import Chunk
from weight_assigner import Weight_assigner, UniformAssigner
from replay_buffer import ReplayBuffer


class PowerReplay:
    def __init__(self, size, batch_size, chunk_size, weight_factors, mode) -> None:
        # TODO move buffer initialization in here
        self.size = size
        self.buffer = ReplayBuffer(size)
        if mode == "uniform":
            self.weight_assigner = UniformAssigner(self.buffer)
        else:
            raise Exception("Implementation pending")
        # TODO initialize weight Assigner using
        self.weight_factors = weight_factors
        self.batch_size = batch_size
        self.sampler = Sampler(batch_size)
        self.chunk_size = chunk_size

    def samplable(self):
        return len(self.buffer.chunks) >= self.size

    def getBatch(self):
        chunks = self.sampler.sample(self.buffer)
        transitions = []
        for chunk in chunks:
            transitions.extend(chunk.transitions)
        return transitions

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
        self.weight_assigner.set_probablities()
        
        
    # new method to add the chunk to the buffer
    def addChunk(self, chunk):
        if (chunk.isComplete() == 2) : chunk.pad_transitions()
        self.buffer.addChunk(chunk)
        self.weight_assigner.set_probablities()

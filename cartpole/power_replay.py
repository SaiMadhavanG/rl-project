from sampler import Sampler
from transition_chunk import Chunk
from weight_assigner import Weight_assigner, UniformAssigner
from replay_buffer import ReplayBuffer
from tracker import Tracker


class PowerReplay:
    def __init__(self, size, batch_size, chunk_size, weight_factors, mode) -> None:
        # TODO move buffer initialization in here
        self.size = size
        self.buffer = ReplayBuffer(size)
        if mode == "uniform":
            self.weight_assigner = UniformAssigner(self.buffer)
        elif mode == "tde":
            self.weight_assigner = Weight_assigner(
                self.buffer, _tde_factor=weight_factors["tde_alpha"]
            )
        elif mode == "returns":
            self.weight_assigner = Weight_assigner(
                self.buffer,
                _estimated_return_factor=weight_factors["estimatedReturn_alpha"],
            )
        elif mode == "combination":
            self.weight_assigner = Weight_assigner(
                self.buffer,
                _tde_factor=weight_factors["tde_alpha"],
                _reward_factor=weight_factors["rewards_alpha"],
                _estimated_return_factor=weight_factors["estimatedReturn_alpha"],
            )
        elif mode == "rarity":
            self.buffer = ReplayBuffer(size, _frequency_hist_ranges=weight_factors["frequency_hist_ranges"])
            self.buffer._rarity = True
            self.buffer._rarity_alpha=weight_factors["rarity_alpha"]
            self.weight_assigner = Weight_assigner(
                self.buffer
            )
        else:
            raise Exception("Implementation pending")
        # TODO initialize weight Assigner using
        self.weight_factors = weight_factors
        self.batch_size = batch_size
        self.sampler = Sampler(batch_size)
        self.chunk_size = chunk_size
        self.chunk_counter = 0

    def samplable(self):
        return len(self.buffer.chunks) >= self.size

    def getBatch(self):
        chunks = self.sampler.sample(self.buffer)
        transitions = []
        for chunk in chunks:
            transitions.extend(chunk.transitions)
        return transitions, chunks

    def addTransitions(self, transitions, episode_id):
        # TODO handle chunk id for when more transitions
        while len(transitions) > self.chunk_size:
            raise Exception("Given transitions more than chunk size")
            # chunk = Chunk(
            #     self.chunk_size, episode_id, _transitions=transitions[: self.chunk_size]
            # )
            # self.buffer.addChunk(chunk)
            # for i in range(self.chunk_size):
            #     transitions.pop(0)
        chunk = Chunk(
            self.chunk_size, self.chunk_counter, episode_id, _transitions=transitions
        )
        self.buffer.addChunk(chunk)
        self.chunk_counter += 1

    def sweep(self, tracker: Tracker):
        chunk_ids = tracker.modified
        self.weight_assigner.set_weights(chunk_ids)
        tracker.modified = []

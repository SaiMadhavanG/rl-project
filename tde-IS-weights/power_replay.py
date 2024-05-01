from sampler import Sampler
from transition_chunk import Chunk
from weight_assigner import Weight_assigner, UniformAssigner
from replay_buffer import ReplayBuffer
from tracker import Tracker
import torch
import numpy as np


class PowerReplay:
    def __init__(self, size, batch_size, chunk_size, weight_factors, mode) -> None:
        # TODO move buffer initialization in here
        self.size = int(size)
        self.buffer = ReplayBuffer(size)
        if mode == "uniform":
            self.weight_assigner = UniformAssigner(self.buffer)
        elif mode == "tde":
            self.weight_assigner = Weight_assigner(
                self.buffer,
                _tde_factor=weight_factors["tde_alpha"],
                _trace_factor=weight_factors["trace_factor"],
                _trace_length=weight_factors["trace_length"],
                _staleness_factor=weight_factors["staleness_alpha"],
            )
        elif mode == "rewards":
            self.weight_assigner = Weight_assigner(
                self.buffer,
                _reward_factor=weight_factors["rewards_alpha"],
                _trace_factor=weight_factors["trace_factor"],
                _trace_length=weight_factors["trace_length"],
                _staleness_factor=weight_factors["staleness_alpha"],
            )
        elif mode == "returns":
            self.weight_assigner = Weight_assigner(
                self.buffer,
                _estimated_return_factor=weight_factors["estimatedReturn_alpha"],
            )
        elif mode == "staleness":
            self.weight_assigner = Weight_assigner(
                self.buffer, _staleness_factor=weight_factors["staleness_alpha"]
            )
        elif mode == "combination":
            self.weight_assigner = Weight_assigner(
                self.buffer,
                _tde_factor=weight_factors["tde_alpha"],
                _reward_factor=weight_factors["rewards_alpha"],
                _estimated_return_factor=weight_factors["estimatedReturn_alpha"],
                _trace_factor=weight_factors["trace_factor"],
                _trace_length=weight_factors["trace_length"],
                _staleness_factor=weight_factors["staleness_alpha"],
            )
        else:
            raise Exception("Implementation pending")
        # TODO initialize weight Assigner using
        self.weight_factors = weight_factors
        self.batch_size = batch_size
        self.sampler = Sampler(batch_size)
        self.chunk_size = chunk_size
        self.chunk_counter = 0
        self.lastSampled = np.zeros(self.size, dtype=int)
        self.tempTransitions = []

    def samplable(self):
        return len(self.buffer.chunks) >= self.size

    def getBatch(self, b=0.0):
        chunks, chunk_IS_weights = self.sampler.sample(self.buffer, b=b)
        transition_IS_weights = torch.repeat_interleave(
            torch.tensor(chunk_IS_weights, dtype=torch.float), self.chunk_size
        )
        transitions = []
        init_id = self.buffer.chunks[0].chunk_id
        for chunk in chunks:
            transitions.extend(chunk.transitions)
            idx = chunk.chunk_id - init_id
            self.lastSampled[idx] = self.chunk_counter
        return transitions, chunks, transition_IS_weights

    def addTransition(self, transition, episode_id):
        self.tempTransitions.append(transition)
        if len(self.tempTransitions) == self.chunk_size + 1:
            self.tempTransitions.pop(0)
        chunk = Chunk(
            self.chunk_size,
            self.chunk_counter,
            episode_id,
            _transitions=self.tempTransitions,
        )
        chunk.pad_transitions()
        self.buffer.addChunk(chunk)
        self.lastSampled[:-1] = self.lastSampled[1:]
        self.lastSampled[-1] = int(self.lastSampled.mean())
        self.chunk_counter += 1

    def sweep(self, tracker: Tracker):
        chunk_ids = tracker.modified
        self.weight_assigner.set_weights(
            chunk_ids, self.lastSampled, self.chunk_counter - 1
        )
        tracker.modified = []

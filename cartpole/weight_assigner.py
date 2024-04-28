# will have access to the replay buffer. Will pass through the buffer and set the weight and probablity parameter of each chunk.
# Will set the probablity and weight attributes of chunks
from replay_buffer import ReplayBuffer
from transition_chunk import Chunk


class Weight_assigner:

    def __init__(
        self,
        _replay_buffer: ReplayBuffer,
        iteration_num=0,
        _tde_factor=0,
        _reward_factor=0,
        _estimated_return_factor=0,
        _fr_ratio_current_factor=0,
        _fr_ratio_global_factor=0,
        _trace_factor=0,
        _staleness_factor=0,
    ):

        self.replay_buffer = _replay_buffer
        self.iteration_num = iteration_num
        self.tde_factor = _tde_factor
        self.reward_factor = _reward_factor
        self.estimated_return_factor = _estimated_return_factor
        self.fr_ratio_current_factor = _fr_ratio_current_factor
        self.fr_ratio_global_factor = _fr_ratio_global_factor
        self.trace_factor = _trace_factor
        self.staleness_factor = _staleness_factor

    # Function for each factor  : currently simplistic
    def tde_func(self, tde):
        return tde**self.tde_factor

    def reward_func(self, reward):
        return reward**self.reward_factor

    def estimated_return_func(self, estimated_return):
        return estimated_return**self.estimated_return_factor

    def fr_ratio_current_func(self, fr_ratio_current):
        return (
            self.fr_ratio_current_factor * fr_ratio_current
        )  # TODO : should be inversely. So update this function

    def fr_ratio_global_func(self, fr_ratio_global):
        return (
            self.fr_ratio_global_factor * fr_ratio_global
        )  # TODO : should be inversely. So update this function

    def lastSampled_func(self, lastSampled):
        staleness = self.iteration_num - lastSampled
        return self.staleness_factor * staleness

    def trace_func(self, trace_weight):
        return self.trace_factor * trace_weight

    # calculates the weight without considering the trace got by the successor
    def without_trace_weight(self, chunk: Chunk):
        return (
            self.tde_func(chunk.tde)
            + self.reward_func(chunk.rewards)
            + self.estimated_return_func(chunk.estimated_return)
            + self.fr_ratio_current_func(chunk.frequency_ration_current)
            + self.fr_ratio_global_func(chunk.frequency_ration_global)
            + self.lastSampled_func(chunk.lastSampled)
        )

    # Assigns the weight to every chunk in the replay buffer
    def set_weights(self, chunks):
        # trace_multiplier = self.trace_factor
        # replay_size = len(self.replay_buffer.chunks)

        # # The latest chunk wont have any trace
        # current_chunk = self.replay_buffer.chunks[-1]
        # current_chunk.set_weight(self.without_trace_weight(current_chunk))

        # for i in range(-1, -replay_size, -1):
        #     current_chunk = self.replay_buffer.chunks[i]
        #     previous_chunk = self.replay_buffer.chunks[i - 1]

        #     previous_chunk_without_trace_weight = self.without_trace_weight(
        #         previous_chunk
        #     )

        #     if previous_chunk.episode_id == current_chunk.episode_id:
        #         previous_chunk.set_weight(
        #             previous_chunk_without_trace_weight
        #             + self.trace_func(trace_multiplier * current_chunk.weight)
        #         )
        #     else:
        #         previous_chunk.set_weight(previous_chunk_without_trace_weight)
        init_id = self.replay_buffer.chunks[0].chunk_id
        for chunk in chunks:
            weight = self.without_trace_weight(chunk)
            chunk.set_weight(weight)
            print(f"this is the weight of the chunk : {chunk.weight}")
            idx = chunk.chunk_id - init_id
            if idx >= 0:
                self.replay_buffer.weights[idx] = weight

    # Method to sweep the replay buffer and assign the probablity to each chunk
    # probablity = chunk weight/ summation of weight of each chunk present in the replay buffer


class UniformAssigner:
    def __init__(self, _replay_buffer: ReplayBuffer):
        self.replay_buffer = _replay_buffer

    def set_weights(self, *args):
        pass

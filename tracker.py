# Methods to make the attributes of the chunks up to date
from replay_buffer import ReplayBuffer
from transition_chunk import Chunk


class Tracker:

    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer

    """
        Single chunk based assigner methods
        No sweeps required
        For all of this the value to be assigned has to be provided by the user of this class
    """

    def set_tde(self, chunk: Chunk, tde):
        chunk.set_tde(tde)

    def set_estimated_return(self, chunk: Chunk, estimated_return):
        chunk.set_rewards(estimated_return)

    def set_rewards(self, chunk: Chunk, rewards):
        chunk.set_rewards(rewards)

    def set_lastSampled(self, chunk: Chunk, lastSampled):
        chunk.set_lastSampled(lastSampled)

    """
        Complete replay buffer sweep based assigner methods
    """

    # pass through all the chunks and set the frequency ration for each of them
    def set_frequency_ratio(self, frequency_mode="global"):
        for chunk in self.replay_buffer.chunks:

            if frequency_mode == "current":
                fr_ratio = self.replay_buffer.frequencyRatio(chunk)
                chunk.set_frequency_ratio_current(fr_ratio)
            else:
                fr_ratio = self.replay_buffer.frequencyRatio(chunk)
                chunk.set_frequency_ratio_global(fr_ratio)

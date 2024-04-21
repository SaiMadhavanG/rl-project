from collections import OrderedDict
import numpy as np


class FrequencyHistogram:
    def __init__(self, n_dim, ranges) -> None:
        self.n_dim = n_dim
        self.ranges = ranges
        assert n_dim == len(ranges)
        self.dicts = [OrderedDict() for i in range(self.n_dim)]
        for i in range(len(self.dicts)):
            assert len(self.ranges[i]) == 3
            for j in np.arange(self.ranges[i][0], self.ranges[i][1], self.ranges[i][2]):
                self.dicts[i][j] = 0

    def __getitem__(self, _keys):
        assert len(_keys) == self.n_dim
        result = [0] * self.n_dim
        for i in range(self.n_dim):
            prevKey = list(self.dicts[i].keys())[0]
            for key in self.dicts[i].keys():
                if _keys[i] < key:
                    result[i] = self.dicts[i][prevKey]
                    break
                prevKey = key
            else:
                result[i] = self.dicts[i][prevKey]

        return result

    def __setitem__(self, _keys, _values):
        assert len(_keys) == self.n_dim
        assert len(_values) == self.n_dim
        for i in range(self.n_dim):
            prevKey = list(self.dicts[i].keys())[0]
            for key in self.dicts[i].keys():
                if _keys[i] < key:
                    self.dicts[i][prevKey] = _values[i]
                    break
                prevKey = key
            else:
                self.dicts[i][prevKey] = _values[i]

    def addOne(self, _keys):
        res = self[_keys]
        newRes = [i + 1 for i in res]
        self[_keys] = newRes

    def removeOne(self, _keys):
        res = self[_keys]
        newRes = [i - 1 for i in res]
        self[_keys] = newRes

    def total(self):
        return [sum(list(dict.values())) for dict in self.dicts]


class ReplayBuffer:
    def __init__(
        self,
        _size,
        _chunks=[],
        n_dim=None,
        _frequency_hist_ranges=[],
        _frequency_mode="current",
    ) -> None:
        """
        n_dim is state vector size needed for making histogram
        _frequency_hist_range should be a tuple containing 3 values (start, end, step)
        """
        self.size = int(_size)
        self.chunks = list(_chunks)

        self.frequency_histogram = (
            FrequencyHistogram(n_dim, _frequency_hist_ranges)
            if _frequency_hist_ranges
            else None
        )
        self.frequncy_mode = _frequency_mode

    def addChunk(self, chunk):
        self.chunks.append(chunk)
        if self.frequency_histogram:
            state_v = chunk.getAvgState()
            self.frequency_histogram.addOne(state_v)

        while len(self.chunks) > self.size:
            self.removeChunk()

    def removeChunk(self):
        chunk = self.chunks.pop(0)
        if self.frequncy_mode == "current" and self.frequency_histogram:
            state_v = chunk.getAvgState()
            self.frequency_histogram.removeOne(state_v)

    def frequencyRatio(self, chunk):
        if self.frequency_histogram:
            state_v = chunk.getAvgState()
            freq = self.frequency_histogram[state_v]
            tot = self.frequency_histogram.total()
            ratio = []
            for i in range(len(freq)):
                ratio.append(freq[i] / tot[i])
            return ratio
        else:
            raise Exception("Frequency histogram not initialized")

    def getProbabilities(self):
        return [chunk.probablity for chunk in self.chunks]

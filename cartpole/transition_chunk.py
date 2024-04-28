# Note : transitions are stored as a list
# Note : while adding or setting transitions the limitation of size of chunk is not there. After all the transitions are added the pad_transitions method should be called to pad the chunk using replicas or to remove extra transitions. This make the size of the transitiions list same as the chunk size.
class Chunk:

    # size has to be given while creating the  object
    def __init__(
        self,
        _size,
        _chunk_id,
        _episode_id=None,
        # this was set to allow the uniform mode. As previously total_weight = 0 was causing division errors.
        _weight=1,
        _tde=0,
        _rewards=0,
        _estimated_return=0,
        _lastSampled=0,
        _IS_weight=0,
        _transitions=[],
        _probablity=0,
        _frequency_ratio_current=0,
        _frequency_ratio_global=0,
    ):
        self.size = _size
        self.chunk_id = _chunk_id
        self.transitions = _transitions
        self.episode_id = _episode_id
        self.weight = _weight
        self.tde = _tde
        self.rewards = _rewards
        self.estimated_return = _estimated_return
        self.lastSampled = _lastSampled
        self.IS_weight = _IS_weight
        self.probablity = _probablity
        self.frequency_ration_current = _frequency_ratio_current
        self.frequency_ration_global = _frequency_ratio_global
        self.timesSampled = 0

    # SETTERS FOR ALL ATTRIBUTES
    def set_episode_id(self, _episode_id):
        self.episode_id = _episode_id

    def set_weight(self, _weight):
        self.weight = _weight

    def set_transitions(self, _transitions):
        self.transitions = _transitions

    def set_tde(self, _tde):
        self.tde = _tde

    def set_rewards(self, _rewards):
        self.rewards = _rewards

    def set_estimated_return(self, _estimated_return):
        self.estimated_return = _estimated_return

    def set_lastSampled(self, _lastSampled):
        self.lastSampled = _lastSampled

    def set_IS_weight(self, _IS_weight):
        self.IS_weight = _IS_weight

    def set_frequency_ratio_current(self, fr_ratio_current):
        self.frequency_ration_current = fr_ratio_current

    def set_frequency_ratio_global(self, fr_ratio_global):
        self.frequency_ration_global = fr_ratio_global

    def set_probablity(self, _probablity):
        self.probablity = _probablity

    # method to use when you are adding transitions one by one
    def add_transition(self, transition):
        self.transitions.append(transition)

    # if number of transitions less than size then pad by duplicating the last transition
    # if number of transitions more than size then remove the extra transition from the front
    def pad_transitions(self):
        trans_len = len(self.transitions)
        if self.size < trans_len:
            n_remove = trans_len - self.size
            self.transitions = self.transitions[n_remove:]

        elif self.size > trans_len:
            last_transition = self.transitions[-1]
            n_replicas = self.size - trans_len
            replicas = [last_transition] * n_replicas
            self.transitions.extend(replicas)

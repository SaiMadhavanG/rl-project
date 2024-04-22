class Transition:
    def __init__(self, state, action, reward, nextState, terminated=False):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.terminated = terminated

    def getState(self):
        return self.state

    def getAction(self):
        return self.action

    def getReward(self):
        return self.reward

    def getNextState(self):
        return self.nextState

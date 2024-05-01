class Transition:
    def __init__(self, state, action, reward, nextState, terminated=False):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.terminated = terminated
        self.tde = 0
        self.estimated_return = 0

    def getState(self):
        return self.state

    def getAction(self):
        return self.action

    def getReward(self):
        return self.reward

    def getNextState(self):
        return self.nextState

    def setTDE(self, tde):
        self.tde = tde

    def setEstimatedReturn(self, estimated_return):
        self.estimated_return = estimated_return

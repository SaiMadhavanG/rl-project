import network
import numpy as np
from copy import deepcopy
import torch


class DQNAgent:
    def __init__(self, numActions, stateDim, m=4, device="cpu"):
        self.numActions = numActions
        self.device = device
        self.stateDim = stateDim
        self.network = network.createMLP(stateDim, numActions, device)
        self.targetNetwork = deepcopy(self.network)

    def selectAction(self, state, network="target", epsilon=0):
        net = self.targetNetwork if network == "target" else self.network
        net.eval()
        rand = np.random.random()
        if rand < epsilon:
            return np.random.choice(self.numActions)
        Q = net(state.to(self.device))
        return torch.argmax(Q[0]).item()


class DQNAtariAgent(DQNAgent):
    def __init__(self, numActions, m=4, device="cpu"):
        self.numActions = numActions
        self.device = device
        self.network = network.createCNN(m, numActions, device)
        self.targetNetwork = deepcopy(self.network)

    def selectAction(self, state, network="target", epsilon=0):
        if state.shape != (4, 84, 84):
            raise Exception(
                f"Input dimension should be (4, 84, 84), received state.shape"
            )
        super().selectAction(state, network, epsilon)

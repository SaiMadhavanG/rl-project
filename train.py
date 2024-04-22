from agent_trainer import AgentTrainer
from agent import DQNAgent
from environment import CartPoleEnvironment
from logger import Logger
from power_replay import PowerReplay
import torch
import pdb

device = "cuda"

agent = DQNAgent(2, 4, device=device)
env = CartPoleEnvironment()
logger = Logger("cf-6")
optimizer = torch.optim.Adam(agent.network.parameters(), lr=1e-3)
powerReplay = PowerReplay(600, 128, 1, [], "uniform")

trainer = AgentTrainer(
    agent,
    env,
    logger,
    optimizer,
    powerReplay,
    device=device,
    epsilon=0.1,
    epsilon_decay=0.999,
    tau=1,
    gamma=1,
)

# pdb.set_trace()
trainer.train_steps(1e4)

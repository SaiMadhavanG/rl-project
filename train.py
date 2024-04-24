from agent_trainer import AgentTrainer
from agent import DQNAgent
from environment import CartPoleEnvironment
from environment import MountainCarEnvironment
from logger import Logger
from power_replay import PowerReplay
import torch

device = "cpu"

agent = DQNAgent(2, 4, device=device)
env = CartPoleEnvironment()
render_env = CartPoleEnvironment("")
logger = Logger("CartPole-t9")
optimizer = torch.optim.Adam(agent.network.parameters(), lr=1e-3)
powerReplay = PowerReplay(10000, 64, 1, [], "uniform")

trainer = AgentTrainer(
    agent,
    env,
    logger,
    optimizer,
    powerReplay,
    device=device,
    epsilon=0.5,
    min_epsilon=0.1,
    epsilon_decay=5e-4,
    gamma=1,
    update_frequency=4,
)

# pdb.set_trace()
trainer.train_steps(1e6)

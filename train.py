from agent_trainer import AgentTrainer
from agent import DQNAgent
from environment import CartPoleEnvironment
from logger import Logger
from power_replay import PowerReplay
import torch

device = "cuda"

agent = DQNAgent(2, 4, device=device)
env = CartPoleEnvironment()
render_env = CartPoleEnvironment("human")
logger = Logger("cf-17")
optimizer = torch.optim.Adam(agent.network.parameters(), lr=1e-3)
powerReplay = PowerReplay(1e4, 128, 1, [], "uniform")

trainer = AgentTrainer(
    agent,
    env,
    logger,
    optimizer,
    powerReplay,
    device=device,
    epsilon=0.3,
    epsilon_decay=2e-3,
    gamma=0.99,
    render_env=render_env,
    update_frequency=8,
)

# pdb.set_trace()
trainer.train_steps(1e4)

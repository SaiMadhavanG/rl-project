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
logger = Logger("cf-14")
optimizer = torch.optim.Adam(agent.network.parameters(), lr=1e-3)
powerReplay = PowerReplay(5e4, 256, 1, [], "uniform")

trainer = AgentTrainer(
    agent,
    env,
    logger,
    optimizer,
    powerReplay,
    device=device,
    epsilon=0.50,
    epsilon_decay=3e-3,
    tau=0.1,
    gamma=1,
    render_env=render_env,
)

# pdb.set_trace()
trainer.train_steps(5e4)

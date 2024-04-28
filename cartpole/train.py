from agent_trainer import AgentTrainer
from agent import DQNAgent
from environment import CartPoleEnvironment
from logger import Logger
from power_replay import PowerReplay
import torch

device = "cpu"

agent = DQNAgent(2, 4, device=device)
env = CartPoleEnvironment()
render_env = CartPoleEnvironment("human")
logger = Logger("Ricky-cartpole-tde-reward-06-06")
optimizer = torch.optim.Adam(agent.network.parameters(), lr=1e-3)
powerReplay = PowerReplay(5e3, 32, 1, {
                          "tde_alpha": 0.6, "rewards_alpha": 0.6, "estimatedReturn_alpha": 0.6}, "tde")

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
    update_frequency=5,
)

# pdb.set_trace()
trainer.train_steps(1e5)

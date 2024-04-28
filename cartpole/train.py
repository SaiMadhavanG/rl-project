from agent_trainer import AgentTrainer
from agent import DQNAgent
from environment import CartPoleEnvironment, AcrobotEnvironment
from logger import Logger
from power_replay import PowerReplay
import torch

device = "cpu"

agent = DQNAgent(2, 4, device=device)
env = CartPoleEnvironment()
render_env = CartPoleEnvironment("human")

# env = AcrobotEnvironment()
# render_env = AcrobotEnvironment("human")

# logger = Logger(
#     "Ricky-acrobot-uniform-outside-epsilons-07-009-4e4-factors-0-0-0")
logger = Logger(
    "bug_testing")
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
    epsilon=0.7,
    min_epsilon=0.09,
    epsilon_decay=4e-4,
    gamma=1,
    update_frequency=4,
)

# pdb.set_trace()
trainer.train_steps(2e5)

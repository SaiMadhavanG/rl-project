from agent_trainer import AgentTrainer
from agent import DQNAgent
from environment import CartPoleEnvironment, LunarLanderEnvironment
from logger import Logger
from power_replay import PowerReplay
import torch

device = "cuda"

agent = DQNAgent(4, 8, device=device)
env = LunarLanderEnvironment()
render_env = LunarLanderEnvironment("render")
logger = Logger("Lunar_Lander_tde_IS_baseline", path="../cartpole/runs/")
optimizer = torch.optim.Adam(agent.network.parameters(), lr=1e-3)
powerReplay = PowerReplay(5e3, 32, 1, {"tde_alpha": 0.6}, "tde")

trainer = AgentTrainer(
    agent,
    env,
    logger,
    optimizer,
    powerReplay,
    device=device,
    epsilon=0.7,
    min_epsilon=0.1,
    epsilon_decay=8e-4,
    gamma=1,
    update_frequency=5,
)

# pdb.set_trace()
trainer.train_steps(2e5)

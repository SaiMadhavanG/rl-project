from agent_trainer import AgentTrainer
from agent import DQNAgent
from environment import CartPoleEnvironment, LunarLanderEnvironment
from logger import Logger
from power_replay import PowerReplay
import torch

device = "cuda"

agent = DQNAgent(2, 4, device=device)
env = CartPoleEnvironment()
render_env = CartPoleEnvironment("human")
logger = Logger("cartpole-chunk-4")
optimizer = torch.optim.Adam(agent.network.parameters(), lr=5e-4)
powerReplay = PowerReplay(
    5e3,
    8,
    4,
    {"trace_factor": 0.2, "trace_length": 20, "tde_alpha": 0.6, "staleness_alpha": 0.1},
    "tde",
)

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

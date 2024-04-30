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
logger = Logger("Lunar_Lander_returns_only_without_IS")
optimizer = torch.optim.Adam(agent.network.parameters(), lr=1e-3)
powerReplay = PowerReplay(5e3, 32, 1, {
                                        "rarity_alpha": 2, 
                                        "frequency_hist_ranges": [(i, j, 100) for i, j in zip(env.env.observation_space.low.tolist(), env.env.observation_space.high.tolist())]
                                      }, "rarity")
# powerReplay = PowerReplay(5e3, 32, 1, {"estimatedReturn_alpha": 1}, "returns")

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
trainer.train_steps(1e5)

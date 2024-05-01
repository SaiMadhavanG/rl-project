from distutils.log import Log
from agent import DQNAgent
from agent_trainer import AgentTrainer
from environment import CartPoleEnvironment
from power_replay import PowerReplay
from logger import Logger

device = "cuda"


checkpoint_path = "C:/programming/projects/rl-project/tde-IS-weights/checkpoints/cartpole-uniform-uf5/checkpoint-400.pth"
render_env = CartPoleEnvironment("human")
agent = DQNAgent(2, 4, device=device)
logger = Logger("eval")


powerReplay = PowerReplay(
    5e3,
    16,
    2,
    {"trace_factor": 0.2, "trace_length": 20, "tde_alpha": 0.6, "staleness_alpha": 0.1},
    "tde",
)
trainer = AgentTrainer(
    agent, render_env, logger, None, powerReplay, device=device, render_env=render_env
)
trainer.load(checkpoint_path)
scores = []
for i in range(10):
    scores.append(trainer.inference_mode())

print(f"Average score: {sum(scores)/len(scores):0.3f}")
